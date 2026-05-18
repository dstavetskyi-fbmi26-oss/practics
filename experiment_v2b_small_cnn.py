# experiment_v2b_small_cnn.py
# ГІПОТЕЗА: CNN-LSTM програє LDA через невідповідність архітектури розміру датасету.
# МЕТОД:    Per-subject (80/20), порівняння 4 підходів:
#   1. LDA              — baseline (ручні ознаки)
#   2. CNN original     — ~120k параметрів (з V2)
#   3. CNN lightweight  — ~11k параметрів
#   4. CNN light + aug  — lightweight + Gaussian noise + amplitude scaling + time shift

import argparse
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from emg_proposals import (
    load_ninapro, butterworth_filter,
    segment_record, normalize_windows,
    filter_labeled, build_feature_matrix,
)

class CNNLSTMOriginal(nn.Module):
    """Оригінальна архітектура з emg_proposals (~120k параметрів)."""
    def __init__(self, n_ch, T, n_cls):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_ch, 64,  3, padding=1), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64,  128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.3),
        )
        self.lstm1 = nn.LSTM(128, 128, batch_first=True, dropout=0.3)
        self.lstm2 = nn.LSTM(128,  64, batch_first=True)
        self.head  = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_cls)
        )

    def forward(self, x):
        x = self.cnn(x).transpose(1, 2)
        x, _ = self.lstm1(x); x, _ = self.lstm2(x)
        return self.head(x[:, -1])


class CNNLSTMLight(nn.Module):
    """
    Легка архітектура (~11k параметрів).
    Розрахована на малі датасети (~800 прикладів на train).
    """
    def __init__(self, n_ch, T, n_cls):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_ch, 16, 3, padding=1), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16,   32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.2),
        )
        self.lstm = nn.LSTM(32, 32, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, n_cls)
        )

    def forward(self, x):
        x = self.cnn(x).transpose(1, 2)
        x, _ = self.lstm(x)
        return self.head(x[:, -1])


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def augment_emg(X: np.ndarray, y: np.ndarray,
                n_copies: int = 3,
                noise_factor: float = 0.05,
                scale_range: tuple = (0.8, 1.2),
                max_shift: int = 20) -> tuple:
    """
    Збільшує датасет у (1 + n_copies) разів через три трансформації:
      • Gaussian noise    — додає реалістичний шум (~5% від std сигналу)
      • Amplitude scaling — масштабує амплітуду [0.8, 1.2] (різна сила скорочення)
      • Time shift        — зсув сигналу по часу [-20, +20] семплів (затримка реакції)

    X: (W, C, T)  →  X_aug: (W*(1+n_copies), C, T)
    """
    rng = np.random.default_rng(0)
    X_all, y_all = [X], [y]

    for _ in range(n_copies):
        Xa = X.copy()

        # Gaussian noise — пропорційний до локального std кожного вікна
        sig_std  = Xa.std(axis=-1, keepdims=True).clip(min=1e-6)
        Xa      += rng.normal(0, noise_factor, Xa.shape) * sig_std

        # Amplitude scaling — одне значення на вікно (різна сила скорочення)
        scale    = rng.uniform(*scale_range, size=(len(Xa), 1, 1)).astype(np.float32)
        Xa      *= scale

        # Time shift — circular roll по часовій осі
        shifts   = rng.integers(-max_shift, max_shift + 1, size=len(Xa))
        Xa       = np.stack([np.roll(Xa[i], shifts[i], axis=-1)
                             for i in range(len(Xa))])

        X_all.append(Xa.astype(np.float32))
        y_all.append(y)

    return np.concatenate(X_all), np.concatenate(y_all)


def train_cnn(ModelClass, X_train, y_train, X_val, y_val,
              epochs, batch_size=128, label='CNN'):
    """Загальна функція тренування для будь-якої архітектури."""
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_ch, T = X_train.shape[1], X_train.shape[2]
    n_cls   = len(np.unique(y_train))

    model = ModelClass(n_ch, T, n_cls).to(device)
    print(f"  {label}: {count_params(model):,} параметрів  →  ", end='', flush=True)

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_va = torch.tensor(X_val,   dtype=torch.float32)
    y_va = torch.tensor(y_val,   dtype=torch.long)

    pin = device.type == 'cuda'
    tr_loader  = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size,
                            shuffle=True, pin_memory=pin, num_workers=0)
    val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=batch_size,
                            pin_memory=pin, num_workers=0)

    crit  = nn.CrossEntropyLoss()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, factor=0.5)

    best_f1, best_state, patience_cnt = 0.0, None, 0

    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in val_loader:
                preds.append(model(xb.to(device)).argmax(1).cpu())
        y_pred_ep = torch.cat(preds).numpy()
        ep_f1 = f1_score(y_val, y_pred_ep, average='macro', zero_division=0)
        sched.step(1 - ep_f1)

        if ep_f1 > best_f1:
            best_f1   = ep_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= 8:
                break

    print(f"F1={best_f1:.3f}  (зупинився на epoch {ep+1})")
    return best_f1


def run_lda(X_train, y_train, X_val, y_val, fs):
    X_tr_f = np.nan_to_num(build_feature_matrix(X_train, fs))
    X_va_f = np.nan_to_num(build_feature_matrix(X_val,   fs))
    pipe   = Pipeline([('sc', StandardScaler()), ('lda', LDA())])
    pipe.fit(X_tr_f, y_train)
    f1 = f1_score(y_val, pipe.predict(X_va_f), average='macro', zero_division=0)
    print(f"  LDA:              {len(X_tr_f[0])} ознак  →  F1={f1:.3f}")
    return f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects', type=int, nargs='+', default=None)
    parser.add_argument('--epochs',   type=int, default=60)
    parser.add_argument('--aug',      type=int, default=3,
                        help='Кількість копій augmentation (default: 3 → датасет ×4)')
    args = parser.parse_args()

    subject_ids = args.subjects or list(range(1, 6))

    print("=" * 62)
    print("  ЕКСПЕРИМЕНТ V2b: Lightweight CNN + Augmentation")
    print("=" * 62)
    print(f"\nСуб'єкти: {subject_ids}  |  epochs: {args.epochs}  |  aug: ×{args.aug+1}")
    print(f"\n  Порівняння 4 підходів per-subject:\n"
          f"  1. LDA              (ручні TD/FD ознаки)\n"
          f"  2. CNN original     (~120k параметрів)\n"
          f"  3. CNN lightweight  (~11k параметрів)\n"
          f"  4. CNN light + aug  (~11k + augmentation ×{args.aug+1})\n")

    all_results = {}

    for sid in subject_ids:
        print(f"\n{'─'*55}")
        print(f"  Суб'єкт {sid}")
        print(f"{'─'*55}")

        record  = load_ninapro('datasets/NinaproDB8_Dataset',
                               subjects=[sid], skip_rest=True)
        filt    = butterworth_filter(record)
        X, y    = segment_record(filt, window_ms=200, step_ms=100)
        X       = normalize_windows(X)
        X, y    = filter_labeled(X, y)

        if len(y) < 30:
            print("  [!] Замало даних"); continue

        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        print(f"  Train: {len(y_tr)} вікон  |  Val: {len(y_va)} вікон  "
              f"|  Класів: {len(np.unique(y))}")

        # Аугментовані дані для CNN light+aug
        X_tr_aug, y_tr_aug = augment_emg(X_tr, y_tr, n_copies=args.aug)
        print(f"  Augmented train: {len(y_tr_aug)} вікон (×{args.aug+1})")

        res = {}
        t0  = time.perf_counter()

        res['lda']        = run_lda(X_tr, y_tr, X_va, y_va, record.fs)
        res['cnn_orig']   = train_cnn(CNNLSTMOriginal, X_tr,     y_tr,     X_va, y_va,
                                      args.epochs, label='CNN original  ')
        res['cnn_light']  = train_cnn(CNNLSTMLight,    X_tr,     y_tr,     X_va, y_va,
                                      args.epochs, label='CNN lightweight')
        res['cnn_aug']    = train_cnn(CNNLSTMLight,    X_tr_aug, y_tr_aug, X_va, y_va,
                                      args.epochs, label='CNN light+aug  ')

        res['elapsed'] = time.perf_counter() - t0
        all_results[sid] = res

        best = max(res, key=lambda k: res[k] if k != 'elapsed' else -1)
        print(f"\n  Переможець: {best}  (F1={res[best]:.3f})  |  "
              f"Час: {res['elapsed']:.0f}s")

    if not all_results:
        return

    sids     = list(all_results.keys())
    methods  = ['lda', 'cnn_orig', 'cnn_light', 'cnn_aug']
    labels   = ['LDA', 'CNN orig', 'CNN light', 'CNN+aug']

    print(f"\n{'═'*68}")
    print(f"  {'S':>3}  {'LDA':>8}  {'CNN orig':>9}  {'CNN light':>10}  {'CNN+aug':>8}  Переможець")
    print(f"{'═'*68}")
    for s in sids:
        r      = all_results[s]
        scores = {m: r[m] for m in methods}
        best_m = max(scores, key=scores.get)
        best_l = labels[methods.index(best_m)]
        print(f"  {s:>3}  {r['lda']:>8.3f}  {r['cnn_orig']:>9.3f}  "
              f"{r['cnn_light']:>10.3f}  {r['cnn_aug']:>8.3f}  {best_l}")

    print(f"{'─'*68}")
    avgs = {m: np.mean([all_results[s][m] for s in sids]) for m in methods}
    best_avg = max(avgs, key=avgs.get)
    print(f"  {'avg':>3}  {avgs['lda']:>8.3f}  {avgs['cnn_orig']:>9.3f}  "
          f"{avgs['cnn_light']:>10.3f}  {avgs['cnn_aug']:>8.3f}  "
          f"→ {labels[methods.index(best_avg)]}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('V2b: Lightweight CNN + Augmentation  per-subject  (Ninapro DB8)',
                 fontweight='bold')
    cmap = ['#e67e22', '#2980b9', '#27ae60', '#8e44ad']

    x = np.arange(len(sids)); w = 0.2
    for i, (m, lbl, c) in enumerate(zip(methods, labels, cmap)):
        scores = [all_results[s][m] for s in sids]
        axes[0].bar(x + (i-1.5)*w, scores, w, label=lbl, color=c, alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'S{s}' for s in sids])
    axes[0].set_title('F1 macro по суб\'єктах')
    axes[0].set_ylabel('F1'); axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=9); axes[0].grid(axis='y', alpha=0.3)


    avg_scores = [avgs[m] for m in methods]
    bars = axes[1].bar(labels, avg_scores, color=cmap, alpha=0.85, width=0.5)
    axes[1].set_title('Середнє F1 macro')
    axes[1].set_ylabel('F1'); axes[1].set_ylim(0, 1)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].axhline(1/9, color='gray', ls='--', lw=1, label='random (1/9)')
    axes[1].legend(fontsize=9)
    for bar, val in zip(bars, avg_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.01,
                     f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

    delta_aug = avgs['cnn_aug'] - avgs['cnn_orig']
    delta_lda = avgs['cnn_aug'] - avgs['lda']
    conclusion = (
        f"CNN light+aug vs CNN orig: {delta_aug:+.3f}\n"
        f"CNN light+aug vs LDA:      {delta_lda:+.3f}"
    )
    axes[1].text(0.98, 0.04, conclusion, transform=axes[1].transAxes,
                 ha='right', va='bottom', fontsize=8,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fname = 'exp_v2b_small_cnn_ninapro.png'
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    print(f"\n  Графік: {fname}")
    plt.show()


if __name__ == '__main__':
    main()
