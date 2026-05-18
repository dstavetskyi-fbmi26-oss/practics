# experiment_v3_cross_subject.py
# ГІПОТЕЗА: LDA з TD/FD ознаками стійкіший до cross-subject variability ніж CNN-LSTM.
# МЕТОД:    Тренувати на N-1 суб'єктах, тестувати на 1 (Leave-One-Out).
#           Це реальний сценарій: модель навчена на одних людях, застосовується до нового.

import argparse
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from emg_proposals import (
    load_ninapro, butterworth_filter,
    segment_record, normalize_windows, filter_labeled,
    build_feature_matrix, build_cnn_lstm, EMGRecord,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def load_subject(sid):
    return load_ninapro('datasets/NinaproDB8_Dataset',
                        subjects=[sid], skip_rest=True)


def prepare_windows(record, step_ms=100):
    """Повертає (X, y) вікна для одного суб'єкта."""
    filtered = butterworth_filter(record)
    X, y = segment_record(filtered, window_ms=200, step_ms=step_ms)
    X = normalize_windows(X)
    return filter_labeled(X, y)


def run_lda(X_train, y_train, X_test, y_test):
    X_feat_tr = np.nan_to_num(build_feature_matrix(X_train, fs=2000))
    X_feat_te = np.nan_to_num(build_feature_matrix(X_test,  fs=2000))
    pipe = Pipeline([('sc', StandardScaler()), ('lda', LDA())])
    pipe.fit(X_feat_tr, y_train)
    y_pred = pipe.predict(X_feat_te)
    return f1_score(y_test, y_pred, average='macro', zero_division=0)


def run_cnn(X_train, y_train, X_test, y_test, n_channels, n_classes,
            epochs=40, batch_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = build_cnn_lstm(n_channels, X_train.shape[2], n_classes)

    X_tr_pt = torch.tensor(X_train, dtype=torch.float32)
    y_tr_pt = torch.tensor(y_train, dtype=torch.long)
    X_te_pt = torch.tensor(X_test,  dtype=torch.float32)
    y_te_pt = torch.tensor(y_test,  dtype=torch.long)

    pin = device.type == 'cuda'
    loader = DataLoader(TensorDataset(X_tr_pt, y_tr_pt),
                        batch_size=batch_size, shuffle=True,
                        pin_memory=pin, num_workers=0)
    val_loader = DataLoader(TensorDataset(X_te_pt, y_te_pt),
                            batch_size=batch_size, pin_memory=pin, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_f1, best_state, patience = 0.0, None, 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            nn.CrossEntropyLoss()(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds.append(model(xb.to(device)).argmax(1).cpu())
        y_pred_ep = torch.cat(preds).numpy()
        ep_f1 = f1_score(y_test, y_pred_ep, average='macro', zero_division=0)

        if ep_f1 > best_f1:
            best_f1   = ep_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience  = 0
        else:
            patience += 1
            if patience >= 7:
                break

    model.load_state_dict(best_state)
    return best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects', type=int, nargs='+', default=None,
                        help='Суб\'єкти для LOO (default: 1–5)')
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()

    subject_ids = args.subjects or list(range(1, 6))

    print("=" * 62)
    print("  ЕКСПЕРИМЕНТ V3: Leave-One-Out cross-subject")
    print("  Тренування на N-1 суб'єктах → тест на 1 новому")
    print("=" * 62)
    print(f"\nСуб'єкти: {subject_ids}")
    print(f"Стратегія: для кожного суб'єкта i:\n"
          f"  train = всі інші суб'єкти\n"
          f"  test  = суб'єкт i\n")

    print("Завантаження...")
    subjects_data = {}
    for sid in subject_ids:
        rec = load_subject(sid)
        X, y = prepare_windows(rec)
        subjects_data[sid] = (X, y, rec.n_channels, len(np.unique(y)))
        print(f"  S{sid}: {X.shape[0]} вікон, {len(np.unique(y))} класів")

    n_channels = subjects_data[subject_ids[0]][2]
    n_classes  = subjects_data[subject_ids[0]][3]

    results = {}

    for test_sid in subject_ids:
        print(f"\n── LOO: test=S{test_sid}, train={[s for s in subject_ids if s != test_sid]}")

        X_trains, y_trains = [], []
        for sid in subject_ids:
            if sid != test_sid:
                X_trains.append(subjects_data[sid][0])
                y_trains.append(subjects_data[sid][1])

        if not X_trains:
            print("  [!] Потрібно >= 2 суб'єкти"); continue

        X_train = np.concatenate(X_trains)
        y_train = np.concatenate(y_trains)
        X_test, y_test = subjects_data[test_sid][0], subjects_data[test_sid][1]

        # Вирівняти класи (можуть бути різні після skip_rest)
        common = sorted(set(y_train) & set(y_test))
        if len(common) < 2:
            print(f"  [!] Замало спільних класів ({len(common)})"); continue

        mask_tr = np.isin(y_train, common)
        mask_te = np.isin(y_test,  common)
        X_train, y_train = X_train[mask_tr], y_train[mask_tr]
        X_test,  y_test  = X_test[mask_te],  y_test[mask_te]
        remap = {c: i for i, c in enumerate(common)}
        y_train = np.array([remap[c] for c in y_train])
        y_test  = np.array([remap[c] for c in y_test])

        print(f"  Train: {len(y_train)} вікон  |  Test: {len(y_test)} вікон")

        lda_f1 = run_lda(X_train, y_train, X_test, y_test)
        cnn_f1 = run_cnn(X_train, y_train, X_test, y_test,
                         n_channels, len(common), epochs=args.epochs)

        results[test_sid] = {'lda_f1': lda_f1, 'cnn_f1': cnn_f1}
        winner = 'CNN ✓' if cnn_f1 > lda_f1 else 'LDA ✓'
        print(f"  LDA F1: {lda_f1:.3f}  |  CNN F1: {cnn_f1:.3f}  |  {winner}")

    if not results:
        return

    sids    = list(results.keys())
    lda_f1s = [results[s]['lda_f1'] for s in sids]
    cnn_f1s = [results[s]['cnn_f1'] for s in sids]
    avg_lda = np.mean(lda_f1s)
    avg_cnn = np.mean(cnn_f1s)
    cnn_wins = sum(1 for s in sids if results[s]['cnn_f1'] > results[s]['lda_f1'])

    print("\n" + "─" * 52)
    print(f"  {'Test S':>7} {'LDA F1':>8} {'CNN F1':>8} {'Δ':>8}")
    print("─" * 52)
    for s in sids:
        d = results[s]['cnn_f1'] - results[s]['lda_f1']
        print(f"  {s:>7} {results[s]['lda_f1']:>8.3f} {results[s]['cnn_f1']:>8.3f} {d:>+8.3f}")
    print("─" * 52)
    print(f"  {'СЕРЕДНЄ':>7} {avg_lda:>8.3f} {avg_cnn:>8.3f}  CNN виграє {cnn_wins}/{len(sids)}")
    print()
    if avg_lda > avg_cnn:
        print("  → LDA стійкіший до нових суб'єктів (cross-subject).")
    else:
        print("  → CNN-LSTM краще узагальнює між суб'єктами.")

    # Графік
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'V3: Leave-One-Out cross-subject  (Ninapro DB8)  —  '
                 f'CNN виграє {cnn_wins}/{len(sids)}', fontweight='bold')

    x, w = np.arange(len(sids)), 0.35
    axes[0].bar(x - w/2, lda_f1s, w, label='LDA',     color='darkorange', alpha=0.85)
    axes[0].bar(x + w/2, cnn_f1s, w, label='CNN-LSTM', color='steelblue',  alpha=0.85)
    axes[0].axhline(avg_lda, color='darkorange', ls='--', lw=1.2, alpha=0.7)
    axes[0].axhline(avg_cnn, color='steelblue',  ls='--', lw=1.2, alpha=0.7)
    axes[0].set_xticks(x); axes[0].set_xticklabels([f'test=S{s}' for s in sids], fontsize=8)
    axes[0].set_title('LOO F1: тестовий суб\'єкт')
    axes[0].set_ylabel('F1'); axes[0].set_ylim(0, 1)
    axes[0].legend(); axes[0].grid(axis='y', alpha=0.3)

    diff   = [c - l for c, l in zip(cnn_f1s, lda_f1s)]
    colors = ['#2ecc71' if d >= 0 else '#e74c3c' for d in diff]
    axes[1].bar([f'S{s}' for s in sids], diff, color=colors, alpha=0.85)
    axes[1].axhline(0, color='black', lw=0.8)
    axes[1].axhline(np.mean(diff), color='purple', ls='--', lw=1.2,
                    label=f'середнє Δ={np.mean(diff):+.3f}')
    axes[1].set_title('CNN − LDA (>0 = CNN краще на новому суб\'єкті)')
    axes[1].set_ylabel('Δ F1'); axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fname = 'exp_v3_cross_subject_loo.png'
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    print(f"  Графік: {fname}")
    plt.show()


if __name__ == '__main__':
    main()
