# experiment_v2_per_subject.py
# ГІПОТЕЗА: CNN-LSTM виграє у LDA коли тренується і тестується на одному суб'єкті
#           (немає cross-subject variability).
# МЕТОД:    Для кожного суб'єкта окремо: завантажити → split 80/20 → LDA + CNN-LSTM.
#           Показати per-subject результати і порівняти з cross-subject (V3).
#

import argparse
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from emg_proposals import (
    load_wyoflex, load_ninapro,
    butterworth_filter, segment_record, normalize_windows,
    filter_labeled, build_feature_matrix, run_ai_pipeline,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import torch


def run_single_subject(record, epochs=40):
    filtered = butterworth_filter(record)
    X, y = segment_record(filtered, window_ms=200, step_ms=100)
    X = normalize_windows(X); X, y = filter_labeled(X, y)
    n_windows = len(y)

    if len(np.unique(y)) < 2 or n_windows < 20:
        return None  # недостатньо даних

    # LDA — train/test split (той самий що й для CNN)
    X_feat = np.nan_to_num(build_feature_matrix(X, record.fs))
    X_tr_f, X_val_f, y_tr, y_val = train_test_split(
        X_feat, y, test_size=0.2, stratify=y, random_state=42
    )
    pipe = Pipeline([('sc', StandardScaler()), ('lda', LDA())])
    pipe.fit(X_tr_f, y_tr)
    lda_f1  = f1_score(y_val, pipe.predict(X_val_f), average='macro', zero_division=0)
    lda_acc = accuracy_score(y_val, pipe.predict(X_val_f))

    # CNN-LSTM
    t0 = time.perf_counter()
    ai = run_ai_pipeline(record, window_ms=200, step_ms=100, epochs=epochs)
    elapsed = time.perf_counter() - t0
    cnn_f1  = f1_score(ai['y_val'], ai['y_pred'], average='macro', zero_division=0)
    cnn_acc = ai['val_accuracy']

    return {
        'n_windows':  n_windows,
        'lda_f1':     lda_f1,
        'lda_acc':    lda_acc,
        'cnn_f1':     cnn_f1,
        'cnn_acc':    cnn_acc,
        'elapsed':    time.perf_counter() - t0 + elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',  default='ninapro', choices=['wyoflex', 'ninapro'])
    parser.add_argument('--subjects', type=int, nargs='+', default=None,
                        help='Список суб\'єктів Ninapro (default: 1–5)')
    parser.add_argument('--patients', type=int, nargs='+', default=None,
                        help='Список пацієнтів WyoFlex (default: 1–5)')
    parser.add_argument('--epochs',   type=int, default=40)
    args = parser.parse_args()

    print("=" * 60)
    print("  ЕКСПЕРИМЕНТ V2: Per-subject (CNN у своєму середовищі)")
    print("=" * 60)

    if args.dataset == 'ninapro':
        subject_ids = args.subjects or list(range(1, 6))
    else:
        subject_ids = args.patients or list(range(1, 6))

    print(f"\nДатасет: {args.dataset}  |  Суб'єкти: {subject_ids}")
    print(f"Стратегія: кожен суб'єкт окремо, 80% train / 20% test\n")

    all_results = {}

    for sid in subject_ids:
        print(f"── Суб'єкт {sid} ──────────────────────────────────")

        if args.dataset == 'ninapro':
            record = load_ninapro('datasets/NinaproDB8_Dataset',
                                  subjects=[sid], skip_rest=True)
        else:
            record = load_wyoflex('datasets/WyoFlex_Dataset',
                                  max_patients=1,
                                  movements=list(range(1, 6)))
            # (wyoflex не підтримує конкретний ID, беремо перших N)

        res = run_single_subject(record, epochs=args.epochs)
        if res is None:
            print(f"  [!] Недостатньо даних — пропускаємо")
            continue

        all_results[sid] = res
        print(f"  Вікон: {res['n_windows']:4d}  |  "
              f"LDA F1: {res['lda_f1']:.3f}  |  CNN F1: {res['cnn_f1']:.3f}  |  "
              f"{'CNN краще ✓' if res['cnn_f1'] > res['lda_f1'] else 'LDA краще'}")

    if not all_results:
        print("Немає результатів."); return

    # Зведена таблиця
    sids   = list(all_results.keys())
    lda_f1 = [all_results[s]['lda_f1'] for s in sids]
    cnn_f1 = [all_results[s]['cnn_f1'] for s in sids]

    avg_lda = np.mean(lda_f1)
    avg_cnn = np.mean(cnn_f1)

    print("\n" + "─" * 55)
    print(f"  {'Суб''єкт':>9} {'Вікон':>7} {'LDA F1':>8} {'CNN F1':>8} {'Переможець':>12}")
    print("─" * 55)
    for s in sids:
        r = all_results[s]
        winner = 'CNN ✓' if r['cnn_f1'] > r['lda_f1'] else 'LDA ✓'
        print(f"  {s:>9} {r['n_windows']:>7} {r['lda_f1']:>8.3f} "
              f"{r['cnn_f1']:>8.3f} {winner:>12}")
    print("─" * 55)
    cnn_wins = sum(1 for s in sids if all_results[s]['cnn_f1'] > all_results[s]['lda_f1'])
    print(f"  {'СЕРЕДНЄ':>9} {'':>7} {avg_lda:>8.3f} {avg_cnn:>8.3f}  "
          f"CNN виграє {cnn_wins}/{len(sids)}")

    # Графік
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'V2: Per-subject  ({args.dataset.upper()})  —  '
                 f'CNN виграє {cnn_wins}/{len(sids)} суб\'єктів', fontweight='bold')

    x = np.arange(len(sids))
    w = 0.35

    # F1 per subject
    axes[0].bar(x - w/2, lda_f1, w, label='LDA',     color='darkorange', alpha=0.85)
    axes[0].bar(x + w/2, cnn_f1, w, label='CNN-LSTM', color='steelblue',  alpha=0.85)
    axes[0].axhline(avg_lda, color='darkorange', ls='--', lw=1.2, alpha=0.7)
    axes[0].axhline(avg_cnn, color='steelblue',  ls='--', lw=1.2, alpha=0.7)
    axes[0].set_xticks(x); axes[0].set_xticklabels([f'S{s}' for s in sids])
    axes[0].set_title('F1 macro по суб\'єктах')
    axes[0].set_ylabel('F1'); axes[0].set_ylim(0, 1)
    axes[0].legend(); axes[0].grid(axis='y', alpha=0.3)

    # Різниця CNN - LDA
    diff   = [c - l for c, l in zip(cnn_f1, lda_f1)]
    colors = ['#2ecc71' if d >= 0 else '#e74c3c' for d in diff]
    axes[1].bar([f'S{s}' for s in sids], diff, color=colors, alpha=0.85)
    axes[1].axhline(0, color='black', lw=0.8)
    axes[1].axhline(np.mean(diff), color='purple', ls='--', lw=1.2,
                    label=f'середнє Δ={np.mean(diff):+.3f}')
    axes[1].set_title('CNN F1 − LDA F1 (>0 = CNN краще)')
    axes[1].set_ylabel('Δ F1'); axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    for i, d in enumerate(diff):
        axes[1].text(i, d + (0.005 if d >= 0 else -0.015),
                     f'{d:+.2f}', ha='center', fontsize=8)

    plt.tight_layout()
    fname = f'exp_v2_per_subject_{args.dataset}.png'
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    print(f"\n  Графік: {fname}")
    plt.show()


if __name__ == '__main__':
    main()
