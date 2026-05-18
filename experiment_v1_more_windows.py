# experiment_v1_more_windows.py
# ГІПОТЕЗА: CNN-LSTM програє LDA через замало вікон (cross-subject, step=100ms).
# МЕТОД:    Збільшити кількість вікон через зменшення step (більший overlap).
#           Порівняти LDA vs CNN-LSTM при різних step_ms.

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
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import f1_score
import torch


def run_one_step(record, step_ms, epochs=30):
    filtered = butterworth_filter(record)

    # LDA
    X, y = segment_record(filtered, window_ms=200, step_ms=step_ms)
    X = normalize_windows(X); X, y = filter_labeled(X, y)
    n_windows = len(y)
    X_feat = np.nan_to_num(build_feature_matrix(X, record.fs))
    pipe = Pipeline([('sc', StandardScaler()), ('lda', LDA())])
    lda_f1 = cross_val_score(pipe, X_feat, y,
                             cv=StratifiedKFold(5, shuffle=True, random_state=42),
                             scoring='f1_macro').mean()

    # CNN-LSTM
    t0 = time.perf_counter()
    ai = run_ai_pipeline(record, window_ms=200, step_ms=step_ms, epochs=epochs)
    elapsed = time.perf_counter() - t0

    ai_acc = ai['val_accuracy']
    ai_f1  = f1_score(ai['y_val'], ai['y_pred'], average='macro', zero_division=0)

    return {
        'step_ms':    step_ms,
        'n_windows':  n_windows,
        'lda_f1':     lda_f1,
        'cnn_acc':    ai_acc,
        'cnn_f1':     ai_f1,
        'elapsed':    elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',  default='ninapro', choices=['wyoflex', 'ninapro'])
    parser.add_argument('--patients', default='all')
    parser.add_argument('--subjects', default='all')
    parser.add_argument('--epochs',   type=int, default=30)
    args = parser.parse_args()

    patients = None if str(args.patients).lower() == 'all' else int(args.patients)
    subjects = None if str(args.subjects).lower() == 'all' else int(args.subjects)

    print("=" * 60)
    print("  ЕКСПЕРИМЕНТ V1: Вплив кількості вікон на точність CNN")
    print("=" * 60)

    if args.dataset == 'ninapro':
        record = load_ninapro('datasets/NinaproDB8_Dataset',
                              subjects=list(range(1, (subjects or 10) + 1)),
                              skip_rest=True)
    else:
        record = load_wyoflex('datasets/WyoFlex_Dataset',
                              max_patients=patients)

    print(f"\nДатасет: {record.dataset}  |  {record.n_channels} кан  "
          f"|  {record.duration_sec:.0f}s  |  {len(record.class_names)} класи")

    step_values = [100, 50, 25]
    results = []

    for step in step_values:
        overlap = 100 * (1 - step / 200)
        print(f"\n── step={step}ms ({overlap:.0f}% overlap) ──────────────────")
        r = run_one_step(record, step_ms=step, epochs=args.epochs)
        results.append(r)
        print(f"  Вікон: {r['n_windows']:5d}  |  LDA F1: {r['lda_f1']:.3f}  "
              f"|  CNN F1: {r['cnn_f1']:.3f}  |  Час: {r['elapsed']:.0f}s")

    # Підсумкова таблиця
    print("\n" + "─" * 60)
    print(f"  {'step_ms':>8} {'вікон':>8} {'overlap':>8} {'LDA F1':>8} {'CNN F1':>8}")
    print("─" * 60)
    for r in results:
        ov = f"{100*(1-r['step_ms']/200):.0f}%"
        print(f"  {r['step_ms']:>8} {r['n_windows']:>8} {ov:>8} "
              f"{r['lda_f1']:>8.3f} {r['cnn_f1']:>8.3f}")

    # Графік
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f'V1: Вплив overlap на CNN vs LDA  ({record.dataset.upper()})',
                 fontweight='bold')

    steps = [r['step_ms'] for r in results]
    windows = [r['n_windows'] for r in results]
    lda_scores = [r['lda_f1'] for r in results]
    cnn_scores = [r['cnn_f1'] for r in results]

    # Кількість вікон
    axes[0].bar([f"{s}ms" for s in steps], windows, color='steelblue', alpha=0.8)
    axes[0].set_title('Кількість вікон')
    axes[0].set_ylabel('вікон')
    for i, v in enumerate(windows):
        axes[0].text(i, v + max(windows)*0.01, str(v), ha='center', fontsize=9)

    # F1 порівняння
    x = np.arange(len(steps))
    w = 0.35
    axes[1].bar(x - w/2, lda_scores, w, label='LDA',      color='darkorange', alpha=0.85)
    axes[1].bar(x + w/2, cnn_scores, w, label='CNN-LSTM',  color='steelblue',  alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"{s}ms" for s in steps])
    axes[1].set_title('F1 macro: LDA vs CNN-LSTM'); axes[1].set_ylabel('F1')
    axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(axis='y', alpha=0.3)
    axes[1].axhline(1/len(record.class_names), color='gray', ls='--', lw=1,
                    label='random baseline')

    # Різниця CNN - LDA
    diff = [c - l for c, l in zip(cnn_scores, lda_scores)]
    colors = ['green' if d >= 0 else 'red' for d in diff]
    axes[2].bar([f"{s}ms" for s in steps], diff, color=colors, alpha=0.8)
    axes[2].axhline(0, color='black', lw=0.8)
    axes[2].set_title('CNN F1 − LDA F1 (>0 = CNN краще)')
    axes[2].set_ylabel('Δ F1')
    axes[2].grid(axis='y', alpha=0.3)
    for i, d in enumerate(diff):
        axes[2].text(i, d + (0.005 if d >= 0 else -0.012), f'{d:+.3f}',
                     ha='center', fontsize=9)

    plt.tight_layout()
    fname = f'exp_v1_more_windows_{record.dataset}.png'
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    print(f"\n  Графік: {fname}")
    plt.show()


if __name__ == '__main__':
    main()
