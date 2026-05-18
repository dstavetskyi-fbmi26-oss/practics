from __future__ import annotations
import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew
from dataclasses import dataclass, field
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


@dataclass
class EMGRecord:
    """Уніфікований формат для всіх датасетів."""
    signal: np.ndarray          # (n_channels, n_samples)
    labels: np.ndarray          # (n_samples,) — int клас; -1 якщо немає мітки
    fs: int                     # частота дискретизації, Гц
    dataset: str                # "wyoflex" | "ninapro" | "custom"
    class_names: list[str] = field(default_factory=list)

    @property
    def n_channels(self) -> int:
        return self.signal.shape[0]

    @property
    def duration_sec(self) -> float:
        return self.signal.shape[1] / self.fs
    

WYOFLEX_MOVEMENTS = {
    1: 'flexion',       2: 'extension',    3: 'ulnar_dev',
    4: 'radial_dev',    5: 'hook_grip',    6: 'power_grip',
    7: 'spherical_grip', 8: 'precision_grip', 9: 'lateral_grip',
    10: 'pinch_grip',
}

def load_wyoflex(root: str | Path,
                 movements: list[int] | None = None,
                 cycle: int = 1,
                 forearm: int = 1,
                 offset: int = 1,
                 use_voltage: bool = True,
                 max_patients: int | None = None) -> EMGRecord:
    import re
    root = Path(root)
    data_dir = root / ('VOLTAGE DATA' if use_voltage else 'DIGITAL DATA')
    if not data_dir.exists():
        data_dir = root
    movements = movements or list(WYOFLEX_MOVEMENTS.keys())
    patients = sorted({
        int(m.group(1))
        for f in data_dir.iterdir()
        if f.is_file() and (m := re.search(r'^P(\d+)', f.name))
    })
    if max_patients:
        patients = patients[:max_patients]

    all_signals, all_labels = [], []

    for patient in patients:
        for mov_id in movements:
            channels = []
            for sensor in range(1, 5):
                fname = f'P{patient}C{cycle}S{sensor}M{mov_id}F{forearm}O{offset}'
                fpath = data_dir / fname
                if not fpath.exists():
                    continue
                # Формат: всі значення через кому на одному рядку
                content = fpath.read_text().strip()
                arr = np.fromstring(content, dtype=np.float32, sep=',')
                channels.append(arr)

            if len(channels) < 2:   # мінімум 2 канали
                continue

            min_len = min(len(c) for c in channels)
            sig = np.array([c[:min_len] for c in channels])  # (n_ch, n_samples)
            all_signals.append(sig)
            all_labels.append(np.full(min_len, mov_id - 1, dtype=int))

    if not all_signals:
        raise FileNotFoundError(
            f"Не знайдено файлів у {data_dir}\n"
            f"Перевір cycle={cycle}, forearm={forearm}, offset={offset}"
        )

    return EMGRecord(
        signal=np.concatenate(all_signals, axis=1),
        labels=np.concatenate(all_labels),
        fs=1000, dataset='wyoflex',
        class_names=[WYOFLEX_MOVEMENTS[m] for m in sorted(movements)],
    )

NINAPRO_DB8_MOVEMENTS = {
    0: 'rest',          1: 'thumb_flex',    2: 'index_flex',
    3: 'middle_flex',   4: 'ring_flex',     5: 'little_flex',
    6: 'thumb_abd',     7: 'wrist_flex',    8: 'wrist_ext',
    9: 'hand_open',
}

def load_ninapro(root: str | Path,
                 exercise: int = 1,
                 subjects: list[int] | None = None,
                 use_restimulus: bool = True,
                 skip_rest: bool = True) -> EMGRecord:
    import scipy.io
    root = Path(root)

    if subjects is None:
        subjects = sorted({
            int(f.stem.split('_')[0][1:])
            for f in root.glob('S*_E*.mat')
        })

    all_signals, all_labels = [], []

    for subj in subjects:
        for fpath in sorted(root.glob(f'S{subj}_E{exercise}_A*.mat')):
            mat = scipy.io.loadmat(str(fpath))
            emg    = mat['emg'].T.astype(np.float32)            # → (16, N)
            lbl_key = 'restimulus' if use_restimulus else 'stimulus'
            labels = mat[lbl_key].ravel().astype(int)

            if skip_rest:
                mask = labels != 0
                emg, labels = emg[:, mask], labels[mask]

            # Перетворити мітки в 0-indexed (якщо skip_rest: 1–9 → 0–8)
            if skip_rest:
                unique = sorted(set(labels))
                remap  = {old: new for new, old in enumerate(unique)}
                labels = np.array([remap[l] for l in labels], dtype=int)

            all_signals.append(emg)
            all_labels.append(labels)

    if not all_signals:
        raise FileNotFoundError(f"Не знайдено .mat файлів у {root} (exercise={exercise})")

    labels_cat  = np.concatenate(all_labels)
    unique_ids  = sorted(set(labels_cat))
    class_names = [NINAPRO_DB8_MOVEMENTS.get(i + (1 if skip_rest else 0), f'gesture_{i}')
                   for i in unique_ids]

    return EMGRecord(
        signal=np.concatenate(all_signals, axis=1),
        labels=labels_cat,
        fs=2000, dataset='ninapro',
        class_names=class_names,
    )


def load_custom(root: str | Path,
                fs: int = 1000,
                delimiter: str | None = None) -> EMGRecord:
    root = Path(root)
    gesture_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    all_signals, all_labels, class_names = [], [], []

    if gesture_dirs:
        # Варіант A: мітки з імен папок
        class_names = [d.name for d in gesture_dirs]
        for cls_id, gdir in enumerate(gesture_dirs):
            files = sorted(list(gdir.glob('*.txt')) + list(gdir.glob('*.csv')))
            for fpath in files:
                sig = _load_txt_file(fpath, delimiter)
                all_signals.append(sig)
                all_labels.append(np.full(sig.shape[1], cls_id))
    else:
        # Варіант B: без міток
        class_names = ['unknown']
        files = sorted(list(root.glob('*.txt')) + list(root.glob('*.csv')))
        for fpath in files:
            sig = _load_txt_file(fpath, delimiter)
            all_signals.append(sig)
            all_labels.append(np.full(sig.shape[1], -1))

    if not all_signals:
        raise FileNotFoundError(f"Не знайдено TXT/CSV файлів у {root}")

    return EMGRecord(
        signal=np.concatenate(all_signals, axis=1),
        labels=np.concatenate(all_labels),
        fs=fs, dataset='custom',
        class_names=class_names,
    )


def _load_txt_file(fpath: Path, delimiter=None) -> np.ndarray:
    """TXT/CSV → (n_channels, n_samples). Автовизначення розділювача."""
    if delimiter is None:
        first_line = fpath.read_text().splitlines()[0]
        if ',' in first_line:
            delimiter = ','
        elif '\t' in first_line:
            delimiter = '\t'
        # None → numpy використає будь-який пробіл
    arr = np.loadtxt(fpath, delimiter=delimiter)
    if arr.ndim == 1:
        return arr.reshape(1, -1)   # 1 канал
    return arr.T                    # (n_samples, n_ch) → (n_ch, n_samples)


def _vmd_filter_channel(x: np.ndarray,
                         K: int,
                         alpha: float,
                         corr_threshold: float) -> np.ndarray:
    from vmdpy import VMD

    n_orig = len(x)
    # Вирівнювання до парного розміру
    if n_orig % 2 != 0:
        x_in = np.append(x, x[-1])
    else:
        x_in = x

    u, _, _ = VMD(x_in, alpha, tau=0.0, K=K, DC=0, init=1, tol=1e-7)

    selected = []
    for mode in u:
        n = min(n_orig, len(mode))          # захист від різних довжин
        corr = np.corrcoef(x[:n], mode[:n])[0, 1]
        if np.isfinite(corr) and abs(corr) > corr_threshold:
            selected.append(mode[:n_orig])  # обрізаємо до оригінального розміру

    return np.sum(selected, axis=0) if selected else x


def vmd_filter(record: EMGRecord,
               K: int | None = None,
               alpha: float = 2000.0,
               corr_threshold: float = 0.1,
               chunk_size: int = 5000) -> EMGRecord:
    if K is None:
        K = 8 if record.dataset == 'custom' else 5

    n_samples = record.signal.shape[1]
    filtered  = np.zeros_like(record.signal)

    for ch in range(record.n_channels):
        out = np.empty(n_samples, dtype=record.signal.dtype)
        for start in range(0, n_samples, chunk_size):
            end   = min(start + chunk_size, n_samples)
            chunk = record.signal[ch, start:end]
            # VMD потребує мінімум ~2×K семплів
            if len(chunk) < K * 10:
                out[start:end] = chunk
            else:
                out[start:end] = _vmd_filter_channel(chunk, K, alpha, corr_threshold)
        filtered[ch] = out

    return EMGRecord(filtered, record.labels, record.fs, record.dataset, record.class_names)


def butterworth_filter(record: EMGRecord,
                        low: float = 20.0,
                        high: float = 450.0,
                        notch_hz: float = 50.0) -> EMGRecord:
    fs = record.fs
    high = min(high, fs / 2 - 1)
    sos = signal.butter(4, [low, high], btype='bandpass', fs=fs, output='sos')
    b_n, a_n = signal.iirnotch(notch_hz, Q=30, fs=fs)

    filtered = np.array([
        signal.filtfilt(b_n, a_n, signal.sosfiltfilt(sos, record.signal[ch]))
        for ch in range(record.n_channels)
    ])
    return EMGRecord(filtered, record.labels, record.fs, record.dataset, record.class_names)


def apply_filter(record: EMGRecord, **vmd_kwargs) -> EMGRecord:
    try:
        return vmd_filter(record, **vmd_kwargs)
    except ImportError:
        print("[WARNING] vmdpy не знайдено — використовується Butterworth. pip install vmdpy")
        return butterworth_filter(record)


def segment_record(record: EMGRecord,
                   window_ms: int = 200,
                   step_ms: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """
      X: (n_windows, n_channels, window_samples)
      y: (n_windows,) — клас більшості семплів у вікні (-1 якщо немає міток)
    """
    win  = int(window_ms * record.fs / 1000)
    step = int(step_ms  * record.fs / 1000)
    n    = record.signal.shape[1]

    windows, labels = [], []
    for start in range(0, n - win + 1, step):
        end = start + win
        windows.append(record.signal[:, start:end])

        chunk = record.labels[start:end]
        valid = chunk[chunk >= 0]
        labels.append(int(np.bincount(valid).argmax()) if len(valid) else -1)

    return np.array(windows), np.array(labels)


def normalize_windows(X: np.ndarray) -> np.ndarray:
    X   = X.astype(np.float32)
    mu  = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1,  keepdims=True) + 1e-8
    return (X - mu) / std


def filter_labeled(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = y >= 0
    return X[mask], y[mask]


def _td_vectorized(X: np.ndarray) -> np.ndarray:
    """
    8 TD ознак для всіх вікон і каналів одночасно.
    X: (n_windows, n_channels, n_samples)
    → (n_windows, n_channels, 8)
    """
    mav = np.mean(np.abs(X), axis=2)
    rms = np.sqrt(np.mean(X ** 2, axis=2))
    wl  = np.sum(np.abs(np.diff(X, axis=2)), axis=2)
    zc  = np.sum(np.abs(np.diff(np.sign(X), axis=2)) >= 2, axis=2).astype(float)
    d   = np.diff(X, axis=2)
    ssc = np.sum(np.diff(np.sign(d), axis=2) != 0, axis=2).astype(float)
    var = np.var(X, axis=2)

    mu   = X.mean(axis=2, keepdims=True)
    std  = X.std(axis=2, keepdims=True) + 1e-8
    norm = (X - mu) / std
    kurt = np.mean(norm ** 4, axis=2) - 3.0   # excess kurtosis
    skw  = np.mean(norm ** 3, axis=2)

    return np.stack([mav, rms, wl, zc, ssc, var, kurt, skw], axis=2)


def _fd_vectorized(X: np.ndarray, fs: float) -> np.ndarray:
    """
    3 FD ознаки для всіх вікон і каналів одночасно.
    X: (n_windows, n_channels, n_samples)
    → (n_windows, n_channels, 3)

    Метод: FFT (швидший за Welch; масштаб TotalPower відрізняється від Welch,
    але для класифікації з нормалізацією результат еквівалентний).
    """
    n      = X.shape[2]
    freqs  = np.fft.rfftfreq(n, d=1.0 / fs)
    psd    = (np.abs(np.fft.rfft(X, axis=2)) ** 2) / n   # (W, C, n_freq)
    total  = psd.sum(axis=2) + 1e-8                       # (W, C)
    mean_f = (psd * freqs).sum(axis=2) / total

    cumsum  = np.cumsum(psd, axis=2)
    half    = (total / 2)[..., np.newaxis]
    med_idx = np.argmin(np.abs(cumsum - half), axis=2)
    med_f   = freqs[med_idx]

    return np.stack([mean_f, med_f, total], axis=2)


def build_feature_matrix(X: np.ndarray, fs: float) -> np.ndarray:
    """
    Векторизоване виділення TD + FD ознак.

    X: (n_windows, n_channels, n_samples)
    → (n_windows, n_channels × 11)

    Порядок ознак на канал: MAV, RMS, WL, ZC, SSC, VAR, KURT, SKEW,
                             MeanFreq, MedianFreq, TotalPower
    """
    td = _td_vectorized(X)     # (W, C, 8)
    fd = _fd_vectorized(X, fs) # (W, C, 3)
    W, C, _ = td.shape
    return np.concatenate([td, fd], axis=2).reshape(W, C * 11)


def run_statistical_pipeline(record: EMGRecord,
                              window_ms: int = 200,
                              step_ms: int = 100) -> dict:
    """
    apply_filter → segment → normalize → TD+FD features → LDA (5-fold CV)
    Повертає словник з метриками та проміжними даними для візуалізатора.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import confusion_matrix, f1_score

    filtered = apply_filter(record)

    X, y = segment_record(filtered, window_ms, step_ms)
    X = normalize_windows(X)
    X, y = filter_labeled(X, y)

    X_feat = build_feature_matrix(X, record.fs)

    pipe = Pipeline([('scaler', StandardScaler()), ('lda', LDA())])
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_feat, y, cv=cv, scoring='f1_macro')

    pipe.fit(X_feat, y)
    y_pred = pipe.predict(X_feat)

    return {
        'method':      'statistical_lda',
        'dataset':     record.dataset,
        'X_windows':   X,           
        'X_features':  X_feat,      
        'y_true':      y,
        'y_pred':      y_pred,
        'f1_cv_mean':  cv_scores.mean(),
        'f1_cv_std':   cv_scores.std(),
        'confusion':   confusion_matrix(y, y_pred),
        'class_names': record.class_names,
        'model':       pipe,
    }


class CNNLSTM(object):
    """Placeholder — реальний клас у validate_ai.py через import torch."""
    pass


def build_cnn_lstm(n_channels: int, window_samples: int, n_classes: int):
    """
    Повертає (model, device) — PyTorch CNN-LSTM на GPU якщо доступний.
      Conv1d(C→64) → BN → Conv1d(64→128) → BN → MaxPool → Dropout
      → LSTM(128→128) → LSTM(128→64)
      → Linear(64→64, relu) → Dropout → Linear(64→n_classes)
    """
    import torch
    import torch.nn as nn

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class _CNNLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(n_channels, 64,  kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64,        128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(0.3),
            )
            self.lstm1  = nn.LSTM(128, 128, batch_first=True, dropout=0.3)
            self.lstm2  = nn.LSTM(128,  64, batch_first=True)
            self.feat   = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3))
            self.head   = nn.Linear(64, n_classes)

        def forward(self, x, return_features=False):
            # x: (B, C, T)
            x = self.cnn(x)                  # (B, 128, T/2)
            x = x.transpose(1, 2)            # (B, T/2, 128)
            x, _ = self.lstm1(x)             # (B, T/2, 128)
            x, _ = self.lstm2(x)             # (B, T/2, 64)
            x = x[:, -1, :]                  # last timestep (B, 64)
            feat = self.feat(x)              # (B, 64)
            logits = self.head(feat)         # (B, n_classes)
            return (logits, feat) if return_features else logits

    model = _CNNLSTM().to(device)
    return model, device


def run_ai_pipeline(record: EMGRecord,
                    window_ms: int = 200,
                    step_ms: int = 100,
                    epochs: int = 30,
                    batch_size: int = 256) -> dict:
    """
      apply_filter → segment → normalize → CNN-LSTM на GPU
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix

    filtered = apply_filter(record)
    X, y     = segment_record(filtered, window_ms, step_ms)
    X        = normalize_windows(X)
    X, y     = filter_labeled(X, y)
    X_pt     = torch.tensor(X,  dtype=torch.float32)
    y_pt     = torch.tensor(y,  dtype=torch.long)

    n_classes      = len(np.unique(y))
    window_samples = X.shape[2]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_pt, y_pt, test_size=0.2, stratify=y, random_state=42
    )

    model, device = build_cnn_lstm(record.n_channels, window_samples, n_classes)
    criterion     = nn.CrossEntropyLoss()
    optimizer     = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=False
    )

    pin = device.type == 'cuda'
    tr_loader  = DataLoader(TensorDataset(X_tr, y_tr),   batch_size=batch_size,
                            shuffle=True,  pin_memory=pin, num_workers=0)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size,
                            pin_memory=pin, num_workers=0)

    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
    best_acc, best_state, patience_cnt = 0.0, None, 0

    for epoch in range(epochs):
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item() * len(yb)
            tr_correct += (out.argmax(1) == yb).sum().item()
            tr_total   += len(yb)

        # Validate
        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss    += criterion(out, yb).item() * len(yb)
                val_correct += (out.argmax(1) == yb).sum().item()
                val_total   += len(yb)

        tr_acc  = tr_correct  / tr_total
        val_acc = val_correct / val_total
        history['loss'].append(tr_loss / tr_total)
        history['val_loss'].append(val_loss / val_total)
        history['accuracy'].append(tr_acc)
        history['val_accuracy'].append(val_acc)
        scheduler.step(val_loss / val_total)

        # Early stopping
        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= 7:
                break

    model.load_state_dict(best_state)
    model.eval()

    # Передбачення на val
    with torch.no_grad():
        y_pred = model(X_val.to(device)).argmax(1).cpu().numpy()
    y_val_np = y_val.numpy()

    return {
        'method':       'ai_cnn_lstm',
        'dataset':      record.dataset,
        'X_windows':    X,
        'y_true':       y_val_np,  
        'y_val':        y_val_np,   
        'y_pred':       y_pred,
        'history':      history,
        'val_accuracy': best_acc,
        'confusion':    confusion_matrix(y_val_np, y_pred),
        'class_names':  record.class_names,
        'model':        model,
        'device':       device,
        'X_val_pt':     X_val,
    }


def compare_results(stat: dict, ai: dict) -> None:
    print("\n" + "═" * 52)
    print(f"  ПОРІВНЯННЯ: {stat['dataset'].upper()}")
    print("═" * 52)
    print(f"  {'Метод':<28} {'Метрика':>20}")
    print(f"  {'-' * 50}")
    lda_metric = f"F1={stat['f1_cv_mean']:.3f}±{stat['f1_cv_std']:.3f}"
    ai_metric  = f"Acc={ai['val_accuracy']:.3f}"
    print(f"  {'LDA + TD/FD ознаки':<28} {lda_metric:>20}")
    print(f"  {'CNN-LSTM (ШІ)':<28} {ai_metric:>20}")
    print("═" * 52)

    # Різниця в підходах до ознак
    n_feat = stat['X_features'].shape[1]
    n_ch   = stat['X_windows'].shape[1]
    print(f"\n  Статистичний: {n_feat} ознак ({n_ch} кан × 11) — ручне визначення")
    print(f"  ШІ:           ознаки вивчені автоматично з сирого сигналу")


if __name__ == '__main__':
    print("=== ДЕМО: синтетичні ЕМГ-дані (2 канали, 4 класи) ===\n")

    rng = np.random.default_rng(42)
    fs, n_classes, n_channels = 1000, 4, 2
    segments, labels = [], []

    for cls in range(n_classes):
        n = fs * 5  # 5 секунд на клас
        t = np.linspace(0, 5, n)
        base_freq = 40 + cls * 35
        clean = np.stack([
            np.sin(2 * np.pi * base_freq * t) * (0.5 + 0.4 * np.sin(2 * np.pi * 1.5 * t))
            for _ in range(n_channels)
        ])
        noise = rng.normal(0, 0.4, clean.shape)  # шум як у власному датасеті
        segments.append(clean + noise)
        labels.append(np.full(n, cls))

    record = EMGRecord(
        signal=np.concatenate(segments, axis=1),
        labels=np.concatenate(labels),
        fs=fs, dataset='custom',
        class_names=[f'gesture_{i}' for i in range(n_classes)],
    )

    print(f"Датасет: {record.n_channels} канали, {record.duration_sec:.0f} с, {record.fs} Гц")
    print(f"Класи: {record.class_names}\n")

    # Butterworth як fallback
    filtered = butterworth_filter(record)
    X, y = segment_record(filtered)
    X = normalize_windows(X)
    X, y = filter_labeled(X, y)
    print(f"Вікна після сегментації: {X.shape}  →  {X.shape[1]} канали × {X.shape[2]} семпли")

    X_feat = build_feature_matrix(X, fs)
    n_feat_per_ch = 11
    print(f"Матриця ознак (LDA):    {X_feat.shape}  →  {record.n_channels} кан × {n_feat_per_ch} ознак\n")

    print("Для запуску з реальними даними:")
    print(r"  record = load_wyoflex(r'D:\path\to\WyoFlex_Dataset')")
    print(r"  record = load_ninapro(r'D:\path\to\NinaproDB8_Dataset', exercise=1)")
    print(r"  record = load_custom(r'D:\path\to\custom_dataset', fs=1000)")
    print()
    print("  stat = run_statistical_pipeline(record)")
    print("  ai   = run_ai_pipeline(record, epochs=30)")
    print("  compare_results(stat, ai)")

    record = load_wyoflex(r'D:\ClaudeCode\practics-project\datasets\WyoFlex_Dataset')
    stat = run_statistical_pipeline(record)