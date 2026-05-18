import tkinter as tk
from tkinter import ttk, filedialog
import threading
import time
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.filterwarnings('ignore')

# Інтеграція з emg_proposals.py (опційна)
try:
    from emg_math import (
        butterworth_filter, apply_filter,
        segment_record, normalize_windows, filter_labeled,
        build_feature_matrix, EMGRecord,
    )
    try:
        from vmdpy import VMD
        HAS_VMD = True
    except ImportError:
        HAS_VMD = False
    HAS_PROPOSALS = True
except ImportError:
    HAS_PROPOSALS = False
    HAS_VMD = False

COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
          '#9467bd', '#8c564b', '#e377c2', '#17becf']
FEATURE_NAMES = ['MAV', 'RMS', 'WL', 'ZC', 'SSC', 'VAR', 'KURT', 'SKEW',
                 'MeanF', 'MedF', 'TotPow']


class DataLoader:
    def __init__(self):
        self.channels = {}
        self.current_file = None
        self.fs = 1000

    def load_data(self, file_name: str) -> bool | str:
        try:
            self.channels.clear()
            ext = file_name.rsplit('.', 1)[-1].lower()
            if ext == 'mat':
                self._parse_mat(file_name)
            else:
                self._parse_csv(file_name)
            if not self.channels:
                return 'Не знайдено жодного валідного каналу.'
            self.current_file = file_name
            return True
        except Exception as e:
            return f'Помилка завантаження: {e}'

    def get_channel_data(self, ch: str, zeroed=True):
        if ch not in self.channels:
            return None, None
        t = self.channels[ch]['time']
        v = self.channels[ch]['voltage']
        return (t - t[0], v) if zeroed and len(t) > 0 else (t, v)

    def to_emg_record(self, selected_channels: list) -> 'EMGRecord | None':
        if not HAS_PROPOSALS or not selected_channels:
            return None
        signals, lengths = [], []
        for ch in selected_channels:
            _, v = self.get_channel_data(ch)
            if v is not None:
                signals.append(v.astype(np.float32))
                lengths.append(len(v))
        if not signals:
            return None
        n = min(lengths)
        sig = np.array([s[:n] for s in signals])
        return EMGRecord(signal=sig, labels=np.full(n, -1),
                         fs=self.fs, dataset='custom', class_names=[])

    def _parse_csv(self, file_name: str):
        df = pd.read_csv(file_name, sep=None, engine='python', encoding='utf-8-sig')
        time_kw = ['time', 'timestamp', 'sec', 't(s)', 't']
        volt_kw = ['volt', 'voltage', 'emg', 'mv', 'signal', 'value', 'ch', 'force', 'traj']

        time_col = next((c for c in df.columns
                         if any(k in str(c).lower() for k in time_kw)), None)
        if not time_col and len(df.columns) > 0:
            first = str(df.columns[0])
            if 'unnamed' in first.lower() or first.strip() == '':
                time_col = df.columns[0]

        volt_cols = [c for c in df.columns
                     if any(k in str(c).lower() for k in volt_kw) and c != time_col]

        # Якщо заголовків немає — вся матриця числова
        if not volt_cols and not time_col:
            arr = df.select_dtypes(include=np.number).values.T
            for i, row in enumerate(arr):
                self.channels[f'ch{i+1}'] = {
                    'time':    np.arange(len(row), dtype=float) / self.fs,
                    'voltage': row.astype(float),
                }
            return

        if time_col and volt_cols:
            t = pd.to_numeric(df[time_col], errors='coerce').values
            dt = np.diff(t[~np.isnan(t)])
            if len(dt) > 0 and dt[0] > 0:
                self.fs = int(round(1.0 / dt[0]))
            for vc in volt_cols:
                v = pd.to_numeric(df[vc], errors='coerce').values
                mask = ~np.isnan(t) & ~np.isnan(v)
                if np.any(mask):
                    self.channels[vc] = {'time': t[mask], 'voltage': v[mask]}

    def _parse_mat(self, file_name: str):
        import scipy.io
        mat = scipy.io.loadmat(file_name)
        if 'emg' in mat:
            emg = mat['emg'].astype(float)
            if emg.ndim == 1:
                emg = emg.reshape(-1, 1)
            elif emg.shape[0] < emg.shape[1]:
                emg = emg.T
            n_samples, n_ch = emg.shape
            self.fs = 2000
            t = np.arange(n_samples) / self.fs
            for i in range(min(n_ch, 16)):
                self.channels[f'ch{i+1}'] = {'time': t, 'voltage': emg[:, i]}

    def clear(self):
        self.channels.clear()
        self.current_file = None


class EMGProcessor:
    @staticmethod
    def envelope(voltage: np.ndarray, window: int = 20) -> np.ndarray:
        v = voltage - voltage.mean()
        v = np.abs(v)
        if window > 1:
            v = np.convolve(v, np.ones(window) / window, mode='same')
        return v

    @staticmethod
    def butterworth(voltage: np.ndarray, fs: int,
                    low: float, high: float, notch: float) -> np.ndarray:
        from scipy import signal as sp
        high = min(high, fs / 2 - 1)
        sos = sp.butter(4, [low, high], btype='bandpass', fs=fs, output='sos')
        out = sp.sosfiltfilt(sos, voltage)
        b, a = sp.iirnotch(notch, Q=30, fs=fs)
        return sp.filtfilt(b, a, out)

    @staticmethod
    def pipeline_stages(voltage: np.ndarray, fs: int,
                        params: dict) -> dict:
        from scipy import signal as sp

        low   = params.get('low', 20)
        high  = params.get('high', 450)
        notch = params.get('notch', 50)
        win   = params.get('window_ms', 200)

        high = min(high, fs / 2 - 1)
        sos  = sp.butter(4, [low, high], btype='bandpass', fs=fs, output='sos')
        v_bp = sp.sosfiltfilt(sos, voltage)
        b, a = sp.iirnotch(notch, Q=30, fs=fs)
        v_filt = sp.filtfilt(b, a, v_bp)

        # Одне сегментне вікно (перше)
        win_samples = int(win * fs / 1000)
        n = len(voltage)
        seg = v_filt[:win_samples] if n >= win_samples else v_filt

        return {
            'raw':      voltage,
            'filtered': v_filt,
            'window':   seg,
        }

    @staticmethod
    def lda_features(record: 'EMGRecord | None', params: dict) -> dict | None:
        """
        Два режими:
          - Без міток (всі -1): показує середні TD/FD ознаки по вікнах (без класифікації)
          - З мітками (≥2 класи): повний LDA з крос-валідацією і F-statistic
        """
        if not HAS_PROPOSALS or record is None:
            return None

        from sklearn.preprocessing import StandardScaler

        filtered = butterworth_filter(record)
        win_ms   = params.get('window_ms', 200)
        step_ms  = params.get('step_ms', 100)
        X, y_all = segment_record(filtered, window_ms=win_ms, step_ms=step_ms)
        X        = normalize_windows(X)

        if len(X) < 5:
            return {'error': f'Замало вікон ({len(X)}). Зменш window_ms або step_ms.'}

        X_feat  = np.nan_to_num(build_feature_matrix(X, record.fs))
        n_feat  = X_feat.shape[1]

        labeled_mask = y_all >= 0
        has_labels   = labeled_mask.sum() > 0 and len(np.unique(y_all[labeled_mask])) >= 2

        if not has_labels:
            feat_means = X_feat.mean(axis=0)
            feat_stds  = X_feat.std(axis=0)
            return {
                'mode':       'no_labels',
                'f_stats':    feat_means,
                'feat_stds':  feat_stds,
                'n_windows':  len(X),
                'n_features': n_feat,
                'f1_mean':    None,
                'f1_std':     None,
                'n_classes':  0,
            }

        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from scipy.stats import f_oneway

        X_lab = X_feat[labeled_mask]
        y_lab = y_all[labeled_mask]
        n_cls = len(np.unique(y_lab))

        f_vals = []
        for i in range(n_feat):
            groups = [X_lab[y_lab == cls, i] for cls in np.unique(y_lab)]
            try:
                f, _ = f_oneway(*groups)
                f_vals.append(float(f) if np.isfinite(f) else 0.0)
            except Exception:
                f_vals.append(0.0)

        n_splits = min(5, min(np.bincount(y_lab.astype(int))))
        n_splits = max(2, n_splits)
        pipe     = Pipeline([('sc', StandardScaler()),
                             ('lda', LinearDiscriminantAnalysis())])
        cv       = StratifiedKFold(n_splits, shuffle=True, random_state=42)
        scores   = cross_val_score(pipe, X_lab, y_lab, cv=cv, scoring='f1_macro')

        return {
            'mode':       'labeled',
            'f_stats':    np.array(f_vals),
            'n_windows':  len(y_lab),
            'n_features': n_feat,
            'f1_mean':    scores.mean(),
            'f1_std':     scores.std(),
            'n_classes':  n_cls,
        }


class FilterPanel(ttk.LabelFrame):
    DEFAULTS = dict(low=20, high=450, notch=50, window_ms=200, step_ms=100, envelope=20)

    def __init__(self, parent, on_change, **kw):
        super().__init__(parent, text='Параметри фільтрації', **kw)
        self.on_change = on_change
        self._vars   = {}
        self._after  = None   # debounce id

        specs = [
            ('low',       'Bandpass Low (Hz)',   5,   200,  1),
            ('high',      'Bandpass High (Hz)',  100, 499,  1),
            ('notch',     'Notch (Hz)',          45,  65,   1),
            ('window_ms', 'Вікно (ms)',          50,  500,  10),
            ('step_ms',   'Крок (ms)',           10,  250,  5),
            ('envelope',  'Envelope win',        1,   100,  1),
        ]
        for key, label, mn, mx, res in specs:
            ttk.Label(self, text=label, font=('Segoe UI', 8)).pack(anchor='w', padx=4)
            var = tk.IntVar(value=self.DEFAULTS[key])
            self._vars[key] = var
            row = ttk.Frame(self)
            row.pack(fill=tk.X, padx=4, pady=1)
            sl = tk.Scale(row, variable=var, from_=mn, to=mx, resolution=res,
                          orient=tk.HORIZONTAL, showvalue=False, length=130,
                          command=self._debounce)
            sl.pack(side=tk.LEFT)
            ttk.Label(row, textvariable=var, width=4).pack(side=tk.LEFT)

        ttk.Button(self, text='Скинути', command=self.reset).pack(pady=4)

    def _debounce(self, _=None):
        if self._after:
            self.after_cancel(self._after)
        self._after = self.after(300, self._fire)

    def _fire(self):
        self._after = None
        self.on_change(self.get())

    def get(self) -> dict:
        return {k: v.get() for k, v in self._vars.items()}

    def reset(self):
        for k, v in self._vars.items():
            v.set(self.DEFAULTS[k])
        self._fire()


class PlotPanel(ttk.Frame):
    def __init__(self, parent, title='', **kw):
        super().__init__(parent, **kw)
        self.title = title
        self._lines    = {}   # name → Line2D
        self._line_vis = {}   # name → BooleanVar
        self._hover_cb = None

        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)
        self.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(6, 3), tight_layout=True)
        self.ax  = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        # Toolbar
        tb = ttk.Frame(self)
        tb.grid(row=1, column=0, sticky='ew')
        ttk.Button(tb, text='↺ Reset', width=7, command=self.reset_view).pack(side=tk.LEFT, padx=2)
        self._legend_frame = ttk.Frame(tb)
        self._legend_frame.pack(side=tk.LEFT, padx=6)

        # Pan
        self._panning = False
        self._pan0    = None
        self._xlim0   = self._ylim0 = None
        self.canvas.mpl_connect('scroll_event',         self._on_scroll)
        self.canvas.mpl_connect('button_press_event',   self._on_press)
        self.canvas.mpl_connect('button_release_event', lambda e: setattr(self, '_panning', False))
        self.canvas.mpl_connect('motion_notify_event',  self._on_move)

        self._empty()

    def set_hover_callback(self, cb):
        self._hover_cb = cb

    def plot(self, data: dict, xlabel='', ylabel=''):
        """
        data: {name: {'x': array, 'y': array, 'color': str, 'alpha': float, 'lw': float}}
        """
        self.ax.clear()
        self._lines.clear()

        for w in self._legend_frame.winfo_children():
            w.destroy()

        for name, d in data.items():
            color = d.get('color', COLORS[len(self._lines) % len(COLORS)])
            alpha = d.get('alpha', 1.0)
            lw    = d.get('lw', 1.0)
            ln, = self.ax.plot(d['x'], d['y'], color=color, alpha=alpha,
                               linewidth=lw, label=name)
            self._lines[name] = ln

            var = self._line_vis.setdefault(name, tk.BooleanVar(value=True))
            var.set(True)
            cb = ttk.Checkbutton(self._legend_frame, text=name[:14], variable=var,
                                 command=lambda n=name, v=var: self._toggle(n, v))
            cb.pack(side=tk.LEFT)

        self.ax.set_title(self.title, fontsize=9)
        if xlabel: self.ax.set_xlabel(xlabel, fontsize=8)
        if ylabel: self.ax.set_ylabel(ylabel, fontsize=8)
        self.ax.tick_params(labelsize=7)
        self._orig_xlim = self.ax.get_xlim()
        self._orig_ylim = self.ax.get_ylim()
        self.canvas.draw_idle()

    def clear(self):
        self._empty()

    def reset_view(self):
        if hasattr(self, '_orig_xlim'):
            self.ax.set_xlim(self._orig_xlim)
            self.ax.set_ylim(self._orig_ylim)
            self.canvas.draw_idle()

    def _empty(self):
        self.ax.clear()
        self.ax.text(0.5, 0.5, 'Немає даних', ha='center', va='center',
                     transform=self.ax.transAxes, color='gray', fontsize=11)
        self.canvas.draw_idle()

    def _toggle(self, name, var):
        if name in self._lines:
            self._lines[name].set_visible(var.get())
            self.canvas.draw_idle()

    def _on_scroll(self, ev):
        if ev.inaxes != self.ax: return
        f = 1.2 if ev.button == 'up' else 0.8
        xl, yl = self.ax.get_xlim(), self.ax.get_ylim()
        rx = (xl[1] - ev.xdata) / (xl[1] - xl[0])
        ry = (yl[1] - ev.ydata) / (yl[1] - yl[0])
        nw = (xl[1] - xl[0]) / f; nh = (yl[1] - yl[0]) / f
        self.ax.set_xlim(ev.xdata - nw * (1 - rx), ev.xdata + nw * rx)
        self.ax.set_ylim(ev.ydata - nh * (1 - ry), ev.ydata + nh * ry)
        self.canvas.draw_idle()

    def _on_press(self, ev):
        if ev.inaxes != self.ax or ev.button != 1: return
        self._panning = True
        self._pan0    = (ev.x, ev.y)
        self._xlim0   = self.ax.get_xlim()
        self._ylim0   = self.ax.get_ylim()

    def _on_move(self, ev):
        if self._hover_cb:
            self._hover_cb(ev.xdata, ev.ydata) if ev.inaxes == self.ax else self._hover_cb(None, None)
        if not self._panning or self._pan0 is None: return
        inv = self.ax.transData.inverted()
        x0, y0 = inv.transform(self._pan0)
        x1, y1 = inv.transform((ev.x, ev.y))
        self.ax.set_xlim(self._xlim0[0] - (x1-x0), self._xlim0[1] - (x1-x0))
        self.ax.set_ylim(self._ylim0[0] - (y1-y0), self._ylim0[1] - (y1-y0))
        self.canvas.draw_idle()


class PipelinePanel(ttk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self._windows    = [] 
        self._window_idx = 0

        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)
        self.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(10, 6), tight_layout=True)
        self.axes = self.fig.subplots(2, 2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        nav = ttk.Frame(self)
        nav.grid(row=1, column=0)
        ttk.Button(nav, text='◄ Попереднє', command=self._prev).pack(side=tk.LEFT, padx=5)
        self._win_label = ttk.Label(nav, text='Вікно 0/0')
        self._win_label.pack(side=tk.LEFT, padx=10)
        ttk.Button(nav, text='Наступне ►', command=self._next).pack(side=tk.LEFT)

        self._draw_empty()

    def update(self, t: np.ndarray, stages: dict, windows: list, fs: int):
        """stages: {'raw', 'filtered', 'window'}, windows: list of 1D arrays"""
        self._windows    = windows
        self._window_idx = 0
        self._t          = t
        self._stages     = stages
        self._fs         = fs
        self._redraw()

    def _prev(self):
        if self._window_idx > 0:
            self._window_idx -= 1
            self._redraw()

    def _next(self):
        if self._window_idx < len(self._windows) - 1:
            self._window_idx += 1
            self._redraw()

    def _redraw(self):
        if not self._windows:
            self._draw_empty(); return

        for ax in self.axes.flat:
            ax.clear()

        t     = self._t
        st    = self._stages
        ax    = self.axes

        # A: Raw
        ax[0,0].plot(t[:len(st['raw'])], st['raw'], color='steelblue', lw=0.7)
        ax[0,0].set_title('1. Сирий сигнал', fontsize=9)

        # B: Filtered
        ax[0,1].plot(t[:len(st['filtered'])], st['filtered'], color='darkorange', lw=0.8)
        ax[0,1].set_title('2. Після Butterworth', fontsize=9)

        # C: Current window highlighted on filtered
        idx   = self._window_idx
        win   = self._windows[idx]
        n_tot = len(self._windows)
        step  = int(0.1 * self._fs)   # приблизний крок у семплах
        t_win = np.arange(len(win)) / self._fs
        ax[1,0].plot(t_win, win, color='green', lw=1)
        ax[1,0].set_title(f'3. Вікно {idx+1}/{n_tot}', fontsize=9)
        self._win_label.config(text=f'Вікно {idx+1}/{n_tot}')

        # D: TD features для поточного вікна
        feat_td = [
            np.mean(np.abs(win)),
            np.sqrt(np.mean(win**2)),
            np.sum(np.abs(np.diff(win))),
            float(np.sum(np.abs(np.diff(np.sign(win))) >= 2)),
            float(np.sum(np.diff(np.sign(np.diff(win))) != 0)),
            np.var(win),
        ]
        labels = ['MAV', 'RMS', 'WL', 'ZC', 'SSC', 'VAR']
        colors = ['#2196F3','#4CAF50','#FF9800','#9C27B0','#F44336','#607D8B']
        ax[1,1].bar(labels, feat_td, color=colors, alpha=0.85)
        ax[1,1].set_title('4. TD-ознаки вікна', fontsize=9)
        ax[1,1].tick_params(axis='x', labelsize=7)

        for a in self.axes.flat:
            a.tick_params(labelsize=7)

        self.canvas.draw_idle()

    def _draw_empty(self):
        for ax in self.axes.flat:
            ax.clear()
            ax.text(0.5, 0.5, '...', ha='center', va='center',
                    transform=ax.transAxes, color='lightgray', fontsize=14)
        self.canvas.draw_idle()


class AnalysisPanel(ttk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.fig  = Figure(figsize=(12, 7), tight_layout=True)
        self.axes = self.fig.subplots(2, 2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        self._lda_f1  = None
        self._cnn_f1  = None
        self._n_cls   = 1
        self._draw_empty()

    def update_lda(self, result: dict):
        ax0, ax1 = self.axes[0, 0], self.axes[0, 1]

        if 'error' in result:
            for ax in [ax0, ax1]:
                ax.clear()
                ax.text(0.5, 0.5, result['error'], ha='center', va='center',
                        transform=ax.transAxes, color='red', fontsize=9)
            self.canvas.draw_idle()
            return

        mode   = result.get('mode', 'labeled')
        f      = result['f_stats']
        top_n  = min(15, len(f))
        top_i  = np.argsort(f)[::-1][:top_n]
        labels = [f'ch{i//11+1}_{FEATURE_NAMES[i%11]}' for i in top_i]

        ax0.clear()
        ax0.barh(labels[::-1], f[top_i[::-1]], color='steelblue', alpha=0.85)
        if mode == 'labeled' and result.get('f1_mean') is not None:
            self._lda_f1 = result['f1_mean']
            self._n_cls  = result.get('n_classes', 1)
            title = (f'LDA топ-{top_n} ознак (F-statistic)\n'
                     f'F1={result["f1_mean"]:.3f}±{result["f1_std"]:.3f}  '
                     f'({result["n_classes"]} класи, {result["n_windows"]} вікон)')
        else:
            title = (f'Ознаки по вікнах (середнє, без міток)\n'
                     f'{result["n_windows"]} вікон — завантажте датасет для LDA')
        ax0.set_title(title, fontsize=8)
        ax0.tick_params(labelsize=7)

        # [0,1] — порівняння (заповниться після CNN)
        self._redraw_comparison()
        self.canvas.draw_idle()

    def update_cnn_training(self, epoch: int, history: dict):
        ax = self.axes[1, 0]
        ax.clear()
        ep = range(1, epoch + 1)
        ax.plot(ep, history['loss'],     color='steelblue',  lw=1.5, label='train loss')
        ax.plot(ep, history['val_loss'], color='steelblue',  lw=1.5, ls='--', label='val loss')
        ax.plot(ep, history['accuracy'],     color='darkorange', lw=1.5, label='train acc')
        ax.plot(ep, history['val_accuracy'], color='darkorange', lw=1.5, ls='--', label='val acc')
        ax.set_title(f'CNN навчання  (epoch {epoch})', fontsize=9)
        ax.set_xlabel('Epoch', fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        self.canvas.draw_idle()

    def update_cnn_final(self, result: dict):
        self._cnn_f1 = result.get('f1', 0.0)
        self._redraw_comparison()

        # [1,1] — GradCAM: до 4 прикладів
        ax = self.axes[1, 1]
        ax.clear()
        gdata = result.get('gradcam_data', [])
        if not gdata:
            ax.text(0.5, 0.5, 'GradCAM недоступний', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
        else:
            n_show = min(len(gdata), 4)
            # Покажемо перший приклад (найпростіше для читабельності)
            d   = gdata[0]
            sig = d['signal']
            sal = d['saliency']
            T   = len(sig)
            t   = np.arange(T)

            # Сигнал
            ax.plot(t, sig, color='steelblue', lw=1, zorder=2)
            # Saliency як кольоровий фон: fill_between по квантилях
            ax.fill_between(t, sig.min(), sig.max(),
                            alpha=sal * 0.6, color='red', zorder=1)
            # Colorbar-заступник — annotate
            ax.set_title(
                f'GradCAM  true:{d["class_name"][:8]}  '
                f'pred:{result["class_names"][d["pred_cls"]][:8] if d["pred_cls"] < len(result["class_names"]) else d["pred_cls"]}  '
                f'{"✓" if d["true_cls"]==d["pred_cls"] else "✗"}',
                fontsize=8
            )
            ax.set_xlabel('семпли', fontsize=8)
            ax.tick_params(labelsize=7)
            # Додаємо кнопки «Попередній/Наступний приклад» через текст
            ax.text(0.98, 0.02, f'1/{len(gdata)} класів',
                    transform=ax.transAxes, ha='right', fontsize=7, color='gray')

        self._gradcam_data   = result.get('gradcam_data', [])
        self._gradcam_idx    = 0
        self._gradcam_result = result
        self.canvas.draw_idle()

    def next_gradcam(self):
        if not hasattr(self, '_gradcam_data') or not self._gradcam_data:
            return
        self._gradcam_idx = (self._gradcam_idx + 1) % len(self._gradcam_data)
        self._draw_single_gradcam(self._gradcam_idx)

    def _draw_single_gradcam(self, idx: int):
        ax  = self.axes[1, 1]
        ax.clear()
        d   = self._gradcam_data[idx]
        sig = d['signal']
        sal = d['saliency']
        t   = np.arange(len(sig))
        ax.plot(t, sig, color='steelblue', lw=1, zorder=2)
        ax.fill_between(t, sig.min(), sig.max(),
                        alpha=sal * 0.6, color='red', zorder=1)
        cls_names = self._gradcam_result.get('class_names', [])
        pred_name = cls_names[d['pred_cls']] if d['pred_cls'] < len(cls_names) else str(d['pred_cls'])
        ax.set_title(
            f'GradCAM [{idx+1}/{len(self._gradcam_data)}]  '
            f'true:{d["class_name"][:8]}  pred:{pred_name[:8]}  '
            f'{"✓" if d["true_cls"]==d["pred_cls"] else "✗"}',
            fontsize=8
        )
        ax.set_xlabel('семпли', fontsize=8)
        ax.tick_params(labelsize=7)
        self.canvas.draw_idle()

    def _redraw_comparison(self):
        ax = self.axes[0, 1]
        ax.clear()
        methods = ['LDA', 'CNN-LSTM']
        scores  = [self._lda_f1 or 0.0, self._cnn_f1 or 0.0]
        colors  = ['darkorange', 'steelblue']
        bars    = ax.bar(methods, scores, color=colors, alpha=0.85, width=0.5)
        ax.set_ylim(0, 1)
        n_cls = max(self._n_cls, 1)
        ax.axhline(1/n_cls, color='gray', ls='--', lw=1, label=f'random (1/{n_cls})')
        ax.legend(fontsize=8)
        for bar, val in zip(bars, scores):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                        f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
        ax.set_title('LDA vs CNN-LSTM  (F1 macro)', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis='y', alpha=0.3)

    def _draw_empty(self):
        hints = [
            'Run LDA → ознаки сигналу',
            'Run LDA → порівняння',
            'Run CNN → криві навчання',
            'Run CNN → GradCAM',
        ]
        for ax, hint in zip(self.axes.flat, hints):
            ax.clear()
            ax.text(0.5, 0.5, hint, ha='center', va='center',
                    transform=ax.transAxes, color='lightgray', fontsize=10)
        self.canvas.draw_idle()


class CollapsibleFrame(ttk.Frame):
    def __init__(self, parent, title: str, expanded: bool = True, **kw):
        super().__init__(parent, **kw)
        self._expanded = expanded
        self.columnconfigure(0, weight=1)

        hdr = ttk.Frame(self, relief='groove')
        hdr.grid(row=0, column=0, sticky='ew')
        hdr.columnconfigure(1, weight=1)

        self._icon = tk.StringVar(value='▼' if expanded else '▶')
        ttk.Button(hdr, textvariable=self._icon, width=3,
                   command=self.toggle).grid(row=0, column=0, padx=2, pady=1)
        ttk.Label(hdr, text=title,
                  font=('Segoe UI', 9, 'bold')).grid(row=0, column=1, sticky='w', padx=2)

        self.body = ttk.Frame(self)
        if expanded:
            self.body.grid(row=1, column=0, sticky='nsew')

    def toggle(self):
        if self._expanded:
            self.body.grid_forget()
            self._icon.set('▶')
        else:
            self.body.grid(row=1, column=0, sticky='nsew')
            self._icon.set('▼')
        self._expanded = not self._expanded

    def expand(self):
        if not self._expanded:
            self.toggle()

    def collapse(self):
        if self._expanded:
            self.toggle()


class Console(ttk.Frame):
    def __init__(self, parent, command_callback=None, **kw):
        super().__init__(parent, **kw)
        self.command_callback = command_callback
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.text = tk.Text(self, height=5, bg='#1e1e1e', fg='#d4d4d4',
                            font=('Consolas', 9), state=tk.DISABLED)
        sb = ttk.Scrollbar(self, command=self.text.yview)
        self.text.configure(yscrollcommand=sb.set)
        self.entry = ttk.Entry(self, font=('Consolas', 9))
        self.entry.bind('<Return>', self._on_enter)

        self.text.grid(row=0, column=0, sticky='nsew')
        sb.grid(row=0, column=1, sticky='ns')
        self.entry.grid(row=1, column=0, sticky='ew', pady=2)

    def _on_enter(self, _):
        cmd = self.entry.get().strip()
        if cmd:
            self.log(cmd, user=True)
            if self.command_callback:
                self.command_callback(cmd)
            self.entry.delete(0, tk.END)

    def log(self, msg: str, user=False):
        self.text.config(state=tk.NORMAL)
        self.text.insert(tk.END, f"{'>' if user else '»'} {msg}\n")
        self.text.see(tk.END)
        self.text.config(state=tk.DISABLED)


class VisualizerApp:
    def __init__(self, root: tk.Tk):
        self.root     = root
        self.root.title('EMG Visualizer')
        self.loader   = DataLoader()
        self.proc     = EMGProcessor()
        self._split   = False
        self._loading = False
        self._labeled_record = None   # EMGRecord з мітками (WyoFlex/Ninapro)

        self._hover_var  = tk.StringVar(value='T: ---\nV: ---')
        self._status_var = tk.StringVar(value='Готово')
        self._filter_params = FilterPanel.DEFAULTS.copy()

        self._build_ui()
        self.console.log('EMG Visualizer ініціалізовано.')
        if not HAS_PROPOSALS:
            self.console.log('[!] emg_proposals.py не знайдено — базовий режим.')
        if not HAS_VMD:
            self.console.log('[!] vmdpy не знайдено — тільки Butterworth.')

    def _build_ui(self):
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f'{int(sw*0.82)}x{int(sh*0.82)}+{int(sw*0.09)}+{int(sh*0.05)}')

        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Toolbar
        tb = ttk.Frame(self.root, relief='solid', padding=2)
        tb.grid(row=0, column=0, sticky='ew')
        for text, cmd in [('New', self._on_new), ('Open', self._on_open)]:
            ttk.Button(tb, text=text, command=cmd, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)
        ttk.Button(tb, text='📂 Load Dataset', command=self._on_load_dataset).pack(side=tk.LEFT, padx=2)
        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)
        ttk.Button(tb, text='▶ Run LDA', command=self._run_lda).pack(side=tk.LEFT, padx=2)
        ttk.Button(tb, text='▶ Run CNN', command=self._run_cnn).pack(side=tk.LEFT, padx=2)
        ttk.Button(tb, text='GradCAM ►', command=self._next_gradcam).pack(side=tk.LEFT, padx=2)
        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)
        self._split_btn = ttk.Button(tb, text='Split ⬜', command=self._toggle_split)
        self._split_btn.pack(side=tk.LEFT, padx=2)
        ttk.Label(tb, textvariable=self._status_var, foreground='gray').pack(side=tk.RIGHT, padx=8)
        self._pbar = ttk.Progressbar(tb, mode='indeterminate', length=100)
        self._pbar.pack(side=tk.RIGHT, padx=4)

        _bg = ttk.Style().lookup('TFrame', 'background')
        h_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                                sashwidth=5, sashpad=0, bd=0, bg=_bg)
        h_pane.grid(row=1, column=0, sticky='nsew', padx=2, pady=2)

        side = ttk.Frame(h_pane)
        h_pane.add(side, minsize=120, width=220, stretch='never')

        v_pane = tk.PanedWindow(h_pane, orient=tk.VERTICAL,
                                sashwidth=5, sashpad=0, bd=0, bg=_bg)
        h_pane.add(v_pane, minsize=300, stretch='always')

        main_frame = ttk.Frame(v_pane)
        v_pane.add(main_frame, stretch='always')

        console_frame = ttk.Frame(v_pane, height=130)
        v_pane.add(console_frame, minsize=60, stretch='never')
        console_frame.rowconfigure(0, weight=1)
        console_frame.columnconfigure(0, weight=1)
        self.console = Console(console_frame, command_callback=self._on_command)
        self.console.grid(row=0, column=0, sticky='nsew')

        side.rowconfigure(0, weight=1)
        side.columnconfigure(0, weight=1)

        side_canvas = tk.Canvas(side, borderwidth=0, highlightthickness=0)
        side_sb     = ttk.Scrollbar(side, orient=tk.VERTICAL, command=side_canvas.yview)
        side_canvas.configure(yscrollcommand=side_sb.set)
        side_sb.pack(side=tk.RIGHT, fill=tk.Y)
        side_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        side_inner = ttk.Frame(side_canvas)
        side_inner.bind('<Configure>',
                        lambda e: side_canvas.configure(
                            scrollregion=side_canvas.bbox('all')))
        side_canvas.create_window((0, 0), window=side_inner, anchor='nw')
        side_canvas.bind('<Enter>',
                         lambda e: side_canvas.bind_all(
                             '<MouseWheel>',
                             lambda ev: side_canvas.yview_scroll(-1*(ev.delta//120), 'units')))
        side_canvas.bind('<Leave>',
                         lambda e: side_canvas.unbind_all('<MouseWheel>'))

        cf_file = CollapsibleFrame(side_inner, title='Файл', expanded=True)
        cf_file.pack(fill=tk.X, pady=2, padx=2)
        self._file_var = tk.StringVar(value='Файл відсутній')
        ttk.Label(cf_file.body, textvariable=self._file_var,
                  wraplength=185, font=('Consolas', 8)).pack(padx=4, pady=3, anchor='w')

        cf_ch = CollapsibleFrame(side_inner, title='Канали', expanded=True)
        cf_ch.pack(fill=tk.X, pady=2, padx=2)
        self._ch_scroll = ttk.Frame(cf_ch.body)
        self._ch_scroll.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(self._ch_scroll, text='Завантажте файл...', foreground='gray').pack()
        self._ch_vars = {}

        cf_filt = CollapsibleFrame(side_inner, title='Параметри фільтрації', expanded=True)
        cf_filt.pack(fill=tk.X, pady=2, padx=2)
        self._filter_panel = FilterPanel(cf_filt.body, on_change=self._on_filter_change)
        self._filter_panel.pack(fill=tk.X)

        cf_info = CollapsibleFrame(side_inner, title='Курсор', expanded=True)
        cf_info.pack(fill=tk.X, pady=2, padx=2)
        ttk.Label(cf_info.body, textvariable=self._hover_var,
                  font=('Consolas', 9), justify=tk.LEFT).pack(padx=4, pady=4)

        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

        self.nb = ttk.Notebook(main_frame)
        self.nb.grid(row=0, column=0, sticky='nsew')

        # Tab: Signal
        self._tab_signal = ttk.Frame(self.nb)
        self.nb.add(self._tab_signal, text=' Signal ')
        self._build_signal_tab()

        # Tab: Pipeline
        self._tab_pipeline = ttk.Frame(self.nb)
        self.nb.add(self._tab_pipeline, text=' Pipeline ')
        self._pipeline_panel = PipelinePanel(self._tab_pipeline)
        self._pipeline_panel.pack(fill=tk.BOTH, expand=True)

        # Tab: Analysis
        self._tab_analysis = ttk.Frame(self.nb)
        self.nb.add(self._tab_analysis, text=' Analysis ')
        self._analysis_panel = AnalysisPanel(self._tab_analysis)
        self._analysis_panel.pack(fill=tk.BOTH, expand=True)

    def _build_signal_tab(self):
        self._tab_signal.rowconfigure(0, weight=1)
        self._tab_signal.columnconfigure(0, weight=1)
        self._tab_signal.columnconfigure(1, weight=1)

        self._plot_left = PlotPanel(self._tab_signal, title='Сигнал')
        self._plot_left.set_hover_callback(self._on_hover)
        self._plot_left.grid(row=0, column=0, sticky='nsew')

        self._plot_right = PlotPanel(self._tab_signal, title='Відфільтрований')
        self._plot_right.grid(row=0, column=1, sticky='nsew')
        self._plot_right.grid_remove()   # прихований поки split вимкнений

    def _on_hover(self, x, y):
        if x is None:
            self._hover_var.set('T: ---\nV: ---')
        else:
            self._hover_var.set(f'T: {x:.3f}s\nV: {y:.4f}')

    def _on_filter_change(self, params: dict):
        self._filter_params = params
        self._refresh()

    def _on_open(self):
        path = filedialog.askopenfilename(
            filetypes=[('EMG files', '*.csv *.txt *.mat'), ('All', '*.*')])
        if not path: return
        self.console.log(f'Відкриваю: {path}')
        self._set_loading(True)
        threading.Thread(target=self._load_thread, args=(path,), daemon=True).start()

    def _load_thread(self, path):
        result = self.loader.load_data(path)
        self.root.after(0, self._on_load_done, result, path)

    def _on_load_done(self, result, path):
        self._set_loading(False)
        if result is not True:
            self.console.log(f'ПОМИЛКА: {result}'); return
        self._file_var.set(path.split('/')[-1].split('\\')[-1])
        self._build_channel_buttons()
        self.console.log(f'Завантажено: {len(self.loader.channels)} каналів, fs≈{self.loader.fs}Hz')
        self._refresh()

    def _on_new(self):
        self.loader.clear()
        self._plot_left.clear()
        self._plot_right.clear()
        self._pipeline_panel._draw_empty()
        self._analysis_panel._draw_empty()
        for w in self._ch_scroll.winfo_children(): w.destroy()
        self._ch_vars.clear()
        ttk.Label(self._ch_scroll, text='Завантажте файл...', foreground='gray').pack()
        self._file_var.set('—')
        self.console.log('Очищено.')

    def _build_channel_buttons(self):
        for w in self._ch_scroll.winfo_children(): w.destroy()
        self._ch_vars.clear()
        channels = list(self.loader.channels.keys())
        for i, ch in enumerate(channels):
            var = tk.BooleanVar(value=(i == 0))
            self._ch_vars[ch] = var
            ttk.Checkbutton(self._ch_scroll, text=ch, variable=var,
                            style='Toolbutton', command=self._refresh).pack(fill=tk.X, pady=1)

    def _refresh(self):
        if not self.loader.current_file: return
        selected = [ch for ch, v in self._ch_vars.items() if v.get()]
        if not selected: self._plot_left.clear(); return

        params = self._filter_params
        t_ch = {}
        for ch in selected:
            t, v = self.loader.get_channel_data(ch)
            if v is not None:
                t_ch[ch] = (t, v)

        # Signal tab — лівий панель: raw
        left_data = {}
        for i, (ch, (t, v)) in enumerate(t_ch.items()):
            c = COLORS[i % len(COLORS)]
            left_data[ch] = {'x': t, 'y': v, 'color': c, 'alpha': 0.5, 'lw': 0.8}
        self._plot_left.plot(left_data, xlabel='час, с', ylabel='V')

        # Правий панель (split): filtered
        if self._split:
            right_data = {}
            for i, (ch, (t, v)) in enumerate(t_ch.items()):
                try:
                    vf = self.proc.butterworth(v, self.loader.fs,
                                               params['low'], params['high'], params['notch'])
                except Exception:
                    vf = v
                env = self.proc.envelope(vf, params['envelope'])
                c = COLORS[i % len(COLORS)]
                right_data[f'{ch} filt'] = {'x': t, 'y': vf,  'color': c, 'alpha': 0.4, 'lw': 0.7}
                right_data[f'{ch} env']  = {'x': t, 'y': env, 'color': c, 'alpha': 1.0, 'lw': 1.5}
            self._plot_right.plot(right_data, xlabel='час, с', ylabel='V')

        # Pipeline tab
        if selected and t_ch:
            ch0   = selected[0]
            t0, v0 = t_ch[ch0]
            try:
                stages = self.proc.pipeline_stages(v0, self.loader.fs, params)
            except Exception as e:
                stages = {'raw': v0, 'filtered': v0, 'window': v0[:200]}

            # Перезібрати всі вікна
            win_ms  = params.get('window_ms', 200)
            step_ms = params.get('step_ms', 100)
            win_s   = int(win_ms * self.loader.fs / 1000)
            step_s  = int(step_ms * self.loader.fs / 1000)
            filt    = stages['filtered']
            windows = [filt[i:i+win_s] for i in range(0, len(filt)-win_s+1, step_s)]

            self._pipeline_panel.update(t0, stages, windows, self.loader.fs)

    def _on_load_dataset(self):
        if not HAS_PROPOSALS:
            self.console.log('emg_proposals.py не знайдено.'); return

        dlg = tk.Toplevel(self.root)
        dlg.title('Load Dataset')
        dlg.resizable(False, False)
        dlg.grab_set()

        kind_var = tk.StringVar(value='wyoflex')
        path_var = tk.StringVar()
        pat_var  = tk.StringVar(value='all')
        mov_var  = tk.StringVar(value='1 2 3 4 5')
        subj_var = tk.StringVar(value='1 2 3')

        ttk.Label(dlg, text='Тип датасету:').grid(row=0, column=0, sticky='w', padx=8, pady=4)
        for col, (lbl, val) in enumerate([('WyoFlex', 'wyoflex'), ('Ninapro DB8', 'ninapro')]):
            ttk.Radiobutton(dlg, text=lbl, variable=kind_var, value=val).grid(
                row=0, column=col+1, padx=4)

        ttk.Label(dlg, text='Папка:').grid(row=1, column=0, sticky='w', padx=8)
        ttk.Entry(dlg, textvariable=path_var, width=36).grid(row=1, column=1, columnspan=2, sticky='ew')
        ttk.Button(dlg, text='…', width=3,
                   command=lambda: path_var.set(
                       filedialog.askdirectory(title='Оберіть папку датасету'))
                   ).grid(row=1, column=3, padx=2)

        ttk.Label(dlg, text='Пацієнти/Суб\'єкти:').grid(row=2, column=0, sticky='w', padx=8)
        ttk.Entry(dlg, textvariable=pat_var, width=10).grid(row=2, column=1, sticky='w')
        ttk.Label(dlg, text='(число або all)', font=('', 8), foreground='gray').grid(row=2, column=2, sticky='w')

        ttk.Label(dlg, text='Рухи (WyoFlex):').grid(row=3, column=0, sticky='w', padx=8)
        ttk.Entry(dlg, textvariable=mov_var, width=20).grid(row=3, column=1, columnspan=2, sticky='w')

        def _load():
            path = path_var.get().strip()
            if not path:
                self.console.log('Вкажіть папку датасету.'); dlg.destroy(); return
            kind = kind_var.get()
            pat  = pat_var.get().strip()
            n    = None if pat.lower() == 'all' else int(pat) if pat.isdigit() else 'all'

            self.console.log(f'Load Dataset: {kind}, {path}')
            dlg.destroy()
            self._set_loading(True)

            def _task():
                try:
                    from emg_proposals import load_wyoflex, load_ninapro
                    if kind == 'wyoflex':
                        movs = [int(x) for x in mov_var.get().split() if x.isdigit()] or None
                        rec  = load_wyoflex(path, movements=movs, max_patients=n)
                    else:
                        subs = list(range(1, (n or 3) + 1))
                        rec  = load_ninapro(path, subjects=subs, skip_rest=True)
                    self.root.after(0, self._on_dataset_loaded, rec)
                except Exception as e:
                    self.root.after(0, self._on_dataset_error, str(e))

            threading.Thread(target=_task, daemon=True).start()

        ttk.Button(dlg, text='Завантажити', command=_load).grid(
            row=4, column=0, columnspan=4, pady=10)

    def _on_dataset_loaded(self, record):
        self._set_loading(False)
        self._labeled_record = record

        self.loader.channels.clear()
        self.loader.fs = record.fs
        t = np.arange(record.signal.shape[1]) / record.fs
        for i in range(record.n_channels):
            self.loader.channels[f'ch{i+1}'] = {
                'time':    t,
                'voltage': record.signal[i].astype(float),
            }
        self.loader.current_file = record.dataset

        self._file_var.set(f'{record.dataset}  {record.n_channels}ch  {record.fs}Hz')
        self._build_channel_buttons()
        self._refresh()

        n_cls = len(record.class_names)
        self.console.log(f'Dataset завантажено: {record.dataset}, '
                         f'{record.n_channels} кан, {n_cls} класи, '
                         f'{record.duration_sec:.0f}s — готово до Run LDA/CNN')

    def _on_dataset_error(self, msg):
        self._set_loading(False)
        self.console.log(f'Dataset помилка: {msg}')

    def _run_lda(self):
        if not HAS_PROPOSALS:
            self.console.log('emg_proposals.py не знайдено.'); return

        # Пріоритет: labeled dataset > поточний відкритий файл
        if self._labeled_record is not None:
            record = self._labeled_record
            self.console.log(f'LDA: запуск на {record.dataset} '
                             f'({record.n_channels} кан, {len(record.class_names)} класи)...')
        else:
            selected = [ch for ch, v in self._ch_vars.items() if v.get()]
            if not selected:
                self.console.log('Виберіть канали або завантажте датасет через "Load Dataset".'); return
            record = self.loader.to_emg_record(selected)
            self.console.log('LDA: запуск на поточному сигналі (без міток)...')

        self._set_loading(True)

        def _task():
            result = self.proc.lda_features(record, self._filter_params)
            self.root.after(0, self._on_lda_done, result)

        threading.Thread(target=_task, daemon=True).start()

    def _on_lda_done(self, result):
        self._set_loading(False)
        if result is None:
            self.console.log('LDA: помилка (emg_proposals недоступний).'); return

        self._analysis_panel.update_lda(result)
        self.nb.select(2)

        mode = result.get('mode', 'unknown')
        if mode == 'labeled' and result.get('f1_mean') is not None:
            self.console.log(
                f'LDA: F1={result["f1_mean"]:.3f}±{result["f1_std"]:.3f}  '
                f'({result["n_windows"]} вікон, {result["n_classes"]} класи, '
                f'{result["n_features"]} ознак)'
            )
        else:
            self.console.log(
                f'LDA: показано ознаки без класифікації '
                f'({result["n_windows"]} вікон, {result["n_features"]} ознак) — '
                f'завантажте датасет з мітками через "Load Dataset"'
            )

    def _toggle_split(self):
        self._split = not self._split
        if self._split:
            self._plot_right.grid()
            self._split_btn.config(text='Split ⬛')
        else:
            self._plot_right.grid_remove()
            self._split_btn.config(text='Split ⬜')
        self._refresh()

    def _run_cnn(self):
        if not HAS_PROPOSALS:
            self.console.log('emg_proposals.py не знайдено.'); return
        try:
            import torch
        except ImportError:
            self.console.log('PyTorch не встановлено: pip install torch --index-url https://download.pytorch.org/whl/cu121')
            return

        record = self._labeled_record
        if record is None:
            self.console.log('Завантажте датасет з мітками через "📂 Load Dataset".')
            return
        if len(np.unique(record.labels[record.labels >= 0])) < 2:
            self.console.log('Потрібно мінімум 2 класи для CNN.')
            return

        self.console.log(f'CNN: запуск на {record.dataset} '
                         f'({record.n_channels} кан, {len(record.class_names)} класи)...')
        self.nb.select(2)
        self._set_loading(True)

        params = self._filter_params.copy()
        n_epochs = 30
        threading.Thread(target=self._cnn_thread,
                         args=(record, params, n_epochs), daemon=True).start()

    def _cnn_thread(self, record, params, n_epochs):
        """Тренування CNN у фоновому потоці з epoch-callback."""
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score
        from emg_proposals import (butterworth_filter, segment_record,
                                   normalize_windows, filter_labeled, build_cnn_lstm)

        try:
            # Препроцесинг
            filtered = butterworth_filter(record)
            X, y     = segment_record(filtered,
                                      window_ms=params.get('window_ms', 200),
                                      step_ms=params.get('step_ms', 100))
            X = normalize_windows(X)
            X, y = filter_labeled(X, y)

            if len(y) < 20:
                self.root.after(0, self._on_cnn_error, f'Замало вікон ({len(y)}).')
                return

            n_cls   = len(np.unique(y))
            X_pt    = torch.tensor(X,  dtype=torch.float32)
            y_pt    = torch.tensor(y,  dtype=torch.long)

            X_tr, X_val, y_tr, y_val = train_test_split(
                X_pt, y_pt, test_size=0.2, stratify=y, random_state=42
            )

            model, device = build_cnn_lstm(record.n_channels, X.shape[2], n_cls)
            crit  = nn.CrossEntropyLoss()
            opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

            pin = device.type == 'cuda'
            tr_loader  = DataLoader(TensorDataset(X_tr, y_tr), batch_size=256,
                                    shuffle=True, pin_memory=pin, num_workers=0)
            val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256,
                                    pin_memory=pin, num_workers=0)

            history   = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
            best_acc  = 0.0
            best_state = None
            patience  = 0

            for epoch in range(1, n_epochs + 1):
                # Train
                model.train()
                tr_loss = tr_corr = tr_tot = 0
                for xb, yb in tr_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    out  = model(xb)
                    loss = crit(out, yb)
                    loss.backward(); opt.step()
                    tr_loss += loss.item() * len(yb)
                    tr_corr += (out.argmax(1) == yb).sum().item()
                    tr_tot  += len(yb)

                # Val
                model.eval()
                vl_loss = vl_corr = vl_tot = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        out = model(xb)
                        vl_loss += crit(out, yb).item() * len(yb)
                        vl_corr += (out.argmax(1) == yb).sum().item()
                        vl_tot  += len(yb)

                history['loss'].append(tr_loss / tr_tot)
                history['accuracy'].append(tr_corr / tr_tot)
                history['val_loss'].append(vl_loss / vl_tot)
                history['val_accuracy'].append(vl_corr / vl_tot)

                val_acc = vl_corr / vl_tot
                self.root.after(0, self._on_cnn_epoch, epoch, dict(history))

                if val_acc > best_acc:
                    best_acc   = val_acc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience   = 0
                else:
                    patience  += 1
                    if patience >= 7:
                        break

            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                y_pred = model(X_val.to(device)).argmax(1).cpu().numpy()
            y_val_np = y_val.numpy()
            f1 = f1_score(y_val_np, y_pred, average='macro', zero_division=0)

            # GradCAM (gradient saliency) для одного прикладу кожного класу.
            # cuDNN LSTM backward вимагає train() режиму — перемикаємо лише для градієнтів.
            gradcam_data = []
            model.train() 
            for cls in range(min(n_cls, 6)):
                idx = np.where(y_val_np == cls)[0]
                if len(idx) == 0: continue
                sample = X_val[idx[0:1]].clone().to(device).requires_grad_(True)
                out    = model(sample)
                model.zero_grad()
                out[0, cls].backward()
                # Saliency = середнє абсолютне значення градієнту по каналах
                saliency = sample.grad[0].abs().mean(dim=0).detach().cpu().numpy()
                saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

                # Передбачення (eval для детермінізму)
                model.eval()
                with torch.no_grad():
                    pred_cls = int(model(X_val[idx[0:1]].to(device)).argmax(1).item())
                model.train()

                cls_name = record.class_names[cls] if cls < len(record.class_names) else str(cls)
                gradcam_data.append({
                    'signal':     X_val[idx[0], 0].numpy(),
                    'saliency':   saliency,
                    'true_cls':   cls,
                    'pred_cls':   pred_cls,
                    'class_name': cls_name,
                })
            model.eval()

            final = {
                'history':     history,
                'val_accuracy': best_acc,
                'f1':           f1,
                'y_val':        y_val_np,
                'y_pred':       y_pred,
                'gradcam_data': gradcam_data,
                'class_names':  record.class_names,
                'n_classes':    n_cls,
            }
            self.root.after(0, self._on_cnn_done, final)

        except Exception as e:
            import traceback
            self.root.after(0, self._on_cnn_error, traceback.format_exc())

    def _on_cnn_epoch(self, epoch: int, history: dict):
        self._analysis_panel.update_cnn_training(epoch, history)
        self._status_var.set(f'CNN epoch {epoch}  val_acc={history["val_accuracy"][-1]:.3f}')

    def _on_cnn_done(self, result: dict):
        self._set_loading(False)
        self._analysis_panel.update_cnn_final(result)
        self.console.log(
            f'CNN: F1={result["f1"]:.3f}  val_acc={result["val_accuracy"]:.3f}  '
            f'({result["n_classes"]} класи)  '
            f'GradCAM: {len(result["gradcam_data"])} прикладів'
        )

    def _on_cnn_error(self, msg: str):
        self._set_loading(False)
        for line in msg.splitlines():
            self.console.log(line)

    def _next_gradcam(self):
        """Перехід до наступного прикладу GradCAM."""
        self._analysis_panel.next_gradcam()

    def _set_loading(self, state: bool):
        self._loading = state
        if state:
            self._pbar.start(12)
            self._status_var.set('Обробка...')
        else:
            self._pbar.stop()
            self._status_var.set('Готово')

    def _on_command(self, cmd: str):
        parts = cmd.strip().split()
        if not parts: return
        c = parts[0].lower()

        if c == 'help':
            self.console.log('Команди: set [param] [val] | filter | lda | status | clear')
        elif c == 'status':
            self.console.log(f'Файл: {self.loader.current_file or "Файл відсутній"}  '
                             f'Каналів: {len(self.loader.channels)}  '
                             f'fs: {self.loader.fs}Hz')
        elif c == 'filter':
            p = self._filter_params
            self.console.log(f'bp=[{p["low"]},{p["high"]}]Hz notch={p["notch"]}Hz '
                             f'win={p["window_ms"]}ms step={p["step_ms"]}ms')
        elif c == 'lda':
            self._run_lda()
        elif c == 'clear':
            self.console.text.config(state=tk.NORMAL)
            self.console.text.delete('1.0', tk.END)
            self.console.text.config(state=tk.DISABLED)
        elif c == 'set' and len(parts) == 3:
            key, val = parts[1], parts[2]
            if key in self._filter_params:
                try:
                    self._filter_params[key] = type(self._filter_params[key])(val)
                    if key in self._filter_panel._vars:
                        self._filter_panel._vars[key].set(int(val))
                    self.console.log(f'Встановлено {key}={val}')
                    self._refresh()
                except ValueError:
                    self.console.log(f'Неправильне значення для {key}')
            else:
                self.console.log(f'Невідомий параметр: {key}')
        else:
            self.console.log(f'Невідома команда. Введіть "help".')


if __name__ == '__main__':
    root = tk.Tk()
    app  = VisualizerApp(root)
    root.mainloop()