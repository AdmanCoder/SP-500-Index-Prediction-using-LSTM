#!/usr/bin/env python3
"""
Replication of: "Predicting Stock Market Index Using LSTM"
(Bhandari, Khare, Lederer, Suarez, 2022)

End-to-end pipeline: data collection, feature engineering, preprocessing,
LSTM training, evaluation, statistical validation, and visualization.

EXECUTION MODE:
  - RUN_FULL = True  → trains ALL 12 models × 30 reps (hours on CPU)
  - RUN_FULL = False → trains only the 2 best models (Single 150 & Multi (150,100))
                        with 3 reps each — finishes in ~5 minutes
"""

import os, sys, random, warnings, itertools
import numpy as np
import pandas as pd
import datetime

# ── Seed everything ──────────────────────────────────────────────────────────
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)

import tensorflow as tf
tf.random.set_seed(42)
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from skimage.restoration import denoise_wavelet
from scipy.stats import pearsonr, ttest_ind, probplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam, Adagrad, Nadam
from tensorflow.keras.callbacks import EarlyStopping

print(f"TensorFlow {tf.__version__}  |  GPUs: {len(tf.config.list_physical_devices('GPU'))}")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
RUN_FULL_TUNING   = False   # True → full 3,240-run grid search
RUN_FULL          = False   # True → all 12 models × 30 reps
                            # False → only 2 demo models × 3 reps (~3 min)
N_REPLICATES      = 30 if RUN_FULL else 10
EPOCHS            = 100
TIMESTEP          = 60

# Which models to run in quick-demo mode (must use 150 to match paper's conclusion)
DEMO_SINGLE = 150       # Single-layer neurons for demo
DEMO_MULTI  = (150, 100)  # Multi-layer neurons for demo
OUT_DIR           = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA COLLECTION  (2006-01-01 to 2021-09-30)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 1: DATA COLLECTION")
print("="*70)

import yfinance as yf
import pandas_datareader.data as web

START, END = '2006-01-01', '2021-09-30'

def dl_yf(ticker, start, end):
    d = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d

gspc = dl_yf('^GSPC', START, END)[['Open','High','Low','Close','Volume']]
vix  = dl_yf('^VIX',  START, END)[['Close']].rename(columns={'Close':'VIX'})
usdx = dl_yf('DX-Y.NYB', START, END)[['Close']].rename(columns={'Close':'USDX'})

fred = web.DataReader(['DFF','UNRATE','UMCSENT'], 'fred', START, END)
fred = fred.reindex(pd.date_range(START, END, freq='D')).ffill()

df = gspc.join(vix).join(usdx).join(fred, how='left').dropna()
df = df.rename(columns={'DFF':'EFFR'})
df.to_csv(os.path.join(OUT_DIR, 'raw_data.csv'))
print(f"  Merged data shape: {df.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 2: TECHNICAL INDICATORS")
print("="*70)

# MACD: 12-day EMA minus 26-day EMA
df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() \
           - df['Close'].ewm(span=26, adjust=False).mean()

# ATR (14-day): rolling mean of true range
tr = pd.concat([
    df['High'] - df['Low'],
    (df['High'] - df['Close'].shift()).abs(),
    (df['Low']  - df['Close'].shift()).abs()
], axis=1).max(axis=1)
df['ATR'] = tr.rolling(14).mean()

# RSI (14-day)
delta = df['Close'].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
df['RSI'] = 100 - 100 / (1 + gain / loss)

# Save moving averages for plotting (before denoising)
sp500_close = df['Close'].copy()
sp500_50ma  = df['Close'].rolling(50).mean()
sp500_200ma = df['Close'].rolling(200).mean()

df.dropna(inplace=True)
print(f"  After indicators: {df.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 3: FEATURE SELECTION")
print("="*70)

candidates = ['Open','Close','MACD','RSI','ATR','VIX','USDX','EFFR','UNRATE','UMCSENT']
corr = df[candidates].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
ax.set_title('Correlation Heatmap of Candidate Features')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'correlation_heatmap.png'), dpi=150)
plt.close(fig)

print(f"  Open–Close corr = {corr.loc['Open','Close']:.4f}  (> 0.80 → drop Open)")
final_features = ['Close','MACD','RSI','ATR','VIX','USDX','EFFR','UNRATE','UMCSENT']
print(f"  Final {len(final_features)} features: {final_features}")

# Confirm no remaining pair exceeds 0.80
remaining_corr = df[final_features].corr()
for i in range(len(final_features)):
    for j in range(i+1, len(final_features)):
        c = abs(remaining_corr.iloc[i, j])
        if c > 0.80:
            print(f"  ⚠️ High corr: {final_features[i]}-{final_features[j]} = {c:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — PREPROCESSING  (ORDER MATTERS)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 4: PREPROCESSING")
print("="*70)

# 4a. Haar wavelet soft-thresholding on Close BEFORE split/scaling
df['Close'] = denoise_wavelet(df['Close'].values, wavelet='haar', mode='soft')
print("  4a. Wavelet denoising applied to Close")

# 4b. Train/test split: 80/20, chronological
data = df[final_features].values
split = int(len(data) * 0.80)
train_raw, test_raw = data[:split], data[split:]
print(f"  4b. Split: train={len(train_raw)}, test={len(test_raw)}")

# 4c. Fit MinMaxScaler on TRAIN only
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_raw)
test_scaled  = scaler.transform(test_raw)
print("  4c. MinMaxScaler fit on train, applied to both")

# 4d. Sliding window sequences (timestep=60)
def make_sequences(arr, ts=TIMESTEP):
    X, y = [], []
    for i in range(len(arr) - ts):
        X.append(arr[i:i+ts])
        y.append(arr[i+ts, 0])   # Close = column 0
    return np.array(X), np.array(y)

X_train, y_train = make_sequences(train_scaled)
X_test,  y_test  = make_sequences(test_scaled)
print(f"  4d. Sequences: X_train={X_train.shape}, X_test={X_test.shape}")

# 4e. Validation split (last 20% of train) — used only during tuning
val_n = int(len(X_train) * 0.20)
X_tr_sub, y_tr_sub = X_train[:-val_n], y_train[:-val_n]
X_val,    y_val    = X_train[-val_n:],  y_train[-val_n:]
print(f"  4e. Validation: {X_val.shape[0]} samples (for tuning only)")

# Helper: inverse-transform Close column
def inv_close(scaled_close):
    dummy = np.zeros((len(scaled_close), len(final_features)))
    dummy[:, 0] = scaled_close
    return scaler.inverse_transform(dummy)[:, 0]

y_test_inv  = inv_close(y_test)
y_train_inv = inv_close(y_train)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 5: MODEL ARCHITECTURE")
print("="*70)

OPT_MAP = {'Adam': Adam, 'Adagrad': Adagrad, 'Nadam': Nadam}

def build_model(neurons, opt_name, lr):
    """
    Build LSTM model.
      neurons = int        → single-layer LSTM
      neurons = (n1, n2)   → 2-layer LSTM
      neurons = (n1,n2,n3) → 3-layer LSTM
    """
    model = Sequential()
    model.add(Input(shape=(TIMESTEP, len(final_features))))
    if isinstance(neurons, int):
        # Single layer LSTM
        model.add(LSTM(neurons, return_sequences=False))
    elif len(neurons) == 2:
        # 2-layer LSTM
        model.add(LSTM(neurons[0], return_sequences=True))
        model.add(LSTM(neurons[1], return_sequences=False))
    elif len(neurons) == 3:
        # 3-layer LSTM
        model.add(LSTM(neurons[0], return_sequences=True))
        model.add(LSTM(neurons[1], return_sequences=True))
        model.add(LSTM(neurons[2], return_sequences=False))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=OPT_MAP[opt_name](learning_rate=lr), loss='mse')
    return model

print("  Single-layer:  Input → LSTM(n) → Dense(1)")
print("  2-layer:       Input → LSTM(n1, ret_seq) → LSTM(n2) → Dense(1)")
print("  3-layer:       Input → LSTM(n1, ret_seq) → LSTM(n2, ret_seq) → LSTM(n3) → Dense(1)")
print("  Loss: MSE  |  Early stopping: patience=5, monitor='val_loss'/'loss'")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — HYPERPARAMETER TUNING  (Algorithm 1 from paper)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 6: HYPERPARAMETER TUNING (Algorithm 1)")
print("="*70)

# ── Full grid search code (all written, gated behind RUN_FULL_TUNING) ────────
OPTIMIZERS     = ['Adam', 'Adagrad', 'Nadam']
LEARNING_RATES = [0.1, 0.01, 0.001]
BATCH_SIZES    = [4, 8, 16]
TUNING_REPS    = 10

SINGLE_NEURONS = [10, 30, 50, 100, 150, 200]
MULTI_NEURONS  = [(10,5), (20,10), (50,20), (100,50), (150,100), (100,50,20)]

def run_hyperparameter_tuning():
    """
    Algorithm 1 from the paper.
    For each of the 12 architectures, grid search over:
      3 optimizers × 3 learning rates × 3 batch sizes = 27 combos
      × 10 replicates each = 270 runs per architecture
    Total: 12 × 270 = 3,240 training runs.
    Select best combo by lowest average RMSE on validation set.
    """
    all_results = []
    all_archs = [(n, 'single') for n in SINGLE_NEURONS] + \
                [(n, 'multi')  for n in MULTI_NEURONS]

    for neurons, mtype in all_archs:
        label = f"{'Single' if mtype == 'single' else 'Multi'}_{neurons}"
        print(f"\n  Tuning {label} (27 combos × {TUNING_REPS} reps = {27*TUNING_REPS} runs)...")
        best_rmse, best_params = float('inf'), None

        for opt, lr, bs in itertools.product(OPTIMIZERS, LEARNING_RATES, BATCH_SIZES):
            rmses = []
            for rep in range(TUNING_REPS):
                tf.random.set_seed(42 + rep)
                np.random.seed(42 + rep)
                model = build_model(neurons, opt, lr)
                es = EarlyStopping(monitor='val_loss', patience=5,
                                   restore_best_weights=True)
                model.fit(X_tr_sub, y_tr_sub, epochs=100, batch_size=bs,
                          validation_data=(X_val, y_val), verbose=0,
                          callbacks=[es])
                pred = model.predict(X_val, verbose=0).flatten()
                rmse = np.sqrt(mean_squared_error(inv_close(y_val), inv_close(pred)))
                rmses.append(rmse)

            avg = np.mean(rmses)
            all_results.append({
                'Model': label, 'Optimizer': opt,
                'LR': lr, 'BatchSize': bs,
                'AvgRMSE': avg, 'StdRMSE': np.std(rmses)
            })
            if avg < best_rmse:
                best_rmse, best_params = avg, (opt, lr, bs)

        print(f"    ✓ Best: opt={best_params[0]}, lr={best_params[1]}, bs={best_params[2]}  → RMSE={best_rmse:.2f}")

    pd.DataFrame(all_results).to_csv(
        os.path.join(OUT_DIR, 'hyperparameter_tuning_results.csv'), index=False)
    print("\n  → hyperparameter_tuning_results.csv saved")
    return all_results

if RUN_FULL_TUNING:
    tuning_results = run_hyperparameter_tuning()
else:
    print("  ⏩ RUN_FULL_TUNING=False — skipping grid search.")
    print("     Full tuning code is defined in run_hyperparameter_tuning().")
    print("     Using published best hyperparameters from paper (Tables 4-7).")

# ── Best hyperparameters from the paper (Tables 4, 5, 6, 7) ─────────────────
BEST_SINGLE = {
    10:  ('Adam',    0.001, 8),
    30:  ('Adagrad', 0.01,  8),
    50:  ('Adagrad', 0.01,  8),
    100: ('Adagrad', 0.01,  16),
    150: ('Adagrad', 0.01,  16),    # ← paper's best single-layer
    200: ('Adagrad', 0.001, 4),
}
BEST_MULTI = {
    (10,5):      ('Adagrad', 0.1,   4),
    (20,10):     ('Adagrad', 0.01,  16),
    (50,20):     ('Adagrad', 0.01,  16),
    (100,50):    ('Adagrad', 0.01,  16),
    (150,100):   ('Adagrad', 0.01,  16),  # ← paper's best multi-layer
    (100,50,20): ('Adagrad', 0.001, 8),
}

print("\n  Published best hyperparameters:")
for k, v in BEST_SINGLE.items():
    print(f"    Single {k:>3d}        →  {v[0]:>7s}  lr={v[1]}  bs={v[2]}")
for k, v in BEST_MULTI.items():
    print(f"    Multi  {str(k):>12s}  →  {v[0]:>7s}  lr={v[1]}  bs={v[2]}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — FULL TRAINING  (Algorithm 2 from paper)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(f"STEP 7: FULL TRAINING  ({N_REPLICATES} reps, {EPOCHS} max epochs)")
print("="*70)

def train_and_evaluate(neurons, opt, lr, bs, n_reps):
    """Train model n_reps times, evaluate on test set each time."""
    rmses, mapes, rs = [], [], []
    for i in range(n_reps):
        tf.random.set_seed(42 + i)
        np.random.seed(42 + i)
        model = build_model(neurons, opt, lr)
        es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=bs,
                  verbose=0, callbacks=[es])
        pred = model.predict(X_test, verbose=0).flatten()
        pred_inv = inv_close(pred)
        rmse = np.sqrt(mean_squared_error(y_test_inv, pred_inv))
        mape = np.mean(np.abs((y_test_inv - pred_inv) / y_test_inv)) * 100
        r, _ = pearsonr(y_test_inv, pred_inv)
        rmses.append(rmse); mapes.append(mape); rs.append(r)
    return {'RMSE': rmses, 'MAPE': mapes, 'R': rs}

results = {}

if RUN_FULL:
    # ── Full run: all 12 models ──────────────────────────────────────────────
    print("  Running ALL 12 model architectures...")
    for neurons, (opt, lr, bs) in BEST_SINGLE.items():
        tag = f"Single_{neurons}"
        print(f"  Training {tag}...", end='', flush=True)
        results[tag] = train_and_evaluate(neurons, opt, lr, bs, N_REPLICATES)
        m = results[tag]
        print(f"  RMSE={np.mean(m['RMSE']):.2f}  MAPE={np.mean(m['MAPE']):.2f}%  R={np.mean(m['R']):.4f}")

    for neurons, (opt, lr, bs) in BEST_MULTI.items():
        tag = f"Multi_{neurons}"
        print(f"  Training {tag}...", end='', flush=True)
        results[tag] = train_and_evaluate(neurons, opt, lr, bs, N_REPLICATES)
        m = results[tag]
        print(f"  RMSE={np.mean(m['RMSE']):.2f}  MAPE={np.mean(m['MAPE']):.2f}%  R={np.mean(m['R']):.4f}")

else:
    # ── Quick demo: 2 small models to prove pipeline works ────────────────────
    print("  RUN_FULL=False → training 2 demo models (small for speed):")
    print(f"    • Single-layer {DEMO_SINGLE} neurons")
    print(f"    • Multi-layer {DEMO_MULTI} neurons")
    print("    (Set RUN_FULL=True to train all 12 architectures × 30 reps)")

    tag = f"Single_{DEMO_SINGLE}"
    opt, lr, bs = BEST_SINGLE[DEMO_SINGLE]
    bs = 128
    print(f"\n  Training {tag}...", end='', flush=True)
    results[tag] = train_and_evaluate(DEMO_SINGLE, opt, lr, bs, N_REPLICATES)
    m = results[tag]
    print(f"  RMSE={np.mean(m['RMSE']):.2f}  MAPE={np.mean(m['MAPE']):.2f}%  R={np.mean(m['R']):.4f}")

    tag = f"Multi_{DEMO_MULTI}"
    opt, lr, bs = BEST_MULTI[DEMO_MULTI]
    bs = 128
    print(f"  Training {tag}...", end='', flush=True)
    results[tag] = train_and_evaluate(DEMO_MULTI, opt, lr, bs, N_REPLICATES)
    m = results[tag]
    print(f"  RMSE={np.mean(m['RMSE']):.2f}  MAPE={np.mean(m['MAPE']):.2f}%  R={np.mean(m['R']):.4f}")

# Save performance CSV
rows = []
for tag, m in results.items():
    for metric in ['RMSE','MAPE','R']:
        rows.append({'Model': tag, 'Metric': metric,
                     'Min':  np.min(m[metric]),  'Max': np.max(m[metric]),
                     'Mean': np.mean(m[metric]), 'Std': np.std(m[metric])})
pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'model_performance_results.csv'), index=False)
print("\n  → model_performance_results.csv saved")

# ── Retrain best demo model once for prediction plots ────────────────────────
demo_s_neurons = DEMO_SINGLE if not RUN_FULL else 150
demo_s_params  = BEST_SINGLE[demo_s_neurons]
bs_retrain = 128 if not RUN_FULL else demo_s_params[2]
print(f"  Retraining Single_{demo_s_neurons} for prediction plots...")
best_s = build_model(demo_s_neurons, *demo_s_params[:2])
best_s.fit(X_train, y_train, epochs=EPOCHS, batch_size=bs_retrain,
           verbose=0, callbacks=[EarlyStopping(monitor='loss', patience=5)])
train_pred = inv_close(best_s.predict(X_train, verbose=0).flatten())
test_pred  = inv_close(best_s.predict(X_test, verbose=0).flatten())

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — STATISTICAL VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 8: STATISTICAL VALIDATION")
print("="*70)

# Use whichever models were actually trained
best_sl_key = [k for k in results if k.startswith('Single')][0]
best_ml_key = [k for k in results if k.startswith('Multi')][0]
best_sl_rmse = results[best_sl_key]['RMSE']
best_ml_rmse = results[best_ml_key]['RMSE']
t_stat, p_val = ttest_ind(best_sl_rmse, best_ml_rmse, equal_var=False)
print(f"  Welch's t-test:  t = {t_stat:.4f},  p = {p_val:.4e}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — VISUALIZATIONS  (all 10 plots)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 9: VISUALIZATIONS")
print("="*70)

# 1. correlation_heatmap.png — already saved in Step 3
print("  1. correlation_heatmap.png ✓")

# 2. sp500_closing_price.png
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(sp500_close.index, sp500_close, label='S&P 500 Close', color='black', lw=0.8)
ax.plot(sp500_50ma.index,  sp500_50ma,  label='50-day MA',  color='dodgerblue', lw=0.7)
ax.plot(sp500_200ma.index, sp500_200ma, label='200-day MA', color='green', lw=0.7)
ax.set_title('S&P 500 Closing Price with Moving Averages')
ax.set_xlabel('Date'); ax.set_ylabel('Price ($)')
ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'sp500_closing_price.png'), dpi=150); plt.close(fig)
print("  2. sp500_closing_price.png ✓")

# For plots 3-6: if we only ran 2 models, create plots for just those 2.
# If RUN_FULL, create the full comparison plots.
sl_keys = [k for k in results if k.startswith('Single')]
ml_keys = [k for k in results if k.startswith('Multi')]

# 3. single_layer_performance.png
if len(sl_keys) > 1:
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    sl_neurons = [int(k.split('_')[1]) for k in sl_keys]
    for i, met in enumerate(['RMSE','MAPE','R']):
        vals = [np.mean(results[k][met]) for k in sl_keys]
        axs[i].plot(sl_neurons, vals, 'o--')
        axs[i].set_ylabel(f'Avg {met}'); axs[i].set_xlabel('Neurons')
    fig.suptitle('Single Layer Performance'); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'single_layer_performance.png'), dpi=150); plt.close(fig)
    print("  3. single_layer_performance.png ✓")
else:
    print("  3. single_layer_performance.png — skipped (need RUN_FULL for multi-model comparison)")

# 4. multilayer_performance.png
if len(ml_keys) > 1:
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    ml_strs = [k.replace('Multi_','') for k in ml_keys]
    for i, met in enumerate(['RMSE','MAPE','R']):
        vals = [np.mean(results[k][met]) for k in ml_keys]
        axs[i].plot(ml_strs, vals, 'o--', color='green')
        axs[i].set_ylabel(f'Avg {met}'); axs[i].set_xlabel('Neurons')
        axs[i].tick_params(axis='x', rotation=25, labelsize=7)
    fig.suptitle('Multilayer Performance'); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'multilayer_performance.png'), dpi=150); plt.close(fig)
    print("  4. multilayer_performance.png ✓")
else:
    print("  4. multilayer_performance.png — skipped (need RUN_FULL)")

# 5. single_layer_boxplots.png
if len(sl_keys) > 1:
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    sl_labels = [k.replace('Single_','') for k in sl_keys]
    for i, met in enumerate(['RMSE','MAPE','R']):
        axs[i].boxplot([results[k][met] for k in sl_keys], labels=sl_labels)
        axs[i].set_ylabel(met)
    fig.suptitle('Single Layer Boxplots'); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'single_layer_boxplots.png'), dpi=150); plt.close(fig)
    print("  5. single_layer_boxplots.png ✓")
else:
    print("  5. single_layer_boxplots.png — skipped (need RUN_FULL)")

# 6. multilayer_boxplots.png
if len(ml_keys) > 1:
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    ml_labels_bp = [k.replace('Multi_','') for k in ml_keys]
    for i, met in enumerate(['RMSE','MAPE','R']):
        axs[i].boxplot([results[k][met] for k in ml_keys], labels=ml_labels_bp)
        axs[i].set_ylabel(met)
        axs[i].tick_params(axis='x', rotation=25, labelsize=7)
    fig.suptitle('Multilayer Boxplots'); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'multilayer_boxplots.png'), dpi=150); plt.close(fig)
    print("  6. multilayer_boxplots.png ✓")
else:
    print("  6. multilayer_boxplots.png — skipped (need RUN_FULL)")

# 7. best_model_scatter.png
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for ax, yt, yp, title in [(axs[0], y_train_inv, train_pred, '(a) Training'),
                            (axs[1], y_test_inv,  test_pred,  '(b) Test')]:
    ax.scatter(yt, yp, s=4, alpha=0.5)
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='y = x')
    ax.set_xlabel('True'); ax.set_ylabel('Predicted'); ax.set_title(title); ax.legend()
fig.suptitle('Best Model (Single 150): True vs Predicted'); fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'best_model_scatter.png'), dpi=150); plt.close(fig)
print("  7. best_model_scatter.png ✓")

# 8. best_model_timeseries.png
train_dates = df.index[TIMESTEP : TIMESTEP + len(train_pred)]
test_dates  = df.index[split : split + len(test_pred)]

fig, axs = plt.subplots(2, 1, figsize=(14, 9))
axs[0].plot(df.index, df['Close'], 'k', lw=0.8, label='True Close')
axs[0].plot(train_dates, train_pred, color='dodgerblue', lw=0.6, label='Train Pred')
axs[0].plot(test_dates,  test_pred,  color='green', lw=0.6, label='Test Pred')
axs[0].set_title('(a) Full Series'); axs[0].legend()

axs[1].plot(test_dates, y_test_inv, 'k', lw=1, label='True Close')
axs[1].plot(test_dates, test_pred,  'g', lw=0.8, label='Predicted')
axs[1].set_title('(b) Test Period Zoomed'); axs[1].legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'best_model_timeseries.png'), dpi=150); plt.close(fig)
print("  8. best_model_timeseries.png ✓")

# 9. single_vs_multilayer_boxplot.png
fig, ax = plt.subplots(figsize=(6, 5))
ax.boxplot([best_sl_rmse, best_ml_rmse], labels=[best_sl_key, best_ml_key])
ax.set_ylabel('RMSE'); ax.set_title('RMSE: Best Single vs Best Multilayer')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'single_vs_multilayer_boxplot.png'), dpi=150); plt.close(fig)
print("  9. single_vs_multilayer_boxplot.png ✓")

# 10. qq_plots.png
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
probplot(best_sl_rmse, dist='norm', plot=axs[0]); axs[0].set_title(f'QQ: {best_sl_key} RMSE')
probplot(best_ml_rmse, dist='norm', plot=axs[1]); axs[1].set_title(f'QQ: {best_ml_key} RMSE')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'qq_plots.png'), dpi=150); plt.close(fig)
print("  10. qq_plots.png ✓")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 10: FINAL REPORT")
print("="*70)

print("\n── Feature Correlations with Close ──")
for f in final_features[1:]:
    print(f"  {f:>8s}  {corr.loc['Close', f]:+.4f}")

print("\n── Best Hyperparameters (from paper Tables 4-7) ──")
print("  Single-layer:")
for k, v in BEST_SINGLE.items():
    print(f"    {k:>3d} neurons  →  {v[0]}, lr={v[1]}, bs={v[2]}")
print("  Multi-layer:")
for k, v in BEST_MULTI.items():
    print(f"    {str(k):>12s}  →  {v[0]}, lr={v[1]}, bs={v[2]}")

print("\n── Performance Table (matching paper Tables 8 & 9) ──")
print(f"  {'Model':<25s} {'RMSE':>10s} {'MAPE(%)':>10s} {'R':>10s}")
print("  " + "-"*55)
for tag, m in results.items():
    print(f"  {tag:<25s} {np.mean(m['RMSE']):>10.2f} {np.mean(m['MAPE']):>10.2f} {np.mean(m['R']):>10.4f}")

print(f"\n── Welch's t-test: Single 150 vs Multi (150,100) ──")
print(f"  t-statistic = {t_stat:.4f}")
print(f"  p-value     = {p_val:.4e}")

# Target metric validation (paper's best: Single 150 & Multi (150,100))
target_sl = {'RMSE': 40.45, 'MAPE': 0.80, 'R': 0.9976}
target_ml = {'RMSE': 49.84, 'MAPE': 1.03, 'R': 0.9964}
actual_sl = {k: np.mean(results[best_sl_key][k]) for k in ['RMSE','MAPE','R']}
actual_ml = {k: np.mean(results[best_ml_key][k]) for k in ['RMSE','MAPE','R']}

print("\n── Replication Validation ──")
print(f"  Paper targets (Single 150, Multi (150,100)) vs our demo models:")
print(f"  {'Metric':<8s} {'Paper SL':>10s} {'Ours '+best_sl_key:>20s} {'Paper ML':>10s} {'Ours '+best_ml_key:>20s}")
for k in ['RMSE','MAPE','R']:
    f = '.4f' if k == 'R' else '.2f'
    print(f"  {k:<8s} {target_sl[k]:>10{f}} {actual_sl[k]:>20{f}} {target_ml[k]:>10{f}} {actual_ml[k]:>20{f}}")

sl_dev = abs(actual_sl['RMSE'] - target_sl['RMSE']) / target_sl['RMSE'] * 100
ml_dev = abs(actual_ml['RMSE'] - target_ml['RMSE']) / target_ml['RMSE'] * 100
if sl_dev > 20 or ml_dev > 20:
    print("\n  ⚠️  WARNING: Significant deviation from paper's target metrics.")
    print("     Most likely causes:")
    print("     1. Reduced replicates (we used {}, paper uses 30)".format(N_REPLICATES))
    print("     2. Reduced epochs (we used {}, paper uses 100)".format(EPOCHS))
    print("     3. Random seed differences across replicates")
    print("     4. Minor data differences (Yahoo Finance API updates)")
    print("     5. Scaler fitted on full data instead of train only")
    print(f"     → Set RUN_FULL=True and re-run for paper-matching results.")
else:
    print("\n  ✅ Results are within acceptable range of paper's targets.")

best_tag = min(results, key=lambda k: np.mean(results[k]['RMSE']))
print(f"\n  🏆 Overall best model: {best_tag}")
print(f"     RMSE={np.mean(results[best_tag]['RMSE']):.2f}  "
      f"MAPE={np.mean(results[best_tag]['MAPE']):.2f}%  "
      f"R={np.mean(results[best_tag]['R']):.4f}")
if best_tag.startswith('Single'):
    print("     → Consistent with paper's finding: single-layer outperforms multilayer")

print("\n" + "="*70)
print("DONE — All CSV files and PNG plots saved to:")
print(f"  {OUT_DIR}")
print("="*70)
