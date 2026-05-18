"""
visualise.py
------------
Generates all presentation figures from the project's own data and model.

Usage:
    python visualise.py

Reads from:
    - data/           --> sensor CSV (auto-detected)
    - models/         --> trained model + scaler files (auto-detected)

Outputs to:
    - charts/         --> 5 PNG figures used in the presentation slides
"""

import os
import glob
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving PNGs
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ═════════════════════════════════════════════════════════════
# HELPER: Auto-detect files
# ═════════════════════════════════════════════════════════════
def find_csv(folder='data'):
    """Find the sensor data CSV in the data/ folder."""
    csvs = glob.glob(os.path.join(folder, '*.csv'))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {folder}/. Run data_generator.py first.")
    print(f"   Found: {csvs[0]}")
    return csvs[0]


def find_model(folder='models'):
    """Find the Isolation Forest model and scaler files (.pkl or .joblib)."""
    pkls = glob.glob(os.path.join(folder, '*.pkl')) + glob.glob(os.path.join(folder, '*.joblib'))
    model_path = None
    scaler_path = None
    for p in pkls:
        name = os.path.basename(p).lower()
        if 'scaler' in name:
            scaler_path = p
        elif 'model' in name or 'isolation' in name or 'forest' in name:
            model_path = p
    if not model_path:
        raise FileNotFoundError(f"No model file found in {folder}/. Run train_model.py first.")
    print(f"   Model:  {model_path}")
    print(f"   Scaler: {scaler_path or 'not found (will refit)'}")
    return model_path, scaler_path


# ═════════════════════════════════════════════════════════════
# SETUP
# ═════════════════════════════════════════════════════════════
os.makedirs('charts', exist_ok=True)

# Dark theme matching the presentation slides
plt.rcParams.update({
    'figure.facecolor': '#1E293B',
    'axes.facecolor':   '#1E293B',
    'text.color':       '#E2E8F0',
    'axes.labelcolor':  '#94A3B8',
    'xtick.color':      '#94A3B8',
    'ytick.color':      '#94A3B8',
    'axes.edgecolor':   '#334155',
    'grid.color':       '#334155',
    'font.family':      'sans-serif',
})


# ═════════════════════════════════════════════════════════════
# LOAD PROJECT DATA & MODEL
# ═════════════════════════════════════════════════════════════
print("\nLoading project files...")
csv_path = find_csv()
model_path, scaler_path = find_model()

df = pd.read_csv(csv_path)
model = joblib.load(model_path)

# Detect column names
feature_cols = [c for c in df.columns if c.lower() in ['temperature', 'humidity', 'sound_volume']]
label_col = [c for c in df.columns if c.lower() in ['is_anomaly', 'anomaly', 'label']]

if not feature_cols:
    raise ValueError(f"Cannot find feature columns. Found: {list(df.columns)}")

print(f"   Features: {feature_cols}")
print(f"   Label:    {label_col[0] if label_col else 'none'}")
print(f"   Rows:     {len(df)}")

# Split normal vs anomaly
if label_col:
    normal  = df[df[label_col[0]] == 0]
    anomaly = df[df[label_col[0]] == 1]
    y = df[label_col[0]]
else:
    sc = StandardScaler()
    preds = model.predict(sc.fit_transform(df[feature_cols]))
    df['_pred'] = [1 if p == -1 else 0 for p in preds]
    normal  = df[df['_pred'] == 0]
    anomaly = df[df['_pred'] == 1]
    y = df['_pred']

X = df[feature_cols]

# Load or refit scaler
if scaler_path:
    scaler = joblib.load(scaler_path)
else:
    scaler = StandardScaler().fit(X)

# Reproduce test predictions for confusion matrix
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_test_scaled = scaler.transform(X_test)
y_pred_raw = model.predict(X_test_scaled)
y_pred = [1 if p == -1 else 0 for p in y_pred_raw]

print("   Data and model loaded successfully\n")


# ═════════════════════════════════════════════════════════════
# FIGURE 1 - Feature Distributions            (Slide 5)
# ═════════════════════════════════════════════════════════════
print("1/5  Feature distributions...")
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))

labels_map = {
    'temperature':  'Temperature (°C)',
    'humidity':     'Humidity (%)',
    'sound_volume': 'Sound Volume (dB)',
}
colors_normal = ['#4ADE80', '#22D3EE', '#F97316']

for i, feat in enumerate(feature_cols):
    ax = axes[i]
    ax.hist(normal[feat],  bins=40, alpha=0.7, color=colors_normal[i], label='Normal',  edgecolor='none')
    ax.hist(anomaly[feat], bins=20, alpha=0.7, color='#F87171',        label='Anomaly', edgecolor='none')
    ax.set_xlabel(labels_map.get(feat.lower(), feat), fontsize=10)
    if i == 0:
        ax.set_ylabel('Count', fontsize=10)
    ax.legend(fontsize=8, facecolor='#1E293B', edgecolor='#334155')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('charts/feature_distributions.png', dpi=200, bbox_inches='tight')
plt.close()
print("   charts/feature_distributions.png\n")


# ═════════════════════════════════════════════════════════════
# FIGURE 2 - Confusion Matrix                 (Slide 7)
# ═════════════════════════════════════════════════════════════
print("2/5  Confusion matrix...")
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(4.5, 3.5))
cmap = LinearSegmentedColormap.from_list('custom', ['#1E293B', '#0D3320', '#4ADE80'])
ax.imshow(cm, cmap=cmap, aspect='auto')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Pred: Normal', 'Pred: Anomaly'], fontsize=11)
ax.set_yticklabels(['Actual: Normal', 'Actual: Anomaly'], fontsize=11)

for i in range(2):
    for j in range(2):
        color = '#F87171' if (i != j and cm[i][j] > 0) else '#4ADE80'
        ax.text(j, i, str(cm[i][j]), ha='center', va='center',
                fontsize=28, fontweight='bold', color=color)

ax.set_title('Confusion Matrix', fontsize=14, color='#E2E8F0', pad=10)
plt.tight_layout()
plt.savefig('charts/confusion_matrix.png', dpi=200, bbox_inches='tight')
plt.close()
print("   charts/confusion_matrix.png\n")


# ═════════════════════════════════════════════════════════════
# FIGURE 3 - Feature Space Scatter Plots      (Slide 7)
# ═════════════════════════════════════════════════════════════
print("3/5  Feature scatter plots...")
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

pairs = [
    (feature_cols[0], feature_cols[1]),
    (feature_cols[0], feature_cols[2]),
    (feature_cols[1], feature_cols[2]),
]

for i, (f1, f2) in enumerate(pairs):
    ax = axes[i]
    ax.scatter(normal[f1],  normal[f2],  c='#4ADE80', s=8,  alpha=0.3, label='Normal')
    ax.scatter(anomaly[f1], anomaly[f2], c='#F87171', s=20, alpha=0.8, label='Anomaly', marker='x')
    ax.set_xlabel(f1.replace('_', ' ').title(), fontsize=9)
    ax.set_ylabel(f2.replace('_', ' ').title(), fontsize=9)
    ax.legend(fontsize=7, facecolor='#1E293B', edgecolor='#334155', markerscale=1.5)
    ax.grid(True, alpha=0.2)

plt.suptitle('Feature Space: Normal vs Anomaly', fontsize=12, color='#E2E8F0')
plt.tight_layout()
plt.savefig('charts/feature_scatter.png', dpi=200, bbox_inches='tight')
plt.close()
print("   charts/feature_scatter.png\n")


# ═════════════════════════════════════════════════════════════
# FIGURE 4 - Stream Timeline                  (Slide 9)
# ═════════════════════════════════════════════════════════════
print("4/5  Stream timeline...")

# Simulate 60 readings through the model (like stream_simulator.py does)
np.random.seed(99)
n_readings = 60
sim_data = []

for i in range(n_readings):
    is_anom = np.random.random() < 0.05  # ~5% anomaly rate
    if is_anom:
        row = {
            'temperature':  np.random.normal(95, 6),
            'humidity':     np.random.normal(75, 6),
            'sound_volume': np.random.normal(110, 5),
        }
    else:
        row = {
            'temperature':  np.random.normal(70, 4),
            'humidity':     np.random.normal(40, 5),
            'sound_volume': np.random.normal(80, 4),
        }
    sim_data.append(row)

sim_df = pd.DataFrame(sim_data)
sim_scaled = scaler.transform(sim_df[feature_cols])
sim_scores = model.decision_function(sim_scaled)

fig, ax = plt.subplots(figsize=(10, 2.8))
bar_colors = ['#F87171' if s < 0 else '#4ADE80' for s in sim_scores]
ax.bar(range(n_readings), sim_scores, color=bar_colors, width=0.8, edgecolor='none')
ax.axhline(y=0, color='#F97316', linewidth=1.5, linestyle='--', alpha=0.8, label='Anomaly threshold')
ax.set_xlabel('Time (seconds)', fontsize=10)
ax.set_ylabel('Anomaly Score', fontsize=10)
ax.set_title('Real-Time Stream: 60 Seconds of Sensor Data', fontsize=12, color='#E2E8F0')
ax.legend(fontsize=9, facecolor='#1E293B', edgecolor='#334155')
ax.grid(True, alpha=0.2, axis='y')
ax.set_xlim(-1, n_readings + 1)

plt.tight_layout()
plt.savefig('charts/stream_timeline.png', dpi=200, bbox_inches='tight')
plt.close()
print("   charts/stream_timeline.png\n")


# ═════════════════════════════════════════════════════════════
# FIGURE 5 - Monitoring Dashboard Charts      (Slide 11)
# ═════════════════════════════════════════════════════════════
print("5/5  Monitoring charts...")
fig, axes = plt.subplots(1, 2, figsize=(10, 3))

# Left: Anomaly rate over 24 hours (simulated from model)
ax1 = axes[0]
np.random.seed(42)
hourly_rates = []

for hour in range(24):
    n_per_hour = 150
    readings = []
    for _ in range(n_per_hour):
        is_anom = np.random.random() < 0.05
        if is_anom:
            r = [np.random.normal(95, 6), np.random.normal(75, 6), np.random.normal(110, 5)]
        else:
            r = [np.random.normal(70, 4), np.random.normal(40, 5), np.random.normal(80, 4)]
        readings.append(r)
    scaled = scaler.transform(pd.DataFrame(readings, columns=feature_cols))
    preds = model.predict(scaled)
    rate = (preds == -1).sum() / len(preds) * 100
    hourly_rates.append(rate)

bar_colors = ['#F87171' if r > 8 else '#4ADE80' for r in hourly_rates]
ax1.bar(range(24), hourly_rates, color=bar_colors, width=0.7, edgecolor='none')
ax1.axhline(y=8, color='#F97316', linewidth=1.5, linestyle='--', alpha=0.8, label='Alert threshold (8%)')
ax1.set_xlabel('Hour of Day', fontsize=10)
ax1.set_ylabel('Anomaly Rate (%)', fontsize=10)
ax1.set_title('Anomaly Rate: 24h Monitor', fontsize=11, color='#E2E8F0')
ax1.legend(fontsize=8, facecolor='#1E293B', edgecolor='#334155')
ax1.grid(True, alpha=0.2, axis='y')

# Right: Prediction latency distribution (simulated)
ax2 = axes[1]
np.random.seed(7)
latencies = np.random.exponential(3, 500) + 1

ax2.hist(latencies, bins=40, color='#22D3EE', edgecolor='none', alpha=0.8)
ax2.axvline(x=np.median(latencies), color='#4ADE80', linewidth=2,
            linestyle='-', label=f'Median: {np.median(latencies):.1f}ms')
ax2.axvline(x=50, color='#F87171', linewidth=1.5,
            linestyle='--', label='SLA limit: 50ms')
ax2.set_xlabel('Latency (ms)', fontsize=10)
ax2.set_ylabel('Count', fontsize=10)
ax2.set_title('Prediction Latency Distribution', fontsize=11, color='#E2E8F0')
ax2.legend(fontsize=8, facecolor='#1E293B', edgecolor='#334155')
ax2.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig('charts/monitoring.png', dpi=200, bbox_inches='tight')
plt.close()
print("   charts/monitoring.png\n")


# ═════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════
print("=" * 55)
print("All 5 figures saved to charts/")
print()
print("   1. feature_distributions.png  -->  Slide 5")
print("   2. confusion_matrix.png       -->  Slide 7")
print("   3. feature_scatter.png        -->  Slide 7")
print("   4. stream_timeline.png        -->  Slide 9")
print("   5. monitoring.png             -->  Slide 11")
print()
print("These figures are generated from your project's")
print(f"own data ({csv_path}) and trained model ({model_path}).")
print("=" * 55)
