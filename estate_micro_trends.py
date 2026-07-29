import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Scikit-Learn (Random Forest & Processing)
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# TensorFlow & Keras (LSTM)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, concatenate
from tensorflow.keras.callbacks import EarlyStopping

# Suppress TF and general warnings for a clean live demo console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

print("==================================================")
print("1. LOADING & PREPROCESSING ALL DATASETS")
print("==================================================")
try:
    cityone = pd.read_csv('cityone.csv')
    kingswood = pd.read_csv('kingswood.csv')
    taikoo = pd.read_csv('taikoo.csv')
    hsi = pd.read_csv('hsi.csv')
    hibor = pd.read_csv('hibor.csv')
except FileNotFoundError:
    print("CRITICAL ERROR: Missing CSV files. Ensure property, hsi.csv, and hibor.csv are present.")
    exit()

# Add Estate identifier and combine
cityone['Estate'] = 'City One'
kingswood['Estate'] = 'Kingswood Villas'
taikoo['Estate'] = 'Taikoo Shing'
df = pd.concat([cityone, kingswood, taikoo], ignore_index=True)

# Parse Dates and Sort
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
hsi['Date'] = pd.to_datetime(hsi['Date'], dayfirst=True, errors='coerce')
hibor['Date'] = pd.to_datetime(hibor['Date'], dayfirst=True, errors='coerce')

df = df.dropna(subset=['Date', 'Price', 'Size']).sort_values('Date')
hsi = hsi.dropna(subset=['Date']).sort_values('Date')
hibor = hibor.dropna(subset=['Date']).sort_values('Date')

# Filter Data up to the Valuation Target Date (July 2026)
valuation_date = pd.to_datetime('2026-07-31')
df = df[df['Date'] <= valuation_date].copy()

# Merge Macroeconomic Data (Backward fill to closest date)
df = pd.merge_asof(df, hsi, on='Date', direction='backward')
df = pd.merge_asof(df, hibor, on='Date', direction='backward')
df['HSI'] = df['HSI'].bfill().ffill()
df['HIBOR'] = df['HIBOR'].bfill().ffill()

# Create a continuous time index for the Neural Network
baseline_date = df['Date'].min()
df['Time_Index'] = (df['Date'] - baseline_date).dt.days

# Clean Floor Levels
if 'Floor' in df.columns:
    df['Floor_Original'] = df['Floor'].astype(str).str.strip()
    df['Floor'] = df['Floor_Original'].replace({'G': '0', 'G/F': '0', 'UG': '1'})
    df['Floor_Num'] = pd.to_numeric(df['Floor'].str.extract(r'(\d+)')[0], errors='coerce').fillna(10)

# Clean Distance
if 'Distance' not in df.columns:
    df['Distance'] = 500  # Fallback
df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce')

# Force all numerical features to be strictly numeric
numerical_features = ['Size', 'Floor_Num', 'View_Score', 'Distance']
for col in numerical_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows missing critical parameters
df = df.dropna(subset=numerical_features + ['Price'])

# Establish Target Value and Time Groupings
df['PSF'] = df['Price'] / df['Size']
df['Year'] = df['Date'].dt.year
df['YearMonth'] = df['Date'].dt.to_period('M')

years = np.sort(df['Year'].unique())

print("==================================================")
print("2. GENERATING OUTPUT 1 & 2: MACRO FEATURE TRENDS")
print("==================================================")
# Calculate monthly volume and weighted PSF per estate to generate Custom Index
combined_index = df.groupby(['YearMonth', 'Estate']).agg(Avg_PSF=('PSF', 'mean'), Volume=('Price', 'count')).reset_index()
total_monthly_vol = combined_index.groupby('YearMonth')['Volume'].sum().reset_index(name='Total_Volume')
combined_index = pd.merge(combined_index, total_monthly_vol, on='YearMonth')
combined_index['Weighted_PSF'] = combined_index['Avg_PSF'] * (combined_index['Volume'] / combined_index['Total_Volume'])
combined_index = combined_index.groupby('YearMonth')['Weighted_PSF'].sum().reset_index()

try:
    base_psf = combined_index.loc[combined_index['YearMonth'] == '2011-02', 'Weighted_PSF'].values[0]
except IndexError:
    base_psf = combined_index['Weighted_PSF'].iloc[0]
combined_index['Custom_Index'] = (combined_index['Weighted_PSF'] / base_psf) * 93.37

# Load the actual Centanet CCI data
try:
    cci_real = pd.read_excel('cci.xlsx')
    date_col = cci_real.columns[0]
    cci_col = next((col for col in cci_real.columns if 'CCI' in str(col).upper() or 'CCL' in str(col).upper()), cci_real.columns[1])
    cci_real = cci_real[[date_col, cci_col]].copy()
    cci_real.columns = ['Date', 'CCI']
    cci_real['Date'] = pd.to_datetime(cci_real['Date'], errors='coerce')
    cci_real['CCI'] = pd.to_numeric(cci_real['CCI'].astype(str).str.replace(',', '').str.strip(), errors='coerce')
    cci_real = cci_real.dropna()
    cci_real['YearMonth'] = cci_real['Date'].dt.to_period('M')
    cci_monthly = cci_real.groupby('YearMonth')['CCI'].mean().reset_index()
    combined_index = pd.merge(combined_index, cci_monthly[['YearMonth', 'CCI']], on='YearMonth', how='left')
    combined_index['CCI'] = combined_index['CCI'].interpolate(method='linear')
except FileNotFoundError:
    combined_index['CCI'] = np.nan

combined_index['Date_Plot'] = combined_index['YearMonth'].dt.to_timestamp()

# --- Train Yearly Random Forest Models ---
preprocessor = ColumnTransformer([
    ('num', MinMaxScaler(), ['Size', 'Floor_Num', 'View_Score', 'Distance']),
    ('cat', OneHotEncoder(sparse_output=False), ['Estate'])
])

global_importance = []
for y in years:
    df_year = df[df['Year'] == y]
    if len(df_year) < 50:
        continue
    X_processed = preprocessor.fit_transform(df_year[['Size', 'Floor_Num', 'View_Score', 'Distance', 'Estate']])
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_processed, df_year['PSF'])
    
    cat_features_out = preprocessor.named_transformers_['cat'].get_feature_names_out(['Estate']).tolist()
    all_feature_names = ['Size', 'Floor', 'View_Score', 'Distance'] + cat_features_out
    
    imp_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': rf.feature_importances_})
    imp_phys = imp_df[~imp_df['Feature'].str.startswith('Estate')].copy()
    imp_phys['Norm_Imp'] = (imp_phys['Importance'] / imp_phys['Importance'].sum()) * 100
    
    res = {'Year': y}
    for _, row in imp_phys.iterrows():
        res[row['Feature']] = row['Norm_Imp']
    global_importance.append(res)

df_trend = pd.DataFrame(global_importance).set_index('Year')

# Generate Scaled Absolute Trend
yearly_index = combined_index.groupby(combined_index['Date_Plot'].dt.year)['Custom_Index'].mean()
yearly_cci = combined_index.groupby(combined_index['Date_Plot'].dt.year)['CCI'].mean()
scaled_macro_evolution = (df_trend / 100).multiply(yearly_index, axis=0).dropna()

# --- Plot Output 1: Absolute Contribution ---
feature_colors = {'Size': '#1f77b4', 'Floor': '#ff7f0e', 'View_Score': '#2ca02c', 'Distance': '#d62728'}
abbr = {'Size': 'Size (Unit Size Effect)', 'Floor': 'Floor (Floor Level)', 'View_Score': 'View (View Quality)', 'Distance': 'Distance (Transport/MTR)'}

fig1 = plt.figure(figsize=(14, 7))
for col in scaled_macro_evolution.columns:
    plt.plot(scaled_macro_evolution.index, scaled_macro_evolution[col], marker='s', label=f"{abbr[col]}", color=feature_colors[col], linewidth=2.5)
    for x, y in zip(scaled_macro_evolution.index, scaled_macro_evolution[col]):
        plt.text(x, y + 3, f"{y:.0f}", ha='center', va='bottom', fontsize=10, color=feature_colors[col], fontweight='bold')

plt.plot(yearly_index.index, yearly_index, 'k--', label="3-Estate Custom Index", linewidth=3)
for x, y in zip(yearly_index.index, yearly_index):
    plt.text(x, y + 5, f"{y:.0f}", ha='center', va='bottom', fontsize=11, color='black', fontweight='bold')

if not yearly_cci.isna().all():
    plt.plot(yearly_cci.index, yearly_cci, color='gray', linestyle=':', marker='x', markersize=6, label="Centanet CCI (Reference)", linewidth=2)
    for x, y in zip(yearly_cci.index, yearly_cci):
        if pd.notna(y):
            lbl = f"{y:.0f}\n(July 2026)" if x == 2026 else f"{y:.0f}"
            plt.text(x, y + (12 if x == 2026 else 8), lbl, ha='center', va='bottom', fontsize=10, color='gray', fontweight='bold')

plt.ylim(bottom=-5, top=max(yearly_index.max(), yearly_cci.max() if not yearly_cci.isna().all() else 0) * 1.15)
plt.title('Absolute Feature Contribution to Price-Per-Square-Foot (Index Points)', fontsize=14, fontweight='bold')
plt.ylabel('Scaled Importance (Points)', fontsize=12, fontweight='bold')
plt.xticks(range(2011, 2027))
plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('feature_trends-final.png', bbox_inches='tight', dpi=120)
plt.close()

# --- Plot Output 2: Percentage Contribution ---
fig2, ax = plt.subplots(figsize=(16, 8))
for col in df_trend.columns:
    marker_map = {'Size': 's', 'Floor': 'o', 'View_Score': '^', 'Distance': 'D'}
    ax.plot(df_trend.index, df_trend[col], marker=marker_map[col], label=abbr[col], color=feature_colors[col], linewidth=3.5)
    for x in df_trend.index:
        y = df_trend.loc[x, col]
        y_offset, va = (1.5, 'bottom') if col in ['Size', 'Floor'] else (-1.5, 'top')
        ax.text(x, y + y_offset, f"{y:.0f}", ha='center', va=va, fontsize=10, fontweight='bold', color=feature_colors[col])

last_year = df_trend.index[-1]
for col in df_trend.columns:
    val = df_trend.loc[last_year, col]
    y_offset = 2 if col == 'Size' else (-3 if col == 'Distance' else 0)
    ax.text(last_year + 0.2, val + y_offset, f"{val:.1f}%", fontweight='bold', fontsize=11, color=feature_colors[col])

ax.set_title('Global AI Feature Importance Trend (15-Year Market Shift)', fontsize=18, fontweight='bold', pad=15)
ax.set_ylabel('Algorithmic Importance (% of Physical Pricing)', fontsize=14, fontweight='bold')
ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
ax.set_xticks(years)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
ax.axvspan(2018, 2021, color='gray', alpha=0.1)
ax.text(2019.5, 95, "Peak Market\n(LTV Squeeze)", ha='center', fontsize=11, fontweight='bold', color='dimgray')
plt.tight_layout()
plt.savefig('global_feature_importance_trend-final.png', bbox_inches='tight', dpi=120)
plt.close()

print("==================================================")
print("3. GENERATING OUTPUT 3: ESTATE MICRO DNA COMPARISON")
print("==================================================")
estates = ['Taikoo Shing', 'City One', 'Kingswood Villas']
dna_data = []

for estate in estates:
    df_est = df[df['Estate'] == estate].copy()
    X = df_est[['Size', 'Floor_Num', 'View_Score', 'Distance']]
    y = df_est['PSF']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y)
    imp = (rf.feature_importances_ / rf.feature_importances_.sum()) * 100
    dna_data.append({'Estate': estate, 'Size': imp[0], 'Floor': imp[1], 'View': imp[2], 'Distance': imp[3]})

df_dna = pd.DataFrame(dna_data).set_index('Estate')

fig3, ax = plt.subplots(figsize=(14, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
labels = ['Size (Unit Size)', 'Floor (Vertical Altitude)', 'View (Aesthetic Quality)', 'Distance (Commute)']

df_dna.plot(kind='bar', color=colors, ax=ax, width=0.8, edgecolor='white')
ax.set_title('The Micro-Psychology: AI Feature Importance by Estate (15-Year Baseline)', fontsize=18, fontweight='bold', pad=20)
ax.set_ylabel('Algorithmic Importance (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Target Estate Demographic', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)

for p in ax.patches:
    height = p.get_height()
    if height > 1:
        ax.text(p.get_x() + p.get_width()/2., height + 2, f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xticklabels(['Taikoo Shing\n(Island Prestige)', 'City One\n(Starter Utility)', 'Kingswood Villas\n(Suburban Space-Seeker)'], rotation=0, fontsize=12, fontweight='bold')
ax.legend(labels, loc='upper right', fontsize=12, framealpha=0.9)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('estate_micro_dna_comparison-final.png', bbox_inches='tight', dpi=120)
plt.close()

print("==================================================")
print("4. GENERATING OUTPUT 4: BASELINE INVENTORY")
print("==================================================")
def categorize_size(row):
    size, estate = row['Size'], row['Estate']
    if estate == 'City One': return 'Small' if size <= 350 else 'Medium' if size <= 500 else 'Large'
    elif estate == 'Kingswood Villas': return 'Small' if size <= 450 else 'Medium' if size <= 600 else 'Large'
    elif estate == 'Taikoo Shing': return 'Small' if size <= 580 else 'Medium' if size <= 750 else 'Large'
    return 'Unknown'

def categorize_view(score):
    if pd.isna(score): return 'Standard'
    return 'Premium' if score >= 7 else 'Standard' if score >= 4 else 'Adverse'

def categorize_distance(dist):
    if pd.isna(dist): return 'Mid'
    return 'Close' if dist <= 300 else 'Mid' if dist <= 600 else 'Far'

df['Size_Tier'] = df.apply(categorize_size, axis=1)
df['View_Tier'] = df['View_Score'].apply(categorize_view)
df['Floor_Tier'] = pd.cut(df['Floor_Num'], bins=[-1, 10, 20, 200], labels=['Low', 'Mid', 'High'])
df['Distance_Tier'] = df['Distance'].apply(categorize_distance)

color_high, color_mid, color_low = '#2ca02c', '#1f77b4', '#8c564b'
colors_tiers = [color_low, color_mid, color_high]
columns = ['Total Market', 'Taikoo Shing', 'City One', 'Kingswood Villas']

fig4, axes = plt.subplots(4, 4, figsize=(24, 22))
fig4.suptitle('Objective Market Inventory (15-Year Demographic Baseline)', fontsize=24, fontweight='bold', y=0.98)

def plot_pie_and_stats(ax, data, title, category_order, color_scheme, estate_title=None):
    counts = data.value_counts()
    pcts = data.value_counts(normalize=True) * 100
    sizes = [counts.get(cat, 0) for cat in category_order]
    
    ax.pie(sizes, colors=color_scheme, autopct='%1.1f%%', startangle=90, textprops={'color': "w", 'weight': 'bold', 'fontsize': 11}, wedgeprops={'edgecolor': 'w', 'linewidth': 1})
    ax.set_title(f"{estate_title}\n\n{title}" if estate_title else title, fontsize=16 if estate_title else 14, fontweight='bold', pad=20 if estate_title else 15)
    
    stats_text = "".join([f"{cat}: {counts.get(cat, 0):,} ({pcts.get(cat, 0):.1f}%)\n" for cat in category_order])
    ax.text(-0.5, 0.5, stats_text.strip(), transform=ax.transAxes, fontsize=11, verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor='gray', alpha=0.8))

for i, col_name in enumerate(columns):
    df_target = df if col_name == 'Total Market' else df[df['Estate'] == col_name]
    plot_pie_and_stats(axes[0, i], df_target['Size_Tier'], 'Size Distribution', ['Small', 'Medium', 'Large'], colors_tiers, estate_title=col_name)
    plot_pie_and_stats(axes[1, i], df_target['View_Tier'], 'View Distribution', ['Adverse', 'Standard', 'Premium'], colors_tiers)
    plot_pie_and_stats(axes[2, i], df_target['Floor_Tier'], 'Floor Distribution', ['Low', 'Mid', 'High'], colors_tiers)
    plot_pie_and_stats(axes[3, i], df_target['Distance_Tier'], 'Distance (to MTR) Distribution', ['Close', 'Mid', 'Far'], [color_high, color_mid, color_low])

plt.subplots_adjust(wspace=1.0, hspace=0.4)
plt.savefig('baseline_inventory_distribution-final.png', bbox_inches='tight', dpi=120)
plt.close()

# ==========================================
# 5. HYBRID RF-LSTM VALUATION & SIMULATION
# ==========================================
print("==================================================")
print("5. HYBRID RF-LSTM VALUATION & SIMULATION")
print("==================================================")

# Trim outliers (bottom 2%)
def trim_outliers(group):
    lower_bound = group['PSF'].quantile(0.02)
    return group[(group['PSF'] >= lower_bound)]

original_count = len(df)
df = df.groupby(['Year', 'Estate', 'Size_Tier'], group_keys=False)[df.columns].apply(trim_outliers, include_groups=False).reset_index(drop=True)
print(f"Trimmed {original_count - len(df)} low-end anomalous transactions.")

# Train Random Forest (Baseline Physical Valuation)
rf_features = ['Estate', 'Size', 'Floor_Num', 'View_Score', 'Distance', 'Time_Index']
X_rf = df[rf_features]
y_rf = df['Price']

rf_preprocessor = ColumnTransformer([
    ('num', MinMaxScaler(), ['Size', 'Floor_Num', 'View_Score', 'Distance', 'Time_Index']),
    ('cat', OneHotEncoder(sparse_output=False), ['Estate'])
])

X_rf_processed = rf_preprocessor.fit_transform(X_rf)
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)

print("Generating Out-Of-Fold (OOF) Baseline Predictions to prevent Data Leakage...")
df['RF_Baseline_Price'] = cross_val_predict(rf_model, X_rf_processed, y_rf, cv=5, n_jobs=-1)
rf_model.fit(X_rf_processed, y_rf) # Final fit for live simulation

# Prepare LSTM Sequences
print("Preparing 3-month macroeconomic sequences for LSTM...")
macro_db = df[['Date', 'HSI', 'HIBOR']].drop_duplicates('Date').sort_values('Date').set_index('Date')

def get_macro_sequence(target_date):
    dates = [target_date - pd.Timedelta(days=60), target_date - pd.Timedelta(days=30), target_date]
    seq = []
    for d in dates:
        idx = macro_db.index.asof(d)
        if pd.isna(idx): idx = macro_db.index[0]
        seq.append([macro_db.loc[idx, 'HSI'], macro_db.loc[idx, 'HIBOR']])
    return seq

sequences = np.array([get_macro_sequence(d) for d in df['Date']])
scaler_lstm = MinMaxScaler()
sequences_2d = sequences.reshape(-1, 2)
sequences_scaled = scaler_lstm.fit_transform(sequences_2d).reshape(sequences.shape)

y_price = df['Price'].values.reshape(-1, 1)
rf_baseline = df['RF_Baseline_Price'].values.reshape(-1, 1)

# --- THE FIX: Scale the Prices! ---
# Prevent the giant raw prices (Millions) from crushing the tiny macro inputs (0-1)
price_scaler = MinMaxScaler()
y_price_scaled = price_scaler.fit_transform(y_price)
rf_baseline_scaled = price_scaler.transform(rf_baseline)

# Build & Train Hybrid Keras Model
print("Building and Training Hybrid RF-LSTM Neural Network (50 Epochs)...")
X_lstm_train, X_lstm_test, X_rf_train, X_rf_test, y_train_scaled, y_test_scaled = train_test_split(
    sequences_scaled, rf_baseline_scaled, y_price_scaled, test_size=0.2, random_state=42
)
# Keep an unscaled y_test copy to calculate the real-dollar MAE later
_, _, _, _, _, y_test_raw = train_test_split(
    sequences_scaled, rf_baseline, y_price, test_size=0.2, random_state=42
)

lstm_input = Input(shape=(3, 2), name='Macro_Timeline')
lstm_layer = LSTM(32, activation='relu')(lstm_input)
lstm_dense = Dense(16, activation='relu')(lstm_layer)

rf_input = Input(shape=(1,), name='RF_Baseline_Price')
rf_dense = Dense(16, activation='relu')(rf_input)

merged = concatenate([lstm_dense, rf_dense])
dense_1 = Dense(128, activation='relu')(merged)
dropout_1 = Dropout(0.2)(dense_1)
dense_2 = Dense(64, activation='relu')(dropout_1)
dropout_2 = Dropout(0.1)(dense_2)
dense_3 = Dense(32, activation='relu')(dropout_2)
output_layer = Dense(1, activation='linear', name='Final_Valuation')(dense_3)

hybrid_model = Model(inputs=[lstm_input, rf_input], outputs=output_layer)
hybrid_model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.004),
    loss='mean_squared_error', metrics=['mean_absolute_error']
)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
hybrid_model.fit(
    [X_lstm_train, X_rf_train], y_train_scaled,
    validation_split=0.2, epochs=50, batch_size=64, callbacks=[early_stop], verbose=1
)

# Evaluation (Must inverse transform back to normal dollars)
y_pred_scaled = hybrid_model.predict([X_lstm_test, X_rf_test], verbose=0)
y_pred = price_scaler.inverse_transform(y_pred_scaled).flatten()
y_test = y_test_raw.flatten()

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"\n=== HYBRID MODEL (RF + LSTM) PERFORMANCE ===")
print(f"R² Score (Variance Explained) : {r2:.4f} ({r2*100:.2f}%)")
print(f"Mean Absolute Pct Error (MAPE): {mape:.2f}%")
print("============================================\n")

# Simulator
print("="*50)
print("🔮 SCENARIO SIMULATOR: JULY 2026 VALUATION 🔮")
print("="*50)

selected_estate = 'Kingswood Villas'
size_sqft, floor_num, view_score, calculated_distance = 540, 15, 6, 825
target_date = pd.to_datetime('2026-07-28')
time_idx = (target_date - baseline_date).days
base_hsi, base_hibor = 25000, 2.72333

scenarios = [
    {"name": "Baseline Prediction", "hsi_change": 0, "hibor_change": 0.0},
    {"name": "Stock Market Bull (+1,000 HSI)", "hsi_change": 1000, "hibor_change": 0.0},
    {"name": "Stock Market Bear (-1,000 HSI)", "hsi_change": -1000, "hibor_change": 0.0},
    {"name": "Interest Rate Cut (-0.5% HIBOR)", "hsi_change": 0, "hibor_change": -0.5},
    {"name": "Interest Rate Hike (+0.5% HIBOR)", "hsi_change": 0, "hibor_change": 0.5}
]

print(f"Target: {selected_estate} | {size_sqft} sqft | {floor_num}/F | View: {view_score}")
print(f"Macro Baseline: HSI = {base_hsi:,} | HIBOR = {base_hibor:.5f}%")
print("-" * 50)

rf_input_df = pd.DataFrame([{'Estate': selected_estate, 'Size': size_sqft, 'Floor_Num': floor_num, 'View_Score': view_score, 'Distance': calculated_distance, 'Time_Index': time_idx}])
X_rf_sim = rf_preprocessor.transform(rf_input_df)
rf_base_pred = rf_model.predict(X_rf_sim)[0]

# --- THE FIX: Scale the Simulator Base Price ---
rf_base_pred_scaled = price_scaler.transform(np.array([[rf_base_pred]]))

base_price = None
for i, scen in enumerate(scenarios):
    scen_hsi = base_hsi + scen['hsi_change']
    scen_hibor = base_hibor + scen['hibor_change']
    seq = np.array([[base_hsi, base_hibor], [base_hsi, base_hibor], [scen_hsi, scen_hibor]])
    seq_scaled = scaler_lstm.transform(seq.reshape(-1, 2)).reshape(1, 3, 2)
    
    # Predict in 0-1 scale, then inverse transform back to normal HKD
    final_price_pred_scaled = hybrid_model.predict([seq_scaled, rf_base_pred_scaled], verbose=0)
    final_price_pred = price_scaler.inverse_transform(final_price_pred_scaled)[0][0]
    
    if i == 0:
        base_price = final_price_pred
        print(f"[SCENARIO {i}] {scen['name']}:\n-> Estimated Value: HKD {final_price_pred:,.0f}")
    else:
        diff_pct = ((final_price_pred - base_price) / base_price) * 100
        sign = "+" if diff_pct > 0 else ""
        print(f"\n[SCENARIO {i}] {scen['name']}:\n-> HSI: {scen_hsi:,} | HIBOR: {scen_hibor:.5f}%\n-> Estimated Value: HKD {final_price_pred:,.0f} ({sign}{diff_pct:.2f}%)")

print("\n" + "="*60)
print("ALL 5 OUTPUTS GENERATED SUCCESSFULLY!")
print("="*60)
plt.show()
