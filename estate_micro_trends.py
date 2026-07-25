import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Loading updated datasets (up to July 2026)...")
# Load datasets
try:
    cityone = pd.read_csv('cityone.csv')
    kingswood = pd.read_csv('kingswood.csv')
    taikoo = pd.read_csv('taikoo.csv')
except FileNotFoundError:
    print("Please ensure cityone.csv, kingswood.csv, and taikoo.csv are in the directory.")
    exit()

# Combine datasets
cityone['Estate'] = 'City One'
kingswood['Estate'] = 'Kingswood Villas'
taikoo['Estate'] = 'Taikoo Shing'
df = pd.concat([cityone, kingswood, taikoo], ignore_index=True)

# Parse Dates and extract the Year
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['Date', 'Price', 'Size', 'View_Score'])
df['Year'] = df['Date'].dt.year

# Calculate Price Per Square Foot (PSF) to track the Market Trend
df['PSF'] = df['Price'] / df['Size']
yearly_psf = df.groupby('Year')['PSF'].mean()

# Categorize View Scores into the 3 Tiers
def categorize_view(score):
    if score >= 7:
        return 'Premium (7-9)'
    elif score >= 4:
        return 'Standard (4-6)'
    else:
        return 'Adverse (1-3)'

df['View_Tier'] = df['View_Score'].apply(categorize_view)

# Calculate transaction volume counts per year for each tier
volume_data = df.groupby(['Year', 'View_Tier']).size().unstack(fill_value=0)

# Ensure columns are in the correct order for the chart
cols = ['Adverse (1-3)', 'Standard (4-6)', 'Premium (7-9)']
volume_data = volume_data.reindex(columns=cols, fill_value=0)

# Convert raw volume to percentages (100% Stacked)
volume_pct = volume_data.div(volume_data.sum(axis=1), axis=0) * 100

print("Generating 15-Year Volume vs. Market Trend Chart...")

# --- PLOTTING ---
fig, ax1 = plt.subplots(figsize=(14, 7))

# Colors matching your presentation theme
colors = ['#f43f5e', '#94a3b8', '#deff9a'] # Red (Adverse), Gray (Standard), Green (Premium)

# 1. Plot the 100% Stacked Bar Chart on the primary Y-axis (Left)
volume_pct.plot(kind='bar', stacked=True, ax=ax1, color=colors, width=0.75, edgecolor='black', alpha=0.85)

ax1.set_ylabel('Percentage of Total Transactions (%)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 100)
ax1.set_title('15-Year Transaction Volume Breakdown vs. Market Price (Flight to Quality)', fontsize=14, fontweight='bold')

# Move the legend outside the chart
ax1.legend(title='View Tier Breakdown', loc='upper left', bbox_to_anchor=(1.05, 1))

# 2. Plot the Market Trend (Avg PSF) on a secondary Y-axis (Right)
ax2 = ax1.twinx()
# Align the X-axis for the line chart with the bar chart positions
x_positions = np.arange(len(yearly_psf.index))
ax2.plot(x_positions, yearly_psf.values, color='black', marker='D', markersize=8, linewidth=3, label='Market Price (Avg PSF)')

ax2.set_ylabel('Market Average Price Per Sqft (HKD)', fontsize=12, fontweight='bold')
ax2.set_ylim(bottom=yearly_psf.min() * 0.8, top=yearly_psf.max() * 1.1)
ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 0.8))

# Annotate the line chart with the PSF values
for i, v in enumerate(yearly_psf.values):
    ax2.text(i, v + 200, f'${v:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

# --- FOOLPROOF FIX: SAVE DIRECTLY TO ROOT FOLDER ---
# No more 'public/' folder. It saves right next to the script!
plt.savefig('feature_trends.png', bbox_inches='tight')
print("Successfully saved chart to feature_trends.png")
