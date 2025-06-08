#!/usr/bin/env python3

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# Create the mining directory if it doesn't exist
os.makedirs('mining', exist_ok=True)

# Load the test cases
print("Loading test cases from public_cases.json...")
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

# Convert to pandas DataFrame
data = []
for case in cases:
    input_data = case['input']
    data.append({
        'days': input_data['trip_duration_days'],
        'miles': input_data['miles_traveled'],
        'receipts': input_data['total_receipts_amount'],
        'reimbursement': case['expected_output']
    })

df = pd.DataFrame(data)

# Calculate derived metrics/ratios
print("Calculating derived metrics and ratios...")
df['miles_per_day'] = df['miles'] / df['days']
df['receipts_per_day'] = df['receipts'] / df['days']
df['receipts_per_mile'] = df['receipts'] / df['miles'].replace(0, np.nan)  # Avoid division by zero
df['reimbursement_per_day'] = df['reimbursement'] / df['days']
df['reimbursement_per_mile'] = df['reimbursement'] / df['miles'].replace(0, np.nan)
df['reimbursement_per_receipt'] = df['reimbursement'] / df['receipts'].replace(0, np.nan)

# Function to create scatter plots with regression line
def plot_relationship(x, y, xlabel, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot
    sns.scatterplot(x=x, y=y, alpha=0.5)
    
    # Add regression line
    sns.regplot(x=x, y=y, scatter=False, color='red')
    
    # Calculate correlation
    corr, p_value = pearsonr(x.dropna(), y.dropna())
    
    plt.title(f"{title}\nCorrelation: {corr:.3f}, p-value: {p_value:.3e}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join('mining', filename))
    plt.close()

# Create basic plots for primary variables
print("Creating basic scatter plots...")
plot_relationship(df['days'], df['reimbursement'], 
                 'Trip Duration (days)', 'Reimbursement ($)', 
                 'Reimbursement vs. Trip Duration', 'days_vs_reimbursement.png')

plot_relationship(df['miles'], df['reimbursement'], 
                 'Miles Traveled', 'Reimbursement ($)', 
                 'Reimbursement vs. Miles Traveled', 'miles_vs_reimbursement.png')

plot_relationship(df['receipts'], df['reimbursement'], 
                 'Total Receipts Amount ($)', 'Reimbursement ($)', 
                 'Reimbursement vs. Receipts Amount', 'receipts_vs_reimbursement.png')

# Create plots for derived metrics
print("Creating plots for derived metrics...")
plot_relationship(df['miles_per_day'], df['reimbursement'], 
                 'Miles per Day', 'Reimbursement ($)', 
                 'Reimbursement vs. Miles per Day', 'miles_per_day_vs_reimbursement.png')

plot_relationship(df['receipts_per_day'], df['reimbursement'], 
                 'Receipts per Day ($)', 'Reimbursement ($)', 
                 'Reimbursement vs. Receipts per Day', 'receipts_per_day_vs_reimbursement.png')

plot_relationship(df['receipts_per_mile'], df['reimbursement'], 
                 'Receipts per Mile ($)', 'Reimbursement ($)', 
                 'Reimbursement vs. Receipts per Mile', 'receipts_per_mile_vs_reimbursement.png')

# Create plots for normalized metrics
print("Creating plots for normalized metrics...")
plot_relationship(df['miles_per_day'], df['reimbursement_per_day'], 
                 'Miles per Day', 'Reimbursement per Day ($)', 
                 'Reimbursement per Day vs. Miles per Day', 'miles_per_day_vs_reimbursement_per_day.png')

plot_relationship(df['receipts_per_day'], df['reimbursement_per_day'], 
                 'Receipts per Day ($)', 'Reimbursement per Day ($)', 
                 'Reimbursement per Day vs. Receipts per Day', 'receipts_per_day_vs_reimbursement_per_day.png')

# Create plots for interesting combinations
print("Creating plots for combined metrics...")
df['miles_plus_receipts'] = df['miles'] + df['receipts']
plot_relationship(df['miles_plus_receipts'], df['reimbursement'], 
                 'Miles + Receipts', 'Reimbursement ($)', 
                 'Reimbursement vs. (Miles + Receipts)', 'miles_plus_receipts_vs_reimbursement.png')

df['miles_times_days'] = df['miles'] * df['days']
plot_relationship(df['miles_times_days'], df['reimbursement'], 
                 'Miles × Days', 'Reimbursement ($)', 
                 'Reimbursement vs. (Miles × Days)', 'miles_times_days_vs_reimbursement.png')

# Create heatmap of correlations
print("Creating correlation heatmap...")
plt.figure(figsize=(12, 10))
corr_matrix = df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title('Correlation Matrix of Trip Variables')
plt.tight_layout()
plt.savefig(os.path.join('mining', 'correlation_heatmap.png'))
plt.close()

# Create 3D scatter plot
print("Creating 3D scatter plot...")
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Scale the data for better visualization
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['days', 'miles', 'receipts']])
days_scaled = scaled_data[:, 0]
miles_scaled = scaled_data[:, 1]
receipts_scaled = scaled_data[:, 2]

# Plot the 3D scatter
scatter = ax.scatter(days_scaled, miles_scaled, receipts_scaled, c=df['reimbursement'], 
                    cmap='viridis', s=30, alpha=0.7)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Reimbursement ($)')

ax.set_xlabel('Trip Duration (scaled)')
ax.set_ylabel('Miles Traveled (scaled)')
ax.set_zlabel('Receipts Amount (scaled)')
ax.set_title('3D Relationship Between Trip Variables and Reimbursement')

plt.tight_layout()
plt.savefig(os.path.join('mining', '3d_relationship.png'))
plt.close()

# Create binned analysis
print("Creating binned analysis...")

# Bin trip duration
df['days_bin'] = pd.cut(df['days'], bins=[0, 1, 3, 7, 14, float('inf')], 
                       labels=['1 day', '2-3 days', '4-7 days', '8-14 days', '15+ days'])

# Calculate average reimbursement by trip duration bin
days_bin_avg = df.groupby('days_bin')['reimbursement'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='days_bin', y='reimbursement', data=days_bin_avg)
plt.title('Average Reimbursement by Trip Duration')
plt.xlabel('Trip Duration')
plt.ylabel('Average Reimbursement ($)')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join('mining', 'avg_reimbursement_by_days_bin.png'))
plt.close()

# Bin miles per day
df['miles_per_day_bin'] = pd.cut(df['miles_per_day'], bins=[0, 50, 100, 200, float('inf')], 
                                labels=['0-50', '51-100', '101-200', '201+'])

# Calculate average reimbursement by miles per day bin
miles_bin_avg = df.groupby('miles_per_day_bin')['reimbursement_per_day'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='miles_per_day_bin', y='reimbursement_per_day', data=miles_bin_avg)
plt.title('Average Daily Reimbursement by Miles per Day')
plt.xlabel('Miles per Day')
plt.ylabel('Average Daily Reimbursement ($)')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join('mining', 'avg_reimbursement_by_miles_per_day_bin.png'))
plt.close()

# Create a pairplot of key variables
print("Creating pairplot...")
pairplot_df = df[['days', 'miles', 'receipts', 'reimbursement', 'miles_per_day', 'receipts_per_day']]
sns.pairplot(pairplot_df, height=2.5, corner=True)
plt.suptitle('Relationships Between Trip Variables', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join('mining', 'pairplot.png'))
plt.close()

# Analyze reimbursement formulas
print("Analyzing potential reimbursement formulas...")

# Calculate different formula variations
df['formula1'] = 100 * df['days'] + 0.5 * df['miles'] + 0.8 * df['receipts']
df['formula2'] = 90 * df['days'] + 0.6 * df['miles'] + 0.9 * df['receipts']
df['formula3'] = 80 * df['days'] + 0.7 * df['miles'] + 1.0 * df['receipts']

# Plot actual vs formula reimbursements
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
sns.scatterplot(x=df['reimbursement'], y=df['formula1'], alpha=0.5)
plt.plot([0, df['reimbursement'].max()], [0, df['reimbursement'].max()], 'r--')
plt.title('Formula 1: 100×Days + 0.5×Miles + 0.8×Receipts')
plt.xlabel('Actual Reimbursement')
plt.ylabel('Formula Result')

plt.subplot(2, 2, 2)
sns.scatterplot(x=df['reimbursement'], y=df['formula2'], alpha=0.5)
plt.plot([0, df['reimbursement'].max()], [0, df['reimbursement'].max()], 'r--')
plt.title('Formula 2: 90×Days + 0.6×Miles + 0.9×Receipts')
plt.xlabel('Actual Reimbursement')
plt.ylabel('Formula Result')

plt.subplot(2, 2, 3)
sns.scatterplot(x=df['reimbursement'], y=df['formula3'], alpha=0.5)
plt.plot([0, df['reimbursement'].max()], [0, df['reimbursement'].max()], 'r--')
plt.title('Formula 3: 80×Days + 0.7×Miles + 1.0×Receipts')
plt.xlabel('Actual Reimbursement')
plt.ylabel('Formula Result')

# Calculate formula errors
df['formula1_error'] = np.abs(df['formula1'] - df['reimbursement'])
df['formula2_error'] = np.abs(df['formula2'] - df['reimbursement'])
df['formula3_error'] = np.abs(df['formula3'] - df['reimbursement'])

plt.subplot(2, 2, 4)
formula_errors = {
    'Formula 1': df['formula1_error'].mean(),
    'Formula 2': df['formula2_error'].mean(),
    'Formula 3': df['formula3_error'].mean()
}
plt.bar(formula_errors.keys(), formula_errors.values())
plt.title('Average Formula Error')
plt.ylabel('Mean Absolute Error')

plt.tight_layout()
plt.savefig(os.path.join('mining', 'formula_analysis.png'))
plt.close()

print(f"Analysis complete! All plots saved to the 'mining' directory.")

# Summary statistics
print("\nSummary Statistics:")
print(df[['days', 'miles', 'receipts', 'reimbursement', 'miles_per_day', 'receipts_per_day']].describe()) 