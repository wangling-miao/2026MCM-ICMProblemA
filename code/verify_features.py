import pandas as pd

# Load the summary file
df = pd.read_csv(r'c:\Users\chenp\Documents\比赛\2026美赛\code\processed_data\all_files_features_summary.csv')

print("=" * 70)
print("数据预处理结果验证")
print("=" * 70)

print(f"\n总共处理了 {len(df)} 个数据集\n")

# Check capacity features
print("1. 容量特征 (Equation 3: Q_k = ∫I(t)dt)")
print("-" * 70)
capacity_cols = [c for c in df.columns if 'capacity' in c.lower()]
print(f"容量相关列: {capacity_cols}")
valid_capacity = df['discharge_capacity'].notna().sum()
print(f"有效放电容量数据: {valid_capacity}/{len(df)} ({100*valid_capacity/len(df):.1f}%)")
if valid_capacity > 0:
    print(f"放电容量范围: {df['discharge_capacity'].min():.3f} - {df['discharge_capacity'].max():.3f} Ah")
    print(f"平均放电容量: {df['discharge_capacity'].mean():.3f} Ah")

# Check resistance features
print("\n2. 内阻特征 (Equation 4: R(SOC, T))")
print("-" * 70)
resistance_cols = [c for c in df.columns if ('resistance' in c.lower() or 'R_' in c) and 'internal' not in c.lower()]
print(f"内阻特征列: {resistance_cols}")

# Check for R(SOC) features
soc_features = [c for c in df.columns if 'R_SOC' in c]
if soc_features:
    print(f"\nSOC相关内阻特征: {soc_features}")
    for feat in soc_features:
        valid = df[feat].notna().sum()
        if valid > 0:
            print(f"  {feat}: {valid} 个有效值, 平均 = {df[feat].mean():.6f} Ω")

# Check for R(T) features
temp_r_features = [c for c in df.columns if 'R_Temp' in c]
if temp_r_features:
    print(f"\n温度相关内阻特征: {temp_r_features}")
    for feat in temp_r_features:
        valid = df[feat].notna().sum()
        if valid > 0:
            print(f"  {feat}: {valid} 个有效值, 平均 = {df[feat].mean():.6f} Ω")

# Check ratio features
ratio_features = [c for c in df.columns if 'ratio' in c.lower()]
if ratio_features:
    print(f"\n内阻增长率特征: {ratio_features}")
    for feat in ratio_features:
        valid = df[feat].notna().sum()
        if valid > 0:
            print(f"  {feat}: {valid} 个有效值, 平均 = {df[feat].mean():.3f}")

# Temperature features
print("\n3. 温度特征")
print("-" * 70)
temp_cols = [c for c in df.columns if 'temperature' in c.lower() and 'R_' not in c]
print(f"温度相关列: {temp_cols}")

# Show example
print("\n4. 示例数据（第1个有完整特征的数据集）")
print("-" * 70)
# Find a row with most features
feature_count = df.notna().sum(axis=1)
best_idx = feature_count.idxmax()
sample = df.iloc[best_idx]

print(f"文件: {sample['file']}")
print(f"Sheet: {sample.get('sheet', 'N/A')}")
print(f"\n关键特征:")
key_features = ['discharge_capacity', 'avg_voltage', 'avg_resistance', 'avg_temperature']
for feat in key_features:
    if feat in sample and pd.notna(sample[feat]):
        print(f"  {feat}: {sample[feat]:.6f}")

print("\n" + "=" * 70)
print("验证完成！")
print("=" * 70)
