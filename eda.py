import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

sns.set_theme(style="whitegrid")

print("General Description")

statlog_german_credit_data = fetch_ucirepo(id=144)

X = statlog_german_credit_data.data.features
y = statlog_german_credit_data.data.targets

df = pd.concat([X, y], axis=1)

df = df.rename(columns={'class': 'target'})

num_instances, num_features = df.shape
print(f"Number of instances (rows): {num_instances}")
print(f"Number of features (columns, including target): {num_features}")

print("\nFeature Types:")
print(df.dtypes)


print("\nClass Distribution")

class_counts = df['target'].value_counts()
class_percentages = df['target'].value_counts(normalize=True) * 100

dist_df = pd.DataFrame({'Count': class_counts, 'Percentage (%)': class_percentages})
print(dist_df)

plt.figure(figsize=(6, 4))

sns.countplot(data=df, x='target', hue='target', legend=False, palette='Set2')
plt.title('Distribution of Target Variable (1=Good, 2=Bad)')
plt.xlabel('Credit Risk')
plt.ylabel('Number of Instances')
plt.show(block=False)


print("\nMissing Values and Duplicates")
missing_values = df.isnull().sum()
missing_percent = (missing_values / num_instances) * 100

missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage (%)': missing_percent})
print("Missing values per feature:")
print(missing_df[missing_df['Missing Values'] > 0]) 

duplicates = df.duplicated().sum()
duplicate_percent = (duplicates / num_instances) * 100
print(f"\nNumber of duplicate rows: {duplicates} ({duplicate_percent:.2f}%)")

print("\nNumeric Features Analysis")
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('target') 

stats_df = df[numeric_cols].describe().T
print("Descriptive Statistics:")
print(stats_df[['mean', 'std', 'min', 'max', '25%', '50%', '75%']])

print("\nOutlier Analysis (IQR Method):")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}: {len(outliers)} potential outliers")

top_std_cols = stats_df['std'].nlargest(15).index
df[top_std_cols].hist(figsize=(12, 10), bins=20, edgecolor='black')
plt.suptitle('Histogram', fontsize=16)
plt.tight_layout()
plt.show(block=False)


print("\nCategorical Features Analysis")
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

for col in categorical_cols:
    print(f"\nFeature: {col}")
    cardinality = df[col].nunique()
    print(f"Cardinality: {cardinality}")
    
    proportions = df[col].value_counts(normalize=True) * 100
    print(proportions)
    
    rare_cats = proportions[proportions < 5.0].index.tolist()
    if rare_cats:
        print(f"Warning: Rare categories found (<5%): {rare_cats}")


print("\neature-Target Relationship")
df['target_numeric'] = df['target'].map({1: 0, 2: 1}) 

corr_matrix = df[numeric_cols.tolist() + ['target_numeric']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Pearson Correlation Heatmap (Numeric Features vs Target)')
plt.show(block=False)

correlations = corr_matrix['target_numeric'].drop('target_numeric').abs().sort_values(ascending=False)
print(f"\nMost informative numeric features:\n{correlations.head(3)}")
print(f"\nLeast informative numeric features:\n{correlations.tail(2)}")

top_2_features = correlations.head(2).index.tolist()

for i, feature in enumerate(top_2_features):
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x='target', y=feature, hue='target', legend=False, palette='Set2')
    plt.title(f'{feature} vs Target')
    plt.xlabel('Credit Risk (1=Good, 2=Bad)')
    
    if i == len(top_2_features) - 1:
        plt.show() 
    else:
        plt.show(block=False)