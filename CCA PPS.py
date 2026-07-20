# ------------------- Library Imports -------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold

df.columns = df.columns.str.strip()
df.columns
df.dtypes
# ------------------- Load Dataset -------------------
dff=pd.read_excel(r"C:\Users\tejaj\Downloads\dataset_gsm_cards_excel -(1) (1).xlsx")

# ------------------- Step 1: Data Cleaning -------------------
# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Fix missing values
df['No. of Quarter Cards'].fillna(df['No. of Quarter Cards'].median(), inplace=True)

# ------------------- Step 2: Type Casting -------------------
# Convert numeric columns appropriately
num_cols = df.select_dtypes(include=['int64','float64']).columns
for col in num_cols:
    if (df[col] % 1 == 0).all():
        df[col] = df[col].astype(int)
    else:
        df[col] = df[col].astype(float)

# Convert datetime to proper format
df['Printing Date/Time'] = pd.to_datetime(df['Printing Date/Time'])
df['Lamination Date/Time'] = pd.to_datetime(df['Lamination Date/Time'])
df.dtypes
# ------------------- Step 3: Outlier Analysis -------------------
def iqr_filter(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    return data[(data[col] >= lower) & (data[col] <= upper)]

for col in num_cols:
    df = iqr_filter(df, col)

#
num_df = df.select_dtypes(include=['number'])

Q1 = num_df.quantile(0.25)
Q3 = num_df.quantile(0.75)
IQR = Q3 - Q1

outliers = ((num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR)))


outlier_count = outliers.sum()
print(outlier_count)

total_outliers = outliers.sum().sum()
print("Total outliers in dataset:", total_outliers)

# ------------------- Step 4: Zero Variance Features -------------------
# Filter numeric columns to those currently in DataFrame
num_cols = [col for col in num_cols if col in df.columns]

print("Numeric columns for variance thresholding:", num_cols)

if len(num_cols) > 0:
    selector = VarianceThreshold(threshold=0)
    selector.fit(df[num_cols])
    constant_features = [column for column in num_cols if column not in df.columns[selector.get_support(indices=True)]]
    if len(constant_features) > 0:
        df.drop(columns=constant_features, inplace=True)
        print(f"Dropped zero variance columns: {constant_features}")
    else:
        print("No zero variance columns to drop.")
else:
    print("No numeric columns available for variance thresholding.")



# ------------------- Step 5: Feature Engineering -------------------
df['Processing Time (Days)'] = (df['Lamination Date/Time'] - df['Printing Date/Time']).dt.days
# Feature 2: Quality Control Results - convert key QC columns to numeric (Pass=1, Fail=0)
qc_cols = ['Quality Control Result (Printing)', 'Quality Control Result (Embedding)', 'Quality Control Result (Personalization)']
for col in qc_cols:
    df[col + '_Binary'] = df[col].map({'Pass': 1, 'Fail': 0})

# Feature 3: Aggregate QC Pass Count (sum of QC checks passed)
df['Total QC Passes'] = df[[col + '_Binary' for col in qc_cols]].sum(axis=1)

# Feature 4: Encoding Operators (convert operator IDs to categorical codes)
operator_cols = ['Printer Operator ID', 'Lamination Operator ID', 'Personalization Operator ID']
for col in operator_cols:
    df[col + '_Encoded'] = df[col].astype('category').cat.codes


# ------------------- Step 9: Business Moments -------------------
moment1 = df.describe().T[['mean']]               # Mean
moment2 = df.describe().T[['std']]                # Standard deviation
moment3 = df.skew().to_frame('Skewness')          # Skewness
moment4 = df.kurt().to_frame('Kurtosis')          # Kurtosis

print("First 4 Business Moments:")
print(moment1.join(moment2).join(moment3).join(moment4))

# ------------------- Step 10: Visualization -------------------
# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap")
plt.show()

# Boxplots for major metrics
plt.figure(figsize=(8,5))
sns.boxplot(data=df[['Accepted Cards', 'Rejected Cards', 'Defect Rate (%)']])
plt.title("Boxplots for Key Metrics")
plt.show()

# Distribution Plots
sns.histplot(df['Efficiency (%)'], kde=True, bins=30, color='green')
plt.title("Efficiency Distribution")
plt.show()

# ------------------- Save Final Preprocessed Dataset -------------------
df.to_csv('final_preprocessed_gsm_cards.csv', index=False)
print("\\nPreprocessing Complete! Final dataset shape:", df.shape)




import pandas as pd
import numpy as np
w=pd.read_csv(r"C:\Users\tejaj\Downloads\dataset_gsm_cards_excel-cca-pps.csv")
w.columns







c=pd.read_excel(r"C:\Users\tejaj\Downloads\dataset_gsm_cards_excel-cca-pps-xcel.xlsx")
print("\nDuplicate rows in dataset:", c.duplicated().sum())
print("\nMissing values per column:\n", c.isnull().sum())















