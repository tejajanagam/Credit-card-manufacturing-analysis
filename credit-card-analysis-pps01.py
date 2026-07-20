# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 08:06:27 2025

@author: tejaj
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
#Load Dataset 
df = pd.read_excel(r"C:\Users\tejaj\Downloads\dataset_gsm_cards_excel -(1) (1).xlsx")
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces in column names
df.dtypes
df.head()
df.info()
df.describe()
df.shape
df.isnull().sum() 
print("Columns:", df.columns.tolist())

# Check Duplicates & Missing Values 
print("\nDuplicate rows in dataset:", df.duplicated().sum())
print("\nMissing values per column:\n", df.isnull().sum())

# Drop duplicate rows
df.drop_duplicates(inplace=True)
print("Dataset shape after dropping duplicates:", df.shape)

#Handle Missing Values 
# Fill numeric columns with median
numeric_cols = df.select_dtypes(include=['int32','int64','float64']).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)
        print(f"Filled missing values in {col} with median.")

# Fill datetime columns with mode 
datetime_cols = ['Printing Date/Time', 'Lamination Date/Time']
for col in datetime_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)
        print(f"Filled missing values in {col} with mode.")
        
#
df['Printing_Year'] = df['Printing Date/Time'].dt.year
df['Printing_Month'] = df['Printing Date/Time'].dt.month


# Fill categorical/object columns with mode
object_cols = df.select_dtypes(include=['object']).columns
for col in object_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)
        print(f"Filled missing values in {col} with mode.")

# Type Casting 
# Numeric columns: convert int/float properly
for col in numeric_cols:
    if (df[col] % 1 == 0).all():
        df[col] = df[col].astype(int)
    else:
        df[col] = df[col].astype(float)



# Outlier Capping (IQR Method) 
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

#number of outliers per column before capping
outlier_mask = ((df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound))
print("\nOutliers per numeric column before capping:\n", outlier_mask.sum())
print("Total outliers in dataset before capping:", outlier_mask.sum().sum())

# Cap the outliers
for col in numeric_cols:
    df[col] = np.where(df[col] > upper_bound[col], upper_bound[col], df[col])
    df[col] = np.where(df[col] < lower_bound[col], lower_bound[col], df[col])

#check if outliers still exist
outlier_mask_after = ((df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound))
print("\nOutliers per numeric column after capping:\n", outlier_mask_after.sum())
print("Total outliers in dataset after capping:", outlier_mask_after.sum().sum())


# Feature Engineering 
# 1. Processing time in days
df['Processing Time (Days)'] = (df['Lamination Date/Time'] - df['Printing Date/Time']).dt.days

# 2. QC Binary Columns (Pass=1, Fail=0)
qc_cols = ['Quality Control Result (Printing)', 'Quality Control Result (Embedding)', 'Quality Control Result (Personalization)']
for col in qc_cols:
    df[col + '_Binary'] = df[col].map({'Pass': 1, 'Fail': 0})

# 3. Total QC Passes
df['Total QC Passes'] = df[[col + '_Binary' for col in qc_cols]].sum(axis=1)



print("\nFinal dataset shape:", df.shape)
print("Final numeric columns:", df.select_dtypes(include=['number']).columns.tolist())
print("Missing values after preprocessing:\n", df.isnull().sum())
print("Duplicate rows after preprocessing:", df.duplicated().sum())

#saving changes
df.to_excel(r"C:\Users\tejaj\Downloads\dataset_gsm_cards_excel-cca-pps-xcel.xlsx")
df.to_csv(r"C:\Users\tejaj\Downloads\dataset_gsm_cards_excel-cca-pps.csv", index=False)
df.to_pickle(r"C:\Users\tejaj\Downloads\dataset_gsm_cards_excel-cca-pps.pkl")  # keeps datatypes
df.dtypes.to_csv(r"C:\Users\tejaj\Downloads\dtypes_info.csv")

print(" Preprocessed dataset saved successfully.")



###EDA PART ###

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

#Select numeric/business columns 

business_cols = ['No. of Sheets Used', 'No. of Cards Printed', 
                 'No. of Half Cards', 'No. of Quarter Cards', 
                 'Accepted Cards', 'Rejected Cards', 'Embedding Errors',
                 'Processing Time (Days)', 'Total QC Passes']

# Four Moments of Bussiness Decisions
moments_df = pd.DataFrame(index=business_cols)

# 1. Mean
moments_df['Mean'] = df[business_cols].mean()

# 2. Variance
moments_df['Variance'] = df[business_cols].var()

# 3. Skewness
moments_df['Skewness'] = df[business_cols].apply(skew)

# 4. Kurtosis
moments_df['Kurtosis'] = df[business_cols].apply(kurtosis)

print("Four Moments for Business Decision Columns:\n")
print(moments_df)

#  Plots 

# 1. Histograms with KDE for each numeric column
for col in business_cols:
    plt.figure(figsize=(6,3))
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f'{col} Distribution')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()

# 2. Boxplots to visualize spread and outliers
for col in business_cols:
    plt.figure(figsize=(6,3))
    sns.boxplot(x=df[col])
    plt.title(f'{col} Boxplot')
    plt.show()

# 3. Skewness and Kurtosis 
plt.figure(figsize=(12,5))
sns.barplot(x=moments_df.index, y=moments_df['Skewness'])
plt.title('Skewness of Business Columns')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12,5))
sns.barplot(x=moments_df.index, y=moments_df['Kurtosis'])
plt.title('Kurtosis of Business Columns')
plt.xticks(rotation=45)
plt.show()


























