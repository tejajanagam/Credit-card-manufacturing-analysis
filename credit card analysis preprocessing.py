# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 16:57:07 2025

@author: tejaj
"""

import pandas as pd
import numpy as np

df=pd.read_excel(r"C:\Users\tejaj\Downloads\dataset_gsm_cards_excel -(1) (1).xlsx")
ds1=df.to_csv(r"C:\Users\tejaj\Downloads\dataset_gsm_cards_excel -rawcsv.csv")
df.dtypes
df.head()
df.info()
b=df.describe()
df.shape

df.isnull().sum() 

# removing duplicates
df.duplicated().sum()
df[df.duplicated(keep=False)]
df = df.drop_duplicates()


df.columns = df.columns.str.strip()

#outliers

num_df = df.select_dtypes(include=['number'])

Q1 = num_df.quantile(0.25)
Q3 = num_df.quantile(0.75)
IQR = Q3 - Q1

outliers = ((num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR)))


outlier_count = outliers.sum()
print(outlier_count)

total_outliers = outliers.sum().sum()
print("Total outliers in dataset:", total_outliers)































numeric_cols = [
    'No. of Sheets Used',
    'No. of Cards Printed',
    'No. of Half Cards',
    'No. of Quarter Cards',
    'Accepted Cards',
    'Rejected Cards',
    'Embedding Errors'
]

# four moments
four_moments = pd.DataFrame({
    'Mean' : df[numeric_cols].mean(),
    'Variance': df[numeric_cols].var(),
    'Skewnes': df[numeric_cols].skew(),
    'Kurtosis': df[numeric_cols].kurt()
})

print("\nFour Moments of Business Decisions:\n")
print(four_moments)
