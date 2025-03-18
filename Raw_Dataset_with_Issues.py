import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

df = pd.read_csv('Raw_Dataset_with_Issues.csv')
print(df.head())

#1. IDENTIFYING MISSING VALUES
missing_columns = df.columns[df.isnull().any()].tolist()
print(f"Columns with missing values: {missing_columns}")

#FILLING MISSING VALUES
df_mean_filled = df.copy()
df_mean_filled['Age'].fillna(df_mean_filled['Age'].mean(), inplace=True)
df_mean_filled['Salary'].fillna(df_mean_filled['Salary'].mean(), inplace=True)
df_mean_filled['City'].fillna('Unknown', inplace=True)
df_mean_filled['Gender'].fillna('Unknown', inplace=True)
df_mean_filled['Marital_Status'].fillna('Unknown', inplace=True)
df_mean_filled['Education'].fillna('Unknown', inplace=True)

# 2. IDENTIFYING DUPLICATES
duplicates = df[df.duplicated()]
print(f"\nDuplicate Rows:\n{duplicates}")
#HANDLING DUPLICATES
df_no_duplicates = df.drop_duplicates()
print(df_no_duplicates)

#3. CATEGORICAL DATA STANDARDIZATION
def standardize_categorical(df, column, mapping):
    df[column] = df[column].map(mapping)
    return df

city_mapping = {
    'New York': 'New York',
    'New-York': 'New York',
    'Los Angeles': 'Los Angeles',
    'LA': 'Los Angeles',
    'Houston': 'Houston',
    'Chicago': 'Chicago',
    'Phoenix': 'Phoenix'
}

gender_mapping = {
    'Male': 'Male',
    'Female': 'Female',
    'M': 'Male',
    'F': 'Female',
    'Other': 'Other',
    'nan': np.nan
}

#4. OUTLIER DETECTION AND HANDLING
#IQR Method
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]
#Z-Score Method
def detect_outliers_zscore(df, column):
    z = np.abs(stats.zscore(df[column]))
    return df[z > 3]

#5.Create Age_Group column
def categorize_age(age):
    if 18 <= age <= 30:
        return 'Young'
    elif 31 <= age <= 50:
        return 'Middle-Aged'
    elif age >= 51:
        return 'Senior'
    else:
        return np.nan
df['Age_Group'] = df['Age'].apply(categorize_age)

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Age_Group')
plt.title('Distribution of Customers by Age Group')
plt.show()

#6. DATA TRANSFORMATION
scaler_minmax = MinMaxScaler()
scaler_standard = StandardScaler()

df['Purchase_Amount_MinMax'] = scaler_minmax.fit_transform(df[['Purchase_Amount']])
df['Purchase_Amount_Standard'] = scaler_standard.fit_transform(df[['Purchase_Amount']])

plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
sns.histplot(df['Purchase_Amount'], bins=20, kde=True)
plt.title('Original Purchase Amount')

plt.subplot(1, 3, 2)
sns.histplot(df['Purchase_Amount_MinMax'], bins=20, kde=True)
plt.title('Min-Max Scaled Purchase Amount')

plt.subplot(1, 3, 3)
sns.histplot(df['Purchase_Amount_Standard'], bins=20, kde=True)
plt.title('Standardized Purchase Amount')

plt.tight_layout()
plt.show()

#7. ECODING CATEGORICAL VARIABLES
# One-Hot Encoding
df_onehot = pd.get_dummies(df, columns=['City', 'Gender', 'Marital_Status', 'Education'])

# Label Encoding
label_encoder = LabelEncoder()
df_label_encoded = df.copy()
for column in ['City', 'Gender', 'Marital_Status', 'Education']:
    df_label_encoded[column] = label_encoder.fit_transform(df_label_encoded[column].astype(str))
    
    df_target_encoded = df.copy()
for column in ['City', 'Gender', 'Marital_Status', 'Education']:
    target_mean = df.groupby(column)['Purchase_Amount'].mean()
    df_target_encoded[column] = df[column].map(target_mean)

print("\nOne-Hot Encoded DataFrame:")
print(df_onehot.head())



print("\nTarget Encoded DataFrame:")
print(df_target_encoded.head())