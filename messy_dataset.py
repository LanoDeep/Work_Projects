import pandas as pd

df = pd.read_csv('messy_dataset.csv')
print("Original Data:")

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Age'] = df['Age'].astype('int64')
df

df['Email'] = df['Email'].str.lower()
df.at[0, 'Email'] = 'eve@example.com'
df.loc[df['Name'] == 'David', 'Email'] = df.loc[df['Name'] == 'David', 'Email'].str.replace('example', 'example.com')
df['Email'] = df['Email'].drop_duplicates()
df

df['Salary ($)'] = df['Salary ($)'].astype(str).str.replace(',', '')
df['Salary ($)'] = df['Salary ($)'].str.replace('$', '')
df

df['Joining Date'] = df['Joining Date'].str.replace('-', '')
df.at[0, 'Joining Date'] = df.at[0, 'Joining Date'].replace('/', '')
df.at[0, 'Joining Date'] = pd.to_datetime(df.at[0, 'Joining Date'], format='%d%m%Y').strftime('%d-%m-%Y')
df.at[1, 'Joining Date'] = pd.to_datetime(df.at[1, 'Joining Date'], format='%Y%m%d').strftime('%d-%m-%Y')
df.at[2, 'Joining Date'] = pd.to_datetime(df.at[2, 'Joining Date'], format='%d%m%Y').strftime('%d-%m-%Y')
df.at[3, 'Joining Date'] = pd.to_datetime(df.at[3, 'Joining Date'], format='%Y%m%d').strftime('%d-%m-%Y')
df.at[7, 'Joining Date'] = pd.to_datetime(df.at[7, 'Joining Date'], format='%B %d, %Y').strftime('%d-%m-%Y')
df

df = df.dropna()
df['Name'] = df['Name'].drop_duplicates()
df = df.dropna()

df.to_csv('cleaned_file.csv', index=False)

print("Cleaned_Data:")
print(df.head())