import pandas as pd
import numpy as np
# Importing data from a CSV file
data = pd.read_csv ('Salary_Data1.csv')

# Display the first few rows of the dataframe
print("Initial Data:")
print(data.head())

# Exporting data to a CSV file
data.to_csv('exported_data.csv', index=False)

# Data Pre-processing Techniques:

# 1.Handling missing values for numeric columns only
numeric_columns = data.select_dtypes(include=[np.number]).columns
data['Age'] = data['Age'].fillna(data['Age'].mean())
data['EstimatedSalary'] = data['EstimatedSalary'].fillna(data['EstimatedSalary'].mean())

# 2. Feature scaling
# Min-Max Scaling
data['Age'] = (data['Age'] - data['Age'].min()) / (data['Age'].max() - data['Age'].min())
data['EstimatedSalary'] = (data['EstimatedSalary'] - data['EstimatedSalary'].min()) / (data['EstimatedSalary'].max() - data['EstimatedSalary'].min())

# Standardization (Z-score normalization)
data['Age'] = (data['Age'] - data['Age'].mean()) / data['Age'].std()
data['EstimatedSalary'] = (data['EstimatedSalary'] - data['EstimatedSalary'].mean()) / data['EstimatedSalary'].std()

# 4. Removing duplicates
data.drop_duplicates(inplace=True)

# 5. Encoding ordinal features
# Assuming there is an ordinal column named 'Education_Level'
data['Education_Level'] = data['Education_Level'].map({'Bachelor': 1, 'Master': 2, 'PhD': 3})

# 6. One-hot encoding
# Assuming there is a categorical column named 'Gender'
data = pd.get_dummies(data, columns=['Gender'])

# Display the pre-processed data
print("\nPre-processed Data:")
print(data.head())
