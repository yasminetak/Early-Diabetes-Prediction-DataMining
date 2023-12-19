import pandas as pd
df = pd.read_csv('C:/Users/HP/Documents/GitHub/Early-Diabetes-Prediction-DataMining/diabetes.csv')
df

df.columns

df.info()

df.describe().T

"""# Data Visualization

Plotting the data distribution plots before removing null values
"""

p =df.hist(figsize = (20,20))

"""# DATA PREPROCESSING"""

# 1 Handling Missing values #
missing_counts = df.isnull().sum()
print(missing_counts)

# 2 Check the features with 0 value
zero_counts = df.eq(0).sum()
print(zero_counts)

#Replace 0 values in important features with median value
import numpy as np
import pandas as pd

df['Glucose'] = df['Glucose'].replace(0, np.median(df['Glucose']))
df['BloodPressure'] = df['BloodPressure'].replace(0, np.median(df['BloodPressure']))
df['SkinThickness'] = df['SkinThickness'].replace(0, np.median(df['SkinThickness']))
df['Insulin'] = df['Insulin'].replace(0, np.median(df['Insulin']))
df['BMI'] = df['BMI'].replace(0, np.median(df['BMI']))

df.head()

# See the data first
print("Data before adding new features")
df.head()

# 4 Feature Engineering #   #the process of transforming raw data into features that can be used to improve the performance of a machine learning model

# Create new categorical variables
NewBMI = pd.Series(["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"], dtype="category")
df["NewBMI"] = NewBMI
df.loc[df["BMI"] < 18.5, "NewBMI"] = NewBMI[0]
df.loc[(df["BMI"] > 18.5) & (df["BMI"] <= 24.9), "NewBMI"] = NewBMI[1]
df.loc[(df["BMI"] > 24.9) & (df["BMI"] <= 29.9), "NewBMI"] = NewBMI[2]
df.loc[(df["BMI"] > 29.9) & (df["BMI"] <= 34.9), "NewBMI"] = NewBMI[3]
df.loc[(df["BMI"] > 34.9) & (df["BMI"] <= 39.9), "NewBMI"] = NewBMI[4]
df.loc[df["BMI"] > 39.9 ,"NewBMI"] = NewBMI[5]


def set_insulin(row):
    if row["Insulin"] >= 16 and row["Insulin"] <= 166:
        return "Normal"
    else:
        return "Abnormal"
df = df.assign(NewInsulinScore=df.apply(set_insulin, axis=1))


NewGlucose = pd.Series(["Low", "Normal", "Overweight", "Secret", "High"], dtype = "category")
df["NewGlucose"] = NewGlucose
df.loc[df["Glucose"] <= 70, "NewGlucose"] = NewGlucose[0]
df.loc[(df["Glucose"] > 70) & (df["Glucose"] <= 99), "NewGlucose"] = NewGlucose[1]
df.loc[(df["Glucose"] > 99) & (df["Glucose"] <= 126), "NewGlucose"] = NewGlucose[2]
df.loc[df["Glucose"] > 126 ,"NewGlucose"] = NewGlucose[3]

print("Data after adding new categorical variable ")
df.head()

# 5 Convert Categorical variables using one hot encoding #

df = pd.get_dummies(df, columns =["NewBMI","NewInsulinScore", "NewGlucose"], drop_first = True)
df.head()

categorical_df = df[['NewBMI_Obesity 1','NewBMI_Obesity 2', 'NewBMI_Obesity 3', 'NewBMI_Overweight','NewBMI_Underweight',
                     'NewInsulinScore_Normal','NewGlucose_Low','NewGlucose_Normal', 'NewGlucose_Overweight', 'NewGlucose_Secret']]

categorical_df.head()

# 6  Split the dataset into features (X) and the target variable (y). The target variable is 'Outcome', and the rest are features.

y = df["Outcome"]
X = df.drop(["Outcome",'NewBMI_Obesity 1','NewBMI_Obesity 2', 'NewBMI_Obesity 3', 'NewBMI_Overweight','NewBMI_Underweight',
                     'NewInsulinScore_Normal','NewGlucose_Low','NewGlucose_Normal', 'NewGlucose_Overweight', 'NewGlucose_Secret'], axis = 1)
cols = X.columns
index = X.index

X.head()

# 7 Normalization #

# The variables in the data set are an effective factor in increasing the performance of the models by standardization.
# There are multiple standardization methods. These are methods such as" Normalize"," MinMax"," Robust" and "Scale".
from sklearn.preprocessing import RobustScaler
transformer = RobustScaler().fit(X)
X = transformer.transform(X)
X = pd.DataFrame(X, columns = cols, index = index)

X.head()

X = pd.concat([X,categorical_df], axis = 1)
X.head()

# Save preprocessed Data

processed_data_csv_path = "C:/Users/HP/Documents/GitHub/Early-Diabetes-Prediction-DataMining/processed_data.csv"
df_preprocessed = pd.concat([X, y], axis=1)  # Combine features (X) and target variable (y)
df_preprocessed

# Save to CSV
df_preprocessed.to_csv(processed_data_csv_path, index=False)