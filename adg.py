import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

df=pd.read_csv(r'C:\\Users\\rosal\\Downloads\\Titanic-Dataset.csv')
#print(df.duplicated().sum())
#print(df.isnull().sum())
df["Age"].fillna(df["Age"].median(), inplace=True)
#print(df.describe())
'''
sns.countplot(x='Survived', data=df,)
plt.title('Target Distribution')
plt.show()

survival_by_class = df.groupby("Pclass")["Survived"].mean()
plt.figure(figsize=(6, 4))
sns.barplot(x=survival_by_class.index, y=survival_by_class.values, palette="coolwarm")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.title("Survival Rate by Passenger Class")
plt.show()

survival_by_name = df.groupby("Name")["Survived"].mean()
plt.figure(figsize=(6, 4))
sns.barplot(x=survival_by_name.index, y=survival_by_name.values, palette="coolwarm")
plt.xlabel("Name")
plt.ylabel("Survival Rate")
plt.title("Survival Rate by Name")
plt.show()

bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]
labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80"]
df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels, include_lowest=True)
survival_by_age_group = df.groupby("AgeGroup")["Survived"].mean()
plt.figure(figsize=(8, 5))
sns.barplot(x=survival_by_age_group.index, y=survival_by_age_group.values, palette="coolwarm")
plt.xlabel("Age Group")
plt.ylabel("Survival Rate")
plt.title("Survival Rate by Age Group")
plt.show()

survival_by_sex = df.groupby("Sex")["Survived"].mean()
plt.figure(figsize=(6, 4))
sns.barplot(x=survival_by_sex.index, y=survival_by_sex.values, palette="coolwarm")
plt.xlabel("Sex")
plt.ylabel("Survival Rate")
plt.title("Survival Rate by Sex")
plt.show()

survival_by_fare = df.groupby("Fare")["Survived"].mean()
plt.figure(figsize=(6, 4))
sns.barplot(x=survival_by_fare.index, y=survival_by_fare.values, palette="coolwarm")
plt.xlabel("FAre")
plt.ylabel("Survival Rate")
plt.title("Survival Rate by Fare")
plt.show()

survival_by_sib = df.groupby("SibSp")["Survived"].mean()
plt.figure(figsize=(6, 4))
sns.barplot(x=survival_by_sib.index, y=survival_by_sib.values, palette="coolwarm")
plt.xlabel("SIblings(?)")
plt.ylabel("Survival Rate")
plt.title("Survival Rate by Siblings(?)")
plt.show()'''

df_new=df.drop(columns=['Name','Cabin','Ticket'])




le = LabelEncoder()
df_new["Sex"] = le.fit_transform(df_new["Sex"]) 

df_new = pd.get_dummies(df_new, columns=["Embarked"], drop_first=True)

#print(df_new)

scaler = StandardScaler()
df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])
X=df_new.drop('Survived',axis=1)
y=df_new['Survived']

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.25, random_state=0)

lr_model=LogisticRegression(random_state=0)
lr_model.fit(X_train, y_train)

y_lr_pred= lr_model.predict(X_test)
print(y_lr_pred)

lr_cr=classification_report(y_test, y_lr_pred)
print(lr_cr)

#models accuraacy -good- 79%. But for class1(survived)- its shit =(. Apparently, Its killed off many ppl who survived(reacll=69)

from xgboost import XGBClassifier
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
y_xgb_pred= xgb_model.predict(X_test)
print(y_xgb_pred)
xgb_cr=classification_report(y_test, y_xgb_pred)
print(xgb_cr)

#accuracy- better 82 - recall75 better at all 