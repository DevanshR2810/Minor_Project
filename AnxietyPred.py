import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# reading the dataset
df = pd.read_excel("minorprojdataset.xlsx")

# separating the features and target variable
x = df.drop(columns=["anxiety"])
y = df["anxiety"]
x.columns

df['anxiety'].value_counts()

x = x.drop(columns=['Unnamed: 3'])
x.columns

label_encoder1 = preprocessing.LabelEncoder()
label_encoder2 = preprocessing.LabelEncoder()
label_encoder3 = preprocessing.LabelEncoder()
label_encoder4 = preprocessing.LabelEncoder()
label_encoder5 = preprocessing.LabelEncoder()
x['habitat'] = x['habitat'].str.strip()
x['age'] = x['age'].str.strip()

# Encoding labels in column 'species'.
x['age'] = label_encoder1.fit_transform(x['age'])
x['education'] = label_encoder2.fit_transform(x['education'])
x['habitat'] = label_encoder3.fit_transform(x['habitat'])
x['usage'] = label_encoder4.fit_transform(x['usage'])
x['wellbeing'] = label_encoder5.fit_transform(x['wellbeing'])
# y = label_encoder1.fit_transform(y)
x.head(20)

from imblearn.over_sampling import SMOTE

smote = SMOTE()

# fit predictor and target variable
x_smote, y_smote = smote.fit_resample(x, y)

x_train, x_test, y_train, y_test = train_test_split(x_smote, y_smote, test_size=0.2, random_state=0)

gnb = GaussianNB()
gnb.fit(x_train, y_train)

# prediction using Naive Bayes
pred = gnb.predict(x_test)

# accuracy
accuracy = accuracy_score(pred, y_test)
print("naive_bayes")
print(accuracy)
print(classification_report(pred, y_test, labels=None))

age = input("Enter age group (young, mid, or old): ")
education = input("Enter educational qualification (PG or UG): ")
habitat = input("Enter habitat (urban, rural, or metropolitan): ")
usage = input("Enter media usage (low, middle, or high): ")
wellbeing = input("Enter wellbeing (low, middle, or high): ")

# # transforming user input
# label_encoder1 = preprocessing.LabelEncoder()
# label_encoder2 = preprocessing.LabelEncoder()
# label_encoder3 = preprocessing.LabelEncoder()
# label_encoder4 = preprocessing.LabelEncoder()
# label_encoder5 = preprocessing.LabelEncoder()

# Encoding labels in column 'species'.
age = label_encoder1.transform([age])
education = label_encoder2.transform([education])
habitat = label_encoder3.transform([habitat])
usage = label_encoder4.transform([usage])
wellbeing = label_encoder5.transform([wellbeing])

user_input = [age[0], education[0], habitat[0], usage[0], wellbeing[0]]
user_df = pd.DataFrame([user_input])
# making a prediction
prediction = gnb.predict(user_df)

# output
print("Your predicted anxiety level is:", prediction)
