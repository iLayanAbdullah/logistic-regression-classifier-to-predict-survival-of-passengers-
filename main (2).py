from IPython.display import clear_output
%pip install gdown==4.5

clear_output()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
titanic_data = pd.read_csv('titanic.csv')

titanic_data= titanic_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
titanic_data = pd.get_dummies(titanic_data, columns=['Sex'])
titanic_data.head()
data_y = titanic_data.drop('Survived',axis=1)
data_x = titanic_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
X_train = X_train.values
y_train = y_train.values
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('The Accuracy is:', accuracy)
important = model.coef_[0]
feature_names = X.columns
most_important_feature = feature_names[important.argmax()]