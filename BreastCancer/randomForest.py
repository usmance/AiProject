import pandas as pd
import numpy as np
# list for column headers
# names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# open file with pd.read_csv
df = pd.read_csv("data.csv")

# print(df.shape)
# print head of data set
# print(df.head())
X = df.drop('id', axis=1)
for i in X.itertuples():
    if i[1]=='B':
        X.at[i[0],'diagnosis']=0
    else:
        X.at[i[0],'diagnosis']=1
X = df.drop('diagnosis', axis=1)


y = df['diagnosis']





from sklearn.model_selection import train_test_split
# implementing train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=66)
from sklearn.ensemble import RandomForestClassifier
# random forest model creation

rfc = RandomForestClassifier(max_depth=2, random_state=0)
rfc.fit(X_train,y_train)
# predictions
rfc_predict = rfc.predict(X_test)

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc')
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rfc_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, rfc_predict))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())