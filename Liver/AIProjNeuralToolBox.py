from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import csv
from numpy import array

dataset=[]; target=[]
with open(r'Liver\indian_liver_patient.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
               dataset.append(row[0:len(row)])
               target.append(row[-1])

for i in dataset:
    if i[1]=='Male':
        i[1]=1
    elif i[1]=='Female':
        i[1]=0

# dataset=np.array(dataset);target=np.array(target)
dataset.pop(0);target.pop(0)
for i in dataset:
    for j in range(len(i)):
        i[j]=float(i[j])
for i in target:
    i=float(i)
dataset=np.array(dataset);target=np.array(target)
# print(dataset)




features_train, features_test, target_train, target_test = train_test_split(dataset, target, test_size= .5)
sc = StandardScaler()
sc.fit(features_train)
StandardScaler(copy=True, with_mean=True, with_std=True)

# Apply the scaler to the feature training data
features_train_std = sc.transform(features_train)

# Apply the SAME scaler to the feature test data
features_test_std = sc.transform(features_test)
ppn = Perceptron(alpha=0.01, class_weight=None, eta0=0.11, fit_intercept=False,
      max_iter=2,  n_jobs=1, penalty=None, random_state=0,
      shuffle=True, tol=None, verbose=0, warm_start=False)

# Train the perceptron
ppn.fit(features_train_std, target_train)
print(ppn.score(features_train_std,target_train))
target_pred = ppn.predict(features_test_std)
print('Accuracy: %.2f' % accuracy_score(target_test, target_pred))