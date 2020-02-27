from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import csv
from numpy import array

dataset=[]; target=[]
with open('data.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
               dataset.append(row[0:len(row)+1])
               target.append(row[1])

#The the target colomun in data set
for i in dataset:
    if i[1]=='M':
        i[1]=1
    elif i[1]=='B':
        i[1]=0
#The target column in target arr


#Removing the first so that the data is YES for the processing
dataset.pop(0);target.pop(0)

#Now we will delete the id coloumn in the data set and the diagnosis col at the end(For keeping the convention)
for i in dataset:
    i.remove(i[0])
    i.append(i[0]) 
    i.remove(i[0])



for i in range(len(target)):
    if target[i]=='M':
       target[i]=1
    else:
        target[i]=0


for i in dataset:
    for j in range(len(i)):
        i[j]=float(i[j])
for i in target:
    i=float(i)




dataset=np.array(dataset);target=np.array(target)
print(len(dataset),len(target))
print(dataset)
features_train, features_test, target_train, target_test = train_test_split(dataset, target, test_size=.5)
sc = StandardScaler()
sc.fit(features_train)
StandardScaler(copy=True, with_mean=True, with_std=True)

# Apply the scaler to the feature training data
features_train_std = sc.transform(features_train)

# Apply the SAME scaler to the feature test data
features_test_std = sc.transform(features_test)
ppn = Perceptron(alpha=0.00001, class_weight=None, eta0=0.1, fit_intercept=True,
      max_iter= 2,  n_jobs=1, penalty=None, random_state=0,
      shuffle=True,  verbose=0, warm_start=False)

# Train the perceptron
ppn.fit(features_train_std, target_train)

print(ppn.score(features_train_std,target_train)) 
target_pred = ppn.predict(features_test_std)
print('Accuracy: %.2f' % accuracy_score(target_test, target_pred))