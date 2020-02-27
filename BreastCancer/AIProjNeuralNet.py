import numpy 
import math
import pandas
import csv
from random import seed
from random import random


dataset=[]; oldWeights=[]

with open(r'BreastCancer\data.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
               dataset.append(row[0:len(row)+1])

#Appending bias with
for i in dataset:
    i.insert(0,1)

#The the target colomun
for i in dataset:
    if i[2]=='M':
        i[2]=1
    elif i[2]=='B':
        i[2]=0

#Removing the first so that the data is YES for the processing
dataset.pop(0)

#Now we will delete the id coloumn in the data set and the diagnosis col at the end(For keeping the convention)
for i in dataset:
    i.remove(i[1])
    i.append(i[1]) 
    i.remove(i[1])

#As we read the CSV, it was in the string so converting it into the float.
for i in dataset:
    for j in range(len(i)):
        i[j]=float(i[j])
  
#Generating random weights //  refers to the number of features including the bias
seed(1)
for i in range(len(dataset[0])-1):
    oldWeights.append(random())
# print(len(oldWeights))#Convering array to numpy array
oldWeights=numpy.array(oldWeights) 

#Calculating activation function
threshold=0
accuracyCount=0
activationFunction=0
# tempInput=[]
newWeights=[]
dCount=0


for i in dataset:
    if dCount<550:
        for j in range(len(dataset[0])-1):
            activationFunction=activationFunction+(i[j]+oldWeights[j])
            
        tempInput=[]
        for k in range(len(dataset[0])-1):
            tempInput=numpy.append(tempInput, i[k])
        # print(len(tempInput)) 
        # print(len(oldWeights))
    
        if activationFunction>threshold:
            out=1
        else:
            out=0
        print(activationFunction)
        tempx=tempInput*(i[-1]-out)*.1
        
        if (i[-1]==out):
            accuracyCount=accuracyCount+1
        else:
            newWeights=tempx+oldWeights
            oldWeights=newWeights
            # print(oldWeights)
        dCount=dCount+1

# print("Acc Training Data: ",(accuracyCount/(len(dataset)))*100)
# print(len(oldWeights))




testingCount=0
errorCount=0
# # #Testing DataSet






testDataset=[]
for i in range(550,len(dataset)):
    testDataset.append(dataset[i])

for i in testDataset:
    for j in range(len(testDataset[0])-1):
        activationFunction=activationFunction+(i[j]+newWeights[j])
    #Making input features
    for k in range(len(dataset[0])-1):
        tempInput=numpy.append(tempInput, i[k])
    if activationFunction>threshold:
        out=1
    else:
        out=0

    if out==i[-1]:
        testingCount=testingCount+1
    else:   
        errorCount=errorCount+1

        
print(testingCount/len(testDataset)*100)
# print(errorCount/len(testDataset)*100)









