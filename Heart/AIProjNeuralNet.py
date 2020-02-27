import numpy 
import math
import pandas
import csv
from random import seed
from random import random


dataset=[]; oldWeights=[]
with open('heart.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
               dataset.append(row[0:14])

#Dividing the dataset between training and testing
testingData=dataset[len(dataset)-10:]
# print(testingData)
dataset=dataset[0:len(dataset)-10]




#Appending bias with
for i in dataset:
    i.insert(0,1)
#Generating random weights // 15 refers to the number of features including the bias
seed(1)
for i in range(14):
    oldWeights.append(random())
# print(len(oldWeights))#Convering array to numpy array
oldWeights=numpy.array(oldWeights)
dataset.pop(0)
for i in dataset:
    for j in range(len(i)):
        i[j]=float(i[j])
#Calculating activation function
threshold=0
accuracyCount=0
activationFunction=0
for i in dataset:
    for j in range(14):
        activationFunction=activationFunction+(i[j]+oldWeights[j])
        #Making input features
        tempInput=numpy.array([i[0],i[1],i[2],i[3],i[4],i[5],i[6],i[7],i[8],i[9],i[10],i[11],i[12],i[13]])
    if activationFunction>threshold:
        out=1
    else:
        out=0
    tempx=tempInput*(i[-1]-out)*.1
    if (i[-1]==out):
        accuracyCount=accuracyCount+1
    else:
        newWeights=tempx+oldWeights
        oldWeights=newWeights
print("Acc Training Data: ",(accuracyCount/len(dataset))*100)
testingCount=0

#Testing DataSet






# testDataset = [[3.5,2.7,0],[7.9,5.1,1],[3.62,5.1,0], [5.9,3.8,1],[9.8,3.5,1], [1.89,4.2,0]]  
# for i in testingData:
#     i.insert(0,1)
# for i in testDataset:
#     activationFunction=(i[0]*newWeights[0])+(i[1]*newWeights[1])+(i[2]*newWeights[2])
#     #Making input features
#     tempInput=numpy.array([i[0],i[1],i[2]])
#     if activationFunction>threshold:
#         out=1
#         print('Actual','Predicted')
#         print(i[3],out)
#     else:
#         out=0
#         print('Actual','Predicted')
#         print(i[3],out)
#     if out==i[3]:
#         testingCount=testingCount+1
#     else:   
#         print('')
#     # tempx=tempInput*(i[3]-out)*.1
#     # if (i[3]==out):
#     #     accuracyCount=accuracyCount+1
#     # else:
#     #     newWeights=tempx+oldWeights
#     #     oldWeights=newWeights
        
# print(testingCount/len(testDataset)*100)










