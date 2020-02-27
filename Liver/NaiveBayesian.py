import numpy as np
from numpy import array
import math
from math import pi
import csv
import random
#Test Data
dataset=[]
testDataset=[]

with open(r'Liver\indian_liver_patient.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
               dataset.append(row[0:len(row)+1])

#The the target colomun
for i in dataset:
    if i[1]=='Male':
        i[1]=1
    elif i[1]=='Female':
        i[1]=0
    if i[-1]=='1':
        i[-1]=0
    elif i[-1]=='2':
        i[-1]=1

#Removing the first so that the data is YES for the processing
dataset.pop(0)

#Now we will delete the id coloumn in the data set and the diagnosis col at the end(For keeping the convention)
# random.shuffle(dataset)
#As we read the CSV, it was in the string so converting it into the float.
for i in dataset:
    for j in range(len(i)):
        i[j]=float(i[j])

dCount=0
label=[]

for i in range(500,len(dataset)):
    testDataset.append(dataset[i])
    label.append(dataset[i][-1])
# print(testDataset)











classOne=[]
classTwo=[]
# print(dataset)
for i in dataset:
    if dCount<500:
        if (i[-1]==0):
            classOne.append(i)
        else:
            classTwo.append(i)
    dCount=dCount+1
print(len(classOne),len(classTwo),len(dataset))


# Taking mean and variance of each feature
meanClassOne=0; meanClassTwo=0; varianceClassOne=0; varianceClassTwo=0
# dataset=array(dataset); classTwo=array(classTwo); classOne=array(classOne)
classOne=array(classOne);classTwo=array(classTwo)
meanClassOne=np.mean(classOne,axis=0)
meanClassTwo=np.mean(classTwo,axis=0)

print(meanClassOne)
varianceClassOne=np.var(classOne,axis=0)
varianceClassTwo=np.var(classTwo,axis=0)
# meanClassOne.append(np.mean(classOne[0],axis=1));meanClassOne.append(np.mean(classOne[1],axis=1))
# meanClassTwo.append(np.mean(classTwo[0],axis=1));meanClassTwo.append(np.mean(classTwo[1],axis=1))
#calculating prior probability

priorProbability=[]
priorProbability.append(len(classOne)/len(dataset)); priorProbability.append(len(classTwo)/len(dataset))

# ####Applying formula 
#Calculating class conditional prob
classConditionalc1=[];classConditionalc2=[]
target=[]
abc=[]
xyz=[]
#Traversing over each sample
accCount=0

Prodc1=1; Prodc2=1
for i in testDataset:

    for j in range(len(i)-1):
        inexp=(i[j]-meanClassOne[j])*(i[j]-meanClassOne[j])
        inexp=inexp*-1
        inexp=inexp/(4*varianceClassOne[j])
        inexp=math.exp(inexp)
        inexp_prev=1/(math.sqrt(2*pi*varianceClassOne[j]))
        final=inexp_prev*inexp
        Prodc1=Prodc1*final
    abc.append(Prodc1)

    Prodc1=1
for i in testDataset:
    for j in range(len(i)-1):
        inexp=(i[j]-meanClassTwo[j])*(i[j]-meanClassTwo[j])
        inexp=inexp*-1
        inexp=inexp/(4*varianceClassTwo[j])
        inexp=math.exp(inexp)
        inexp_prev=1/(math.sqrt(2*pi*varianceClassTwo[j]))
        final=inexp_prev*inexp
        Prodc2=Prodc2*final
    xyz.append(Prodc2)

    Prodc2=1
# print(priorProbability)
abc=np.array(abc) ; xyz=np.array(xyz)
f1=abc*priorProbability[0]; f2=xyz*priorProbability[1]

for i in range(len(f1)):
    if f1[i]>f2[i]:
        target.append(0)
    else:
        target.append(1)
accCount=0
for i in range(len(target)):
    if target[i]==label[i]:
        accCount=accCount+1
    
print('Accuracy : ',(accCount/len(testDataset)*100))








#Multiplying with Prior Probability


# targetClassified=[]
# ProbProdc1=1; ProbProdc2=1
# for i in range(len(classConditionalc1)):
#     ProbProdc1=ProbProdc1*classConditionalc1[i]
# a=ProbProdc1*priorProbability[0]

# for j in range(len(classConditionalc2)):
#     ProbProdc2=ProbProdc2*classConditionalc2[i]
# b=ProbProdc2*priorProbability[1]
# if(a>b):
#     targetClassified.append(0)
# else:
#     targetClassified.append(1)


# accCount=0
# for i in range(len(dataset)):
#     if dataset[i][2]==targetClassified[i]:
#         accCount=accCount+1

# print(Prodc1,Prodc2)
# print('Accuracy',accCount/len(dataset))

    