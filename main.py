import numpy 
import math
import pandas
import csv


#Test Data
# dataset=[[3.393533211,2.331273381,0],[3.110073483,1.781539638,0],[1.343808831,3.368360954,0],[3.582294042,4.67917911,0],[2.280362439,2.866990263,0],[7.423436942,4.696522875,1],[5.745051997,3.533989803,1],[9.172168622,2.511101045,1],[7.792783481,3.424088941,1],[7.939820817,0.791637231,1]]  
dataset=[]
with open('heart.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
               dataset.append(row[0:13])
for i in dataset:
    i.insert(0,1)
print(dataset)
oldWeights = [-0.1, 0.1893640140000007, -0.21018117710000003] 
oldWeights=numpy.array(oldWeights)


# #Seperating class
# classOne=[]
# classTwo=[]
# for i in dataset:
#     i.insert(0,1)
#     # if(i[2]==0):
#     #     classOne.append(i)
#     # else:
#     #     classTwo.append(i)
# print(dataset)

# #Calculating activation function
# threshold=0
# accuracyCount=0
# for i in dataset:
#     activationFunction=(i[0]*oldWeights[0])+(i[1]*oldWeights[1])+(i[2]*oldWeights[2])
#     #Making input features
#     tempInput=numpy.array([i[0],i[1],i[2]])
#     if activationFunction>threshold:
#         out=1
#     else:
#         out=0
#     tempx=tempInput*(i[3]-out)*.1
#     if (i[3]==out):
#         accuracyCount=accuracyCount+1
#     else:
#         newWeights=tempx+oldWeights
#         oldWeights=newWeights
        
#     print(newWeights)

# print("Acc Training Data: ",accuracyCount/len(dataset))
# testingCount=0
# #Testing DataSet

# testDataset = [[3.5,2.7,0],[7.9,5.1,1],[3.62,5.1,0], [5.9,3.8,1],[9.8,3.5,1], [1.89,4.2,0]]  
# for i in testDataset:
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