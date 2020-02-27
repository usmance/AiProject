import numpy 
import math
from numpy import array
#task1
dataset = [[10,2.5,0], [10.5,3,0], [11,2,0], [11.5,2.5,0], [12.5,4,1], [12,4.5,1], [13,5,1], [13.5,5.5,1],[13.5,5.5,1]]
#seperating classes
c0=[];c1=[]
for i in dataset:
    if i[2]==0:
        c0.append(i[0:2])
    else:
        c1.append(i[0:2])

#Making mean vector 
print(c0)
temp=numpy.sum(c0,axis=0,dtype=int)
temp=temp/len(c0)
u0=temp[0:2]

temp=numpy.sum(c1,axis=0,dtype=int)
temp=temp/len(c1)
u1=temp[0:2]
print('v1== ',u0)

#Finding class probabilites
class_prob=[]
x1=len(c0)/len(dataset)
class_prob.append(x1)
x1=len(c1)/len(dataset)
class_prob.append(x1)
print(class_prob)

#Making co-variance mat
covarianceMat=0
tempx=[];tempy=[]
for i in c0:
    a=i[0]-u0[0]
    b=i[1]-u0[1]
    x=[a,b]
    tempx.append(x)
print(tempx)
for i in c1:
    a=i[0]-u1[0]
    b=i[1]-u1[1]
    x=[a,b]
    tempy.append(x)
print(tempy)
class0_f0=[];class0_f1=[];class1_f0=[];class1_f1=[]
# Calculating co-variance
for i,j in tempx:
    class0_f0.append(i); class0_f1.append(j)
PreConv=[]
PreConv.append(class0_f0);PreConv.append(class0_f1)
PreConv2=[]
for i,j in tempy:
    class1_f0.append(i); class1_f1.append(j)
PreConv2.append(class1_f0);PreConv2.append(class1_f1)

CovarianceMat0=numpy.cov(PreConv)
det_CovarianceMat0=numpy.linalg.det(CovarianceMat0)
CovarianceMat1=numpy.cov(PreConv2)
det_CovarianceMat1=numpy.linalg.det(CovarianceMat1)

print(CovarianceMat0)

#Applying the final formula 
testSample=[12,3]
test_x=testSample-u0
test_x=array(test_x)
transpose_x=numpy.transpose(test_x)*-.5

test_y=testSample-u1
test_y=array(test_y)
transpose_y=numpy.transpose(test_y)*-.5

Inverse2=numpy.linalg.inv(CovarianceMat1)
in_exp2=numpy.dot(transpose_y,Inverse2)
in_exp2=numpy.dot(in_exp2,test_y)
print(in_exp2)

Inverse=numpy.linalg.inv(CovarianceMat0)
in_exp=numpy.dot(transpose_x,Inverse)
in_exp=numpy.dot(in_exp,test_x)
print(in_exp)
#Final Formula;
p1=(1/(2*3.142*(math.sqrt(det_CovarianceMat0))))*math.exp(in_exp)
p2=(1/(2*3.142*(math.sqrt(det_CovarianceMat1))))*math.exp(in_exp2)


pofx=p1*class_prob[0]+p2*class_prob[1]
print(pofx)

#Applying Bayseian
b1=((class_prob[0])*p1)/pofx
b2=((class_prob[1])*p2)/pofx

print(b1,b2)

if b1>b2:
    print("Test Sample belongs to class 1")
else:
    print("Test Sample belongs to class 2")
