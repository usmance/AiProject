import csv
import matplotlib.pyplot as plt
# x=[1,2]
# print(len(x))
Time_Arr=[]
Current_Val=[]
with open('fyp_data_v1.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        Time_Arr.append(row[0])
        Current_Val.append(float(row[1]))

for i in range(len(Current_Val)):
    if Current_Val[i]>0 and Current_Val[i]<=0.5:
        Current_Val[i]=0
    elif Current_Val[i]>0.5 and Current_Val[i]<=1:
        Current_Val[i]=.5
    elif Current_Val[i]>1 and Current_Val[i]<=1.5:
        Current_Val[i]=1
    elif Current_Val[i]>1.5 and Current_Val[i]<=2:
        Current_Val[i]=1.5
    elif Current_Val[i]>2 and Current_Val[i]<=2.5:
        Current_Val[i]=2
    elif Current_Val[i]>2.5 and Current_Val[i]<=3:
        Current_Val[i]=2.5
    elif Current_Val[i]>3 and Current_Val[i]<=3.5:
        Current_Val[i]=3
    elif Current_Val[i]>3.5 and Current_Val[i]<=4:
        Current_Val[i]=3.5
    elif Current_Val[i]>4 and Current_Val[i]<=4.5:
        Current_Val[i]=4
    elif Current_Val[i]>4.5 and Current_Val[i]<=5:
        Current_Val[i]=4.5
   
    
    
plt.plot(Current_Val)
plt.show()