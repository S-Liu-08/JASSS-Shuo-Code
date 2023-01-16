# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 23:29:11 2023

@author: 15164
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import os


def myThread(totalNum,parentPath,a,z):


    lambda_1=0.5          #parameter lambda
    row = 1
    
    nowIndex = 0
    threshold=np.arange(0,1.01,0.01)                    #selectable threshold values
    for m,num in enumerate(threshold):
        epsilon1=num                                     #the confidence threshold
        for n in range(0,(len(threshold))):
            epsilon2=threshold[n]                       #the repulsive threshold
                    
            x =  np.zeros(totalNum,float)               #an array of opinions in the network at time t
            y =  np.zeros(totalNum,float)               #an array of opinions in the network at time t+1

            for num in range(0,totalNum):
                x[num] = z[num]                         #the initial value is loaded for each agent.
                y[num] = x[num]                         #preparing for the opinion update at time t+1.
            
            
            evo=pd.DataFrame()
            for t in range(1000):                       #the number of iteration steps 
                num=pd.DataFrame(x,columns=['time'+str(t)+''])
                evo=pd.concat([evo,num],axis=1)
                for i in range(totalNum):
                    near = []
                    for k in range(totalNum):            
                        if a[i][k] != 0:                
                            near.append(k)               #choose the neighbor k of the agent i.
                    data1= [ x[k] for k in near if round(abs(x[k]-x[i]),4)<=epsilon1 and a[i,k]==1 ]         #The neighbor opinion is within the confidence threshold and the link to the agent i is positive,Noted as type A1.
                    data2= [ x[k] for k in near if round(abs(x[k]-x[i]),4)>=epsilon2 and a[i,k]==-1 ]        #The neighbor opinion is within the repulsive threshold and the link to the agent i is negative,Noted as type A2.
                    lenA=len(data1)                      #The number of type A1
                    lenB=len(data2)                      #The number of type A2

                    if lenB != 0 or lenA != 0:           #The case where type A1 and type A2 are not both empty sets.   
                        if lenB != 0:                    #If type A2 is not the empty set.
                            if lenA == 0:                #Type A1 is the empty set.
                                e2=sum(data2)            #Sum the neighbor opinions that satisfy type A2.
                                d=e2/lenB                #Calculate the mean value of the corresponding opinions.
                                f=x[i]+lambda_1*(lenB/len(near)*(x[i]-d)) #Stay away from the neighbors who are within the repulsive threshold based on their last opinion.
                                if 0<=f<=1:              #Apply constraints to control opinions in the range of 0 to 1.
                                    y[i]=round(f,4)
                                elif f<0:
                                    y[i]=0
                                else:
                                    y[i]=1
                            else:                         #Type A1 is not the empty set.
                                e1=sum(data1)             #Sum of all opinions whose difference with agent i is less than or equal to the confidence threshold.
                                e2=sum(data2)             #Sum of all opinions whose difference with agent i is greater than or equal to the repulsive threshold.
                                c=e1/lenA                 #Calculate the mean value of the type A1 opinions.
                                d=e2/lenB                 #Calculate the mean value of the type A2 opinions.
                                f=x[i]+lambda_1*(lenA/len(near)*(c-x[i])+lenB/len(near)*(x[i]-d))   #Converge toward neighboring opinions that are within the confidence threshold and away from neighboring opinions that are within the repulsion threshold.
                                if 0<=f<=1:               #Apply constraints to control opinions in the range of 0 to 1.
                                    y[i]=round(f,4)
                                elif f<0:
                                    y[i]=0
                                else:
                                    y[i]=1    
                        else:                             #Type A2 is the empty set.
                            e1=sum(data1)                 #Sum of all opinions whose difference with agent i is less than or equal to the confidence threshold.     
                            c=e1/lenA                     #Calculate the mean value of the type A1 opinions.
                            f=x[i]+lambda_1*(lenA/len(near)*(c-x[i]))  #Converge toward neighboring opinions that are within the confidence threshold.
                            if 0<=f<=1:                   #Apply constraints to control opinions in the range of 0 to 1.
                                y[i]=round(f,4)
                            elif f<0:
                                y[i]=0
                            else:
                                y[i]=1

                for num in range(0,totalNum):  #At each time step, each agent in the network simultaneously updates its view once.
                    if x[num] !=  y[num]:      #Both type A1 and type A2 are empty sets and the opinions are consistent with the previous time step.
                        x[num] = y[num]                
            
            evo.to_csv(f"{parentPath}/randomorignal/ep-{round(epsilon1,3)}-{round(epsilon2,3)}.csv",index=None) #Store files

            nowIndex = nowIndex + 1
            
            row = row + 1


if __name__ == "__main__":
    parentPath = str(Path(__file__).parent)
    pathName = parentPath + '/randomorignal'
    if Path(pathName).is_dir():
        shutil.rmtree(pathName)
    os.mkdir(pathName)                    #Storage path for the evolution of opinions.
    
    """The initial Opinion"""
    testinitialopinionPath = f"{parentPath}/initialopinion.csv"  #The initial opinion value set by the file testinitialopinion is read.
    zz = pd.read_csv(testinitialopinionPath,header=None)
    firstName = zz.columns[0]
    zzList = zz[firstName].tolist()
    totalNum = len(zzList)
    z =  np.zeros(totalNum,float)
    for i in range(0,totalNum):
        z[i] = zzList[i]

    """The initial network structure"""
    a = np.zeros((totalNum,totalNum)) 
    aPath = f"{parentPath}/a.xlsx"             #The initial network structure set by the file testa is read.
    sheet = pd.read_excel(aPath)
    totalRow = sheet.shape[0]
    num = 0
    while num < totalRow:
        row = sheet.loc[num]
        n = row["From"]
        m = row["To"] 
        result = row["Weight"]
        a[int(n)][int(m)] = result
        num+=1

    myThread(totalNum,parentPath,a,z)

