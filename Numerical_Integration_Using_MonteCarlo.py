# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 23:35:55 2021

@author: Shahrukh
"""

import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(23)
f=lambda x:x**2
a=0
b=3
NumSteps=10000
xintegral=[]
yintegral=[]
xrectangle=[]
yrectangle=[]
#DetectingMinMAx
ymin=f(a)
ymax=ymin
for i in range(NumSteps):
    x=a+(b-a)*i/NumSteps
    y=f(x)    
    if y<ymin: ymin=y
    if y>ymax: ymax=y
#Montecarlo method
A=(b-a)*(ymax-ymin)
N=100000
M=0
for i in range(N):
    x=a+(b-a)*random.random()
    y=ymin+(ymax-ymin)*random.random()       
    if y<=f(x):
        M=M+1
        xintegral.append(x)
        yintegral.append(y)    
    else :
        xrectangle.append(x)
        yrectangle.append(y)        
NumericalIntegration=M/N *A
print('Numerical Integration= ' + str(NumericalIntegration))  
 #Visualizing the results
xlin=np.linspace(a,b)
ylin=[]
for i in xlin:
    ylin.append(f(i))    
plt.axis([0,b,0,f(b)])
plt.plot(xlin,ylin,color='red',linewidth=4)    
plt.scatter(xintegral,yintegral,color='blue',marker='.')
plt.scatter(xrectangle,yrectangle,color='yellow',marker='*')
plt.title('Numerical Method using Monte Carlo Method')
    
 


    
    



















