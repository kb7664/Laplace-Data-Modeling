# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:23:40 2015a

@author: kboyd
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#---------------------------------------------------------------------
def vpot(a,b,y,x,V): #Note: x & y definitions are reversed relative to write-up.
    phi=0.
    for n in range(0,10): #You may need to play with number of iterations.
        k=2.*np.float(n)+1.
        A=np.sinh(k*np.pi*b/a)
        B=np.sinh(k*np.pi*y/a)
        C=np.sin(k*np.pi*x/a)
        phi=phi+B*C/k/A   
        #print(n,phi,A,B,y,b)
    return phi*V*4./np.pi
#---------------------------------------------------------------------
Vset=20. #This is the amplitude of the potential at x=20. At x=0,x=27,y=0, V=0 
V=np.loadtxt('./Laplace FINAL Lab NoBound.txt')

y=np.linspace(0,20,19)
x=np.linspace(0,28,27)
Y,X=np.meshgrid(y,x,sparse=False, indexing='ij')
fig=plt.figure(1)
#ax=fig.add_subplot(111, projection='3d')
#ax = Axes3D(plt.gcf())
#ax.plot_wireframe(X,Y,V)#X,Y,& Z must all be arrays with same dimensions for all 3d visualization

#Load in model
Vmod=np.loadtxt('Laplace Data Model NewBound.txt')
ymod=np.linspace(0,20,19)
xmod=np.linspace(0,28,27)
Ymod,Xmod=np.meshgrid(ymod,xmod,sparse=False, indexing='ij')
fig=plt.figure(1)
plt.show()
#ax=fig.add_subplot(111, projection='3d')

ax = Axes3D(plt.gcf())
ax.scatter3D(Ymod,Xmod,Vset-Vmod,color='green')
ax.hold(True)
ax.scatter3D(Y,X,Vset-V,color='red')
#next, theoretical model
b=20.
a=28.

Van=np.zeros((len(ymod),len(xmod)))+Vset #start off with all values at Vset
for i in range(0,len(ymod)-1):
    for j in range(0,len(xmod)):
        Van[i,j]=vpot(a,b,ymod[i],xmod[j],Vset)
  
#ax.plot_wireframe(Ymod,Xmod,Van,color='blue')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Potential (V)')

#--------------------------------------------------------------------
#THE HISTOGRAM 
#We start by declaring an array representing the difference
ind=0 
diff=np.zeros(425)
for i in range(1,18):
    for j in range(1,26):
            diff[ind]=V[i,j]-Vmod[i,j]
            ind=ind+1

#Construct the histogram
plt.figure(2)
plt.hist(diff , bins=20)
plt.title("Experimental Difference Histogram")
plt.xlabel("Difference Value (V)")
plt.ylabel("Frequency")
plt.show()       
#Mean and Median
print('Mean:') 
print np.mean(diff)
print('Median:') 
print np.median(diff)
#--------------------------------------------------------------------
# to figure out chi^2 and reduced chi^2: Need to bring up text files 

D = np.array(np.loadtxt("Laplace Data Model.txt"))
M = np.array(np.loadtxt("Laplace.txt"))
Chi_squared = np.sum((D-M)**2)
print(Chi_squared)
Reduced_Chi_squared = Chi_squared/425
print(Reduced_Chi_squared)
"""
D = np.array(np.loadtxt("Laplace Data Model NewBound.txt"))
M = np.array(np.loadtxt("Laplace FINAL Lab NoBound.txt"))
Chi_squared = np.sum((D-M)**2)
print(Chi_squared)
Reduced_Chi_squared = Chi_squared/425
print(Reduced_Chi_squared)

D = np.array(np.loadtxt("Laplace Data Model Origninal NewBound.txt"))
M = np.array(np.loadtxt("Laplace NoBound.txt"))
Chi_squared = np.sum((D-M)**2)
print(Chi_squared)
Reduced_Chi_squared = Chi_squared/425
print(Reduced_Chi_squared)
"""