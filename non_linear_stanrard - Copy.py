import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
 
def GetData(dev = 0.1):
    np.random.seed(271828182)
    xl = np.linspace(-3.14, 3.14, 100)
    l = xl.size
    y = np.sin(xl) + np.random.randn(l) * dev
    return xl, y
 
def F(p,x):
    a,b,c,d = p
    return a*np.sin(b*x+c)+d
 
def dU(p,x,y):
    a, b, c, d = p
    Fa = np.sin(c+b*x)
    Fb = a*x*np.cos(c+b*x)
    Fc = a*np.cos(c+b*x)
    Fd = 1
    return np.array([Fa,Fb,Fc,Fd])
 
def dF(p,x,y):
    return (F(p,x)-y)*dU(p,x,y)
 
def Hess(p,x,y):
    a, b, c, d = p
    g = dU(p,x,y)
    u = F(p,x)
 
    h = np.zeros((4,4))
    h[0,1]=h[1,0]=x*np.cos(c+b*x)
    h[0,2]=h[2,0]=np.cos(c+b*x)
    h[1,1]= -a*x*x*np.sin(c+b*x)
    h[1,2]=h[2,1]= -a*x*np.sin(c+b*x)
    h[2,2]=-a*np.sin(c+b*x)
    for i in range(4):
        for j in range(4):
            h[i,j]=g[i]*g[j] + (u - y)*h[i,j]
    return h
 
def Quadratic(p,x,y):
    a,b,c,d = p
    return 0.5*(F(p,x)-y)**2
 
def Error(p,X,Y):
    n = X.shape[0]
    err = 0
    for i in range(n):
        err += Quadratic(p,X[i],Y[i])
    return err
 
#(F(x+h) - F(x))/h
def dfNumeric(p,x,y):
    G = np.zeros(4)
    EPS = 1e-5
    for i in range(4):
        xp = p.copy()
        xp[i]+=EPS
        G[i] = (Quadratic(xp,x,y)- Quadratic(p,x,y))/EPS
    return G
 
def HsNumeric(p,x,y):
    H = np.zeros((4,4))
    EPS = 1e-5
    for i in range(4):
        xp = p.copy()
        xp[i]+=EPS
        H[i,:] = (dfNumeric(xp,x,y)- dfNumeric(p,x,y))/EPS
    return H
 
def GradDescent(X,Y, Max,etta):
    n = X.shape[0]
    x = np.array([1.01,1.01,-0,-0], dtype=float)
    # x = np.random.randn(4)*0.1
    # x += np.random.randn(4)*0.1
    print(x)
    for step in range(Max):
        G = np.zeros(4)
        H = np.zeros((4,4))
        for i in range(n):
            G += dF(x, X[i],Y[i])
            H += Hess(x, X[i],Y[i])
        x -= np.linalg.pinv(H).dot(G)
        Q = Error(x,X,Y)
        print(Q)
    return x
 
 
 
x, y = GetData(0.3)
 
# p = np.array([1.,1.,0,0], dtype=float)
# print(dF(p,x[2],y[2]))
# print()
# print(dfNumeric(p,x[2],y[2]))
# print()
# print(dF(p,x[2],y[2]) - dfNumeric(p,x[2],y[2]))
# quit()
 
# p = np.array([1.,1.,0,0], dtype=float)
# print(HsNumeric(p,x[2],y[2]))
# print()
# print(Hess(p,x[2],y[2]))
# print()
# print(Hess(p,x[2],y[2]) - HsNumeric(p,x[2],y[2]))
# quit()
 
Xh = GradDescent(x,y,100,0.002)
 
Y1 = np.zeros(y.size)
for i in range(y.size):
    Y1[i] = F(Xh,x[i])
 
plt.scatter(x,y)
plt.plot(x, Y1, label='y pred', color = "orange")
plt.show()