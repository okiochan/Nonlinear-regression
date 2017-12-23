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
    return d*np.sin(a*x+c)+b

def dF(p,x,y):
    a,b,c,d = p
    u = d*np.sin(a*x+c)+b-y
    Fa = u*d*np.cos(a*x+c)*x
    Fb = u
    Fc = u*d*np.cos(a*x+c)
    Fd = u*np.sin(a*x+c)
    return np.array([Fa,Fb,Fc,Fd])

def Hess(X,Y,p):
    l = X.shape[0]
    a, b, c, d = p
    result = np.zeros((4,4))
    for i in range(l):
        u = d*np.sin(a*X[i]+c)+b
        g = dF(p,X[i],Y[i])
        
        h = np.zeros((4,4))
        h[0,1]=h[1,0]=np.cos(c + a*X[i])*X[i]
        h[0,2]=h[2,0]=np.cos(c + a*X[i])
        h[1,1]= -d*X[i]*X[i]*np.sin(c + a*X[i])
        h[1,2]=h[2,1]= -d*X[i]*np.sin(c + a*X[i])
        h[2,2]=-d*np.sin(c + a*X[i])
        for j in range(4):
            for k in range(4):
                h[j,k]=g[j]*g[k] + (u - Y[i])*h[j,k]
        result += h
    return result
    
def Quadratic(p,x,y):
    a,b,c,d = p
    return 0.5*(d*np.sin(a*x+c)+b-y)**2

def Error(p,X,Y):
    n = X.shape[0]
    err = 0
    for i in range(n):
        err += Quadratic(p,X[i],Y[i])
    return err

# #(F(x+h) - F(x))/h
# def dfNumeric(p,x,y):
    # G = np.zeros(4)
    # EPS = 1e-5
    # for i in range(4):
        # xp = p.copy()
        # xp[i]+=EPS
        # G[i] = (Quadratic(xp,x,y)- Quadratic(p,x,y))/EPS
    # return G

def GradDescent(X,Y, Max,etta):
    n = X.shape[0]
    # x = np.random.randn(4)
    x = np.array([1.,1.,0,0], dtype=float)
    for step in range(Max):
        G = np.zeros(4)
        for i in range(n):
            G += dF(x, X[i],Y[i])
        H = Hess(X,Y,x)
        x -= np.linalg.inv(H).dot(G)
        Q = Error(x,X,Y)
        print(Q)
    return x



x, y = GetData(0.3)

Y1 = np.zeros(y.size)
Xh = GradDescent(x,y,200,0.002)

for i in range(y.size):
    Y1[i] = F(Xh,x[i])

plt.scatter(x,y)
plt.plot(x, Y1, label='y pred', color = "orange")
plt.show()

