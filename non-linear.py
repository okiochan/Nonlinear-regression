import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def LSRegression(X,y):
    X = np.atleast_2d(X).T
    l = X.shape[0]
    n = X.shape[1]

    #concatenate ones in front of matrix
    ones = np.atleast_2d(np.ones(l)).T
    X = np.concatenate((ones,X),axis=1)

    # learn
    res = np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(y))
    return res[1:(n+1)], res[0]

def Cost(X,y,p):
    l = X.shape[0]
    a, b, c, d = p
    result = 0
    for i in range(l):
        result += (a * np.sin(b * X[i] + c) + d - y[i]) ** 2
    return result

def Grad(X,y,p):
    l = X.shape[0]
    a, b, c, d = p
    result = np.zeros(4)
    for i in range(l):
        f = a * np.sin(b * X[i] + c) + d - y[i]
        g = np.array([ np.sin(b*X[i]+c),a*np.cos(b*X[i]+c)*X[i],a*np.cos(b*X[i]+c),1 ], dtype=float)
        result += f * g
    return result

def Hess(X,y,p):
    l = X.shape[0]
    a, b, c, d = p
    result = np.zeros((4,4))
    for i in range(l):
        f = a * np.sin(b * X[i] + c) + d - y[i]
        g = np.array([ np.sin(b*X[i]+c),a*np.cos(b*X[i]+c)*X[i],a*np.cos(b*X[i]+c),1 ], dtype=float)
        h = np.zeros((4,4))
        h[0,1]=h[1,0]=np.cos(b*X[i]+c)*X[i]
        h[0,2]=h[2,0]=np.cos(b*X[i]+c)
        h[1,1]=-a*np.sin(b*X[i]+c)*X[i]**2
        h[1,2]=h[2,1]=-a*np.sin(b*X[i]+c)*X[i]
        h[2,2]=-a*np.sin(b*X[i]+c)
        for j in range(4):
            for k in range(4):
                h[j,k]=f*h[j,k]+g[j]*g[k]
        result += h
    return result

def GetData(dev = 0.1):
    np.random.seed(271828182)
    xl = np.linspace(-3.14, 3.14, 100)
    l = xl.size
    y = np.sin(xl) + np.random.randn(l) * dev
    return xl, y

np.set_printoptions(formatter={'float':lambda x: '%.4f' % x})

xl, y = GetData(0.2)
l = xl.shape[0]

p = [1,1,0,0]
for iter in range(20):
    step = -np.linalg.inv(Hess(xl,y,p)).dot(Grad(xl,y,p))
    print(p)
    p += step
print(p)

w = LSRegression(xl,y)
print(w)

y1 = np.zeros(l)
y2 = np.zeros(l)
for i in range(l):
    a, b, c, d = p
    y1[i] = a * np.sin(b * xl[i] + c) + d
    y2[i] = w[0] * xl[i] + w[1]

plt.plot(xl, y2, label='y pred', color = "violet")
plt.plot(xl, y1, label='y pred', color = "orange")
plt.scatter(xl,y)
#plt.scatter(xl,y1)
#plt.scatter(xl,y2)
plt.show()
