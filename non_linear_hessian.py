import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import non_linear_stanrard as NLS
import matplotlib.patches as mpatches

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
    err = Quadratic(p,X,Y).sum()
    return err / X.shape[0]

#(F(x+h) - F(x))/h
def dfNumeric(p,x,y):
    G = np.zeros(4)
    EPS = 1e-5
    for i in range(4):
        xp = p.copy()
        xp[i]+=EPS
        G[i] = (Quadratic(xp,x,y)- Quadratic(p,x,y))/EPS
    return G

#grad chislenno
def HsNumeric(p,x,y):
    H = np.zeros((4,4))
    EPS = 1e-5
    for i in range(4):
        xp = p.copy()
        xp[i]+=EPS
        H[i,:] = (dfNumeric(xp,x,y)- dfNumeric(p,x,y))/EPS
    return H

#zolotoe sechenie
def Golden(f, l, r, EPS=1e-9, maxiter=100):
    d = r-l
    phi = (math.sqrt(5) - 1) / 2
    f1 = f(l + d * phi * phi)
    f2 = f(l + d * phi)
    i = 0
    while d>EPS and i<maxiter:
        i += 1
        d *= phi
        if f2<f1:
            l += d * phi
            f1 = f2
            f2 = f(l + d * phi)
        else:
            f2 = f1
            f1 = f(l + d * phi * phi)
    return l + d/2

def GradDescentWithHessian(X,Y, Max,etta):
    n = X.shape[0]
    #nachalnoe priblizenie
    #x = np.array([1.01,1.01,-0.,-1], dtype=float)
    x = np.ones(4)
    print(x)
    for step in range(Max):
        G = np.zeros(4)
        H = np.zeros((4,4))
        for i in range(n):
            G += dF(x, X[i],Y[i])
            H += Hess(x, X[i],Y[i])
        #pinv - psevdo obratnaya matrix, needs if exist null eigen values
        #direction
        d = -np.linalg.pinv(H).dot(G)
        
        def Qalpha(alpha):
            return Error(x + alpha * d, X, Y)
        #we get optimal lendth (step size)
        alpha_best = Golden(Qalpha, 0, 10)
        x = x + alpha_best * d
        
        Q = Error(x,X,Y)
        print(Q)
    return x
    
#------------------------------------main-------------------------------------------
x, y = GetData(0.3)
plt.scatter(x,y)

Xh = GradDescentWithHessian(x,y,100,0.002)
Y1 = np.zeros(y.size)
for i in range(y.size):
    Y1[i] = F(Xh,x[i])
plt.plot(x, Y1, label='y pred', color = "orange")

Xh2 = NLS.GradDescent(x,y,2000,0.002)
Y2 = np.zeros(y.size)
for i in range(y.size):
    Y2[i] = NLS.F(Xh2,x[i])
#plt.plot(x, Y2, label='y pred', color = "violet")

print("\n\nSSE for Hessian")
print( Error(Xh,x,y))
print("SSE for GD")
print( Error(Xh2,x,y))

plt.legend(handles=[mpatches.Patch(color='orange', label='standard Newton'),mpatches.Patch(color='violet', label='standard GD')])
plt.show()