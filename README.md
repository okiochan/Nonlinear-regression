# Nonlinear-regression

Для выборки вида:

![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/1.png)

**линейная регрессия не подходит**

Модель для выборки подбирает эксперт. Данная выборка похожа на синус

Запишем МНК
![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/mnk.gif)

Запустим алгоритм на нашей выборке; Минимизировала градиентным спуском.

![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/2.png)


код программы [здесь]( https://github.com/okiochan/Nonlinear-regression/blob/master/non_linear_stanrard.py)

Можно мин-ть градиентным спуском. Но есть град. спуск с гессианом  ![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/CodeCogsEqn(1).gif)
(сходится с квадратичной скоростью, но функция должна быть выпукла)

Для нахождения матрицы Гессе, запишем нашу функцию так:
![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/h1.gif)

Посчитаем матрицу Гессе вторых производных сначала в общем виде:
по первой производной
![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/h2.gif)
по второй производной
![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/h3.gif)

Эту часть ![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/h4.gif) вычислим в математике

![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/h5.png)

Реализуем Hessian на python

```
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
    
```

код программы [здесь]( https://github.com/okiochan/Nonlinear-regression/blob/master/non_linear_hessian.py)
