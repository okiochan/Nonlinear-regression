# Nonlinear-regression

Для выборки вида:

![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/img/1.png)

**линейная регрессия не подходит**

Модель для выборки подбирает эксперт. Данная выборка похожа на синус

Запишем МНК
![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/formula/mnk.gif)

Запустим алгоритм на нашей выборке; Минимизировала градиентным спуском.

![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/img/2.png)

код программы **non_linear_stanrard.py**


Можно мин-ть градиентным спуском. Но есть метод Ньютона, алгоритм Ньютона (также известный как метод касательных) — это итерационный численный метод нахождения корня (нуля) заданной функции. 
Метод описан ![здесь]( https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%9D%D1%8C%D1%8E%D1%82%D0%BE%D0%BD%D0%B0)
Применим его для поиска экстремума нашей функции. 

Пусть необходимо найти минимум функции многих переменных
![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/formula/n1.gif)
Эта задача равносильна задаче нахождения нуля градиента {\displaystyle \nabla f({\vec {x}})}

Но есть град. спуск с гессианом  ![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/CodeCogsEqn(1).gif)
(сходится с квадратичной скоростью, но функция должна быть выпукла)

Для нахождения матрицы Гессе, запишем нашу функцию так:
![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/formula/h1.gif)

Посчитаем матрицу Гессе вторых производных сначала в общем виде:
по первой производной
![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/formula/h2.gif)
по второй производной
![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/formula/h3.gif)

Эту часть ![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/formula/h4.gif) вычислим в математике

![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/formula/h5.png)

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
