# Nonlinear-regression

Для выборки вида:

![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/img/1.png)

**линейная регрессия не подходит**

Модель для выборки подбирает эксперт. Данная выборка похожа на синус

Запишем МНК
![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/formula/mnk.gif)

Запустим алгоритм на нашей выборке; Минимизируем градиентным спуском.

![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/img/2.png)

код программы **non_linear_stanrard.py**


Можно мин-ть градиентным спуском. Но есть **метод Ньютона**, алгоритм Ньютона (также известный как метод касательных) — это итерационный численный метод нахождения корня (нуля) заданной функции. 
Подробно метод описан [здесь]( https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%9D%D1%8C%D1%8E%D1%82%D0%BE%D0%BD%D0%B0)
Применим его для поиска экстремума нашей функции. 

Пусть необходимо найти минимум функции многих переменных
![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/formula/n1.gif)
Эта задача равносильна задаче нахождения нуля градиента 
![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/formula/n2.gif)

Применим метод Ньютона, и запишем в удобном итеративном виде (Н - гессиан, матрица Гёссе):
![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/formula/n3.gif)

Следует отметить, что в случае квадратичной функции метод Ньютона находит экстремум за одну итерацию.
Сходится с квадратичной скоростью, но функция должна быть выпукла.

**Метод Ньютона — Рафсона** является улучшением метода Ньютона нахождения экстремума, описанного выше. Основное отличие заключается в том, что на очередной итерации каким-либо из методов одномерной оптимизации выбирается оптимальный шаг:

![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/formula/n4.gif)

![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/formula/n5.gif)

# Реализуем метод Ньютона-Рафсона

Для нахождения **матрицы Гёссе**, запишем нашу функцию так:
![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/formula/h1.gif)

Посчитаем матрицу Гессе вторых производных сначала в общем виде:
по первой производной
![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/formula/h2.gif)
по второй производной
![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/formula/h3.gif)

Эту часть ![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/formula/h4.gif) вычислим в математике

![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/formula/h5.png)

Реализуем Hessian(полученные формулы) на python

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

Теперь сделаем нужный спуск, не забудем про лямбду (длину вектора)

```
    for i in range(n):
        G += dF(x, X[i],Y[i])
        H += Hess(x, X[i],Y[i])
    #направление вектора
    d = -np.linalg.pinv(H).dot(G)
    
    def Qalpha(alpha):
        return Error(x + alpha * d, X, Y)
    #ищем длину вектора
    alpha_best = Golden(Qalpha, 0, 10)
    x = x + alpha_best * d
```

Лямбду найдем методом **золотого сечения**, в программме - Golden, принимает функцию (функционал качества), границы, эпсилон и кол-во итераций

```
Golden(f, l, r, EPS=1e-9, maxiter=100)
```

код программы в **non_linear_hessian.py**

Вот как отработают оба алгоритма: **Градиентный спуск с шагом 0.002** и **Метод Ньютона — Рафсона**. Мы видим, что оба **сошлись к одному результату**. По скорости, Метод Ньютона — Рафсона гораздо быстрее.

![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/img/n1.png)

Сравним ошибки. Мы видимЮ что у Ньютона — Рафсона SSE меньше и он быстрее сходится.
![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/img/n11.png)


С применением метода Ньютона — Рафсона, функция сходится в разы быстрее, при этом квадратичная ошибка не ухудшается.

Можно также заметить, что **стандартный метод Ньютона** (без лямбды) не всегда сходится. На данном примере, при начальном x0 = (1,1,1,1)  метод не сошелся

![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/img/n2.png)

SSE
![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/img/n22.png)
