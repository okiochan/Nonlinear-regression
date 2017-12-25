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

Эту часть ![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/h3.gif) вычислим в математике

![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/h5.png)


