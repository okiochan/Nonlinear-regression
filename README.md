# Nonlinear-regression

Для выборки вида:

![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/1.png)

**линейная регрессия не подходит**

Модель для выборки подбирает эксперт. Данная выборка похожа на синус

Запишем МНК
![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/CodeCogsEqn.gif)

Можно мин-ть градиентным спуском. Но есть град. спуск с гессианом  ![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/CodeCogsEqn(1).gif)
(сходится с квадратичной скоростью, но функция должна быть выпукла)

Запустим алгоритм на нашей выборке;
**оранжевый - результат нелинейной регрессии**

![](https://raw.githubusercontent.com/okiochan/Nonlinear-regression/master/2.png)

код программы [здесь]( https://github.com/okiochan/Nonlinear-regression/blob/master/non_linear_stanrard.py)
