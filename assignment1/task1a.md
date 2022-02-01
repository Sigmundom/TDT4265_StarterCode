## Task 1b

$$ 1: \frac{\delta C^n(w)}{\delta w_{kj}}
= \frac{\delta}{\delta w_{kj}} \left(-\sum_{k=1}^K y_kln(\hat{y}_k)\right) =
 -\sum_{k=1}^K y_k \frac{\delta}{\delta w_{kj}}\left(ln(\hat{y}_k)\right)$$

 $$ 2: \frac{\delta}{\delta w_{kj}} ln(\hat{y}_k) = \frac{\delta}{\delta w_{kj}} ln\left(\frac{e^{z_k}}{\sum_{k'}^K e^{z_{k'}}}\right) = \frac{\delta}{\delta w_{kj}} \left(ln(e^{z_k})-ln(\sum_{k'}^K e^{z_{k'}})\right) $$
$$ 3: \frac{\delta}{\delta w_{kj}} ln(e^{z_k})
= \frac{1}{e^{z_k}} \frac{\delta}{\delta w_{kj}} e^{z_k}
=  \frac{1}{e^{z_k}} e^{z_k} \frac{\delta}{\delta w_{kj}} z_k $$
$$4: \frac{\delta}{\delta w_{kj}} ln(\sum_{k'}^K e^{z_{k'}})
= \frac{1}{\sum_{k'}^K e^{z_{k'}}} \frac{\delta}{\delta w_{kj}}\sum_{k'}^K e^{z_{k'}}
= \frac{1}{\sum_{k'}^K e^{z_{k'}}} \sum_{k'}^K e^{z_{k'}}\frac{\delta}{\delta w_{kj}}z_{k'}
= \frac{1}{\sum_{k'}^K e^{z_{k'}}} \sum_{k'}^K e^{z_{k'}}\frac{\delta}{\delta w_{kj}}z_{k'}
$$
$$5:
\frac{\delta}{\delta w_{kj}} z_{k'}
= \frac{\delta}{\delta w_{kj}} \sum_i^I w_{ki} x_i
= \begin{cases}
0 \; for\; i \neq j \; or \; k\neq k' \\
x_i \; for \; i =j \; and \; k = k'
\end{cases} = x_i $$

Substituting 5 in 3 and 4 gives $x_i$ and $\frac{e^{z_k}}{\sum_{k'}^K e^{z_{k'}}}x_i = \hat{y}_kx_i$ respectivley.
Substituting 3 and 4 in 2 gives








## Task 1a

$$ \frac{\delta C^{n}(w)}{\delta w_i} = - \frac{\delta (y^nln(\hat{y}^n) + (1-y^n) ln(1 - \hat{y}^n) )}{\delta w_i} $$
To simplify notation $\hat{y}^n = f(x)$ and all subscripts are omitted

$$= -y\frac{\delta}{\delta w} ln(f(x)) + (1-y) \frac{\delta} {\delta w} ln(1-f(x))$$
$$ = - \frac{y f'(x)}{f(x)} - \frac{(y-1)f'(x)}{1-f(x)} $$
$$ = \frac{y}{f(x)} x f(x) (1-f(x)) - \frac{y-1}{1-f(x)} x f(x) (1-f(x)) $$
$$ = -yx + yxf(x) - yxf(x) + xf(x) $$
$$ = -(y-f(x))x $$
$$ = -(y^n - \hat{y}^n)x_i^n $$