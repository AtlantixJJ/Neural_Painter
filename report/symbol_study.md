## Normal Convex Process

### Formalization

Parameter $\theta(t)$ is a vector which evolve with time. Input $x$ is a vector.

Network output is $f(x; \theta) = \theta ^T x$.

Network loss is $l(x; \theta) = \frac{1}{2}(f(x; \theta) - y)^2$.

The evolution function is $\frac{d\theta}{dt} = -\lambda * \nabla_\theta l(x; \theta)$.

As the elements of vector are independent, we can define elementwise division to move the derivative to divisor part:

$\frac{d\theta}{\nabla_\theta l(x; \theta)} = -\lambda * dt$. On left side, the divisor part is a function of $\theta$.

### Solution

$\frac{d\theta}{\nabla_\theta l(x; \theta)} \Rightarrow \frac{d\theta}{(f(x) - y)\nabla_\theta f(x; \theta)} \Rightarrow \frac{d\theta}{(\theta^T x - y) x ^ T}$

The equation is:

$\frac{d\theta}{(\theta^T x - y)  x ^ T} = -\lambda * dt \Rightarrow \frac{1}{x^t} \int \frac{d\theta}{\theta^T x - y} = - \lambda * t + C$

As $\frac{dln (\theta^T x - y)}{d\theta} = \frac{x^T}{\theta^T x - y}$, the equation can be solved:

$ln (\theta^T x - y) = (x^T)^2 (-\lambda * t + C) \Rightarrow f(x; \theta) - y = e^{C(x^T)^2}e^{-(x^T)^2\lambda t}$.

Suppose initial condition is $\theta(0)=\theta_0$, $ln (\theta_0^T x - y) = (x^T)^2 C$.

In conclusion, $f(x; \theta) = y + (\theta_0^T x - y) e ^ {-(x^T)^2 \lambda t}$.

### Discussion



1. Exponetial training process is obtained. This is also observed in common training process.

2. 

This theory's limitations are:

1. Single layer, linear model, not applicable to deep layer, non-linear model.

2. Convolution is not studied.

3. Batch is not considered.

4. Learning rate should be small to simulate differential process.

## Adversarial Convex Process

There only discriminator is defined: $f(x; \theta) = \theta^T x$. And true data and false data are denoted as $r$ and $k$. The loss function is defined as $l(\theta, k; r) = -f(r) + f(k)$, because we want real data to be classified as positive and fake data to be classified as negative.

As defined in comman GAN game, the training process is $min_\theta max_k l(\theta, k; r)$.

To be more concrete, we have to following evolution equation:

$\frac{d\theta}{dt} = - \lambda \nabla_\theta l (\theta, k) = \lambda \nabla_\theta(f(r) - f(k)) = \lambda (r^T - k^T)$.......Eq.1

$\frac{dk}{dt} = \mu \nabla_k l (\theta, k) = \mu \nabla_k f(k) = \mu \theta^T$......Eq.2

Taking another derivative w.r.t. t: $\Rightarrow \frac{d^2 k}{dt^2} = \mu (\frac{d \theta}{dt})^T$, and substitude back:

$\frac{d^2 k}{dt^2} = \mu \lambda (r - k) \Rightarrow \frac{d^2 k}{dt^2} + \lambda\mu k - \lambda\mu r=0$.

The above is a two-order non-linear ordinary differential equation. Solve using sympy:

$k(t) = C_1*e^{-t*\sqrt{-\lambda\mu}} + C_2*e^{t*\sqrt{-\lambda\mu}} + r = C_1e^{-\sqrt{\lambda\mu} t i} + C_2e^{\sqrt{\lambda\mu} t i} + r = C_3 cos(\sqrt{\lambda\mu}t) + r$

Plugin initial condition $k(0) = k_0 \Rightarrow C_3 = k_0 - r$, so $k(t) = r + (k_0 - r)cos(\sqrt{\lambda\mu}t)$

As $\frac{1}{\mu} \frac{dk}{dt} = \theta^T$, we have $\theta(t) = \frac{r}{\mu} + (k_0-r)\sqrt{\frac{\lambda}{\mu}}sin(\sqrt{\lambda\mu}t)$, and the corresponding result:

$l(\theta) = \theta(t)^T (k(t) - r) = [\frac{r}{\mu} + (k_0-r)^T\sqrt{\frac{\lambda}{\mu}}sin(\sqrt{\lambda\mu}t)] [(k_0 - r)cos(\sqrt{\lambda\mu}t)] = \frac{r}{\mu}(k_0 - r)cos(\sqrt{\lambda\mu}t) + \frac{1}{2} \sqrt{\frac{\lambda}{\mu}} ||k_0 - r||^2 sin(2\sqrt{\lambda\mu}t)$