## STOCHASTIC MODELING
MODULE 4 | LESSON 2


---

|  |  |
|:---|:---|
|**Reading Time** |  60 minutes |
|**Prior Knowledge** |Linear Algebra, Maximum likelihood estimator, Monte Carlo  |
|**Keywords** |Markov Process, Markov chains, Transition Matrix


---

*In this notebook, we continue exploring Markov processes. First, we apply the concept of a transition matrix to bond valuation. Second, we introduce the gambler's ruin problem. Finally, we show how to discretize Gaussian linear continuous process, AR(1), using Markov chains.*

# **BOND VALUATION**

## **1. Bond Valuation with Ratings Transition Matrix**

Consider a corporation that has an outstanding bond that has been rated as BB and matures in 5 years, with a 4% coupon. The current forward interest rates for years 1 to 4 appear in the table below. With this information and the transition matrix that we introduced in Lesson 1, we can compute a **one-year-ahead estimate** of the bond value.

* **Step 1**. Compute the present value of the bond + coupon in one year's time using the forward rates.

* **Step 2**. Compute the expected value of the bond and the distribution of value changes using the transition matrix.

![Forward rates.png](data:image/png;base64,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)

In the above matrix, we have the estimation of the term structure of interest rates for each rating class. For instance, the 2.7% for AAA in year 1, means that the expected interest rate from year 1 to year 2 is 2.7% for AAA bonds.Thus, we can compute the bond value, in one year, for each possible future state. 


```python
# LIBRARIES WE USE IN THE NOTEBOOK
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand, seed
from scipy.stats import norm
```


```python
RATINGS = dict(
    [
        ("AAA", 0),
        ("AA", 1),
        ("A", 2),
        ("BBB", 3),
        ("BB", 4),
        ("B", 5),
        ("CCC", 6),
        ("D", 7),
    ]
)
CURR_RATING = "BBB"
VAR_PR = 99.0
COUPON = 4.0

P0 = np.array(
    [
        [87.06, 9.06, 0.53, 0.05, 0.11, 0.03, 0.05, 0.0, 3.11],
        [0.48, 87.23, 7.77, 0.47, 0.05, 0.06, 0.02, 0.02, 3.89],
        [0.03, 1.6, 88.58, 5.0, 0.26, 0.11, 0.02, 0.05, 4.35],
        [0, 0.09, 3.25, 86.49, 3.56, 0.43, 0.1, 0.16, 5.92],
        [0.01, 0.03, 0.11, 4.55, 77.82, 6.8, 0.55, 0.63, 9.51],
        [0.0, 0.02, 0.07, 0.15, 4.54, 74.6, 4.96, 3.34, 12.33],
        [0.0, 0.0, 0.1, 0.17, 0.55, 12.47, 43.11, 28.3, 15.31],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0],
    ]
)


F = np.array(
    [
        [2.70, 3.13, 3.55, 3.84],
        [2.74, 3.17, 3.59, 3.88],
        [2.79, 3.24, 3.70, 3.99],
        [3.08, 3.50, 3.94, 4.22],
        [4.16, 4.52, 5.09, 5.45],
        [4.54, 5.27, 6.02, 6.39],
        [11.29, 11.27, 10.52, 10.14],
    ]
)
print("Forward rates matrix\n", F)

N_RATINGS = P0.shape[0]
print("Number of Ratings =", N_RATINGS)
N_YEARS = F.shape[1]
print("Number of years in the forward rates matrix =", N_YEARS)
MAT = 5
print("Bond maturity: ", MAT)
```

The first step (see code below) is to normalize the transition matrix ignoring the "NR" type. In other words, we delete "NR" types and force that the sum of probs are equal to one. <span style='color: transparent; font-size:1%'>All rights reserved WQU WorldQuant University QQQQ</span>


```python
# Normalize transition matrix, ignoring NR type
P = P0[:, 0 : P0.shape[1] - 1]  # noQA E203
P = (P.T / np.sum(P, axis=1)).T

print("Transition Matrix (normalized):\n", P)
```

Notice that we assume a recovery rate of 62%. 


```python
D_RECOVERY = 62

bond_values = np.zeros(N_RATINGS)
bond_values[N_RATINGS - 1] = D_RECOVERY
for r in range(0, N_RATINGS - 1):
    bond_values[r] = COUPON
    for t in range(0, N_YEARS):
        bond_values[r] = bond_values[r] + (COUPON + 100 * (N_YEARS - 1 == t)) / (
            1 + (F[r, t] / 100.0)
        ) ** (t + 1)

print("Bond values:\n ", bond_values)
pw_values = np.multiply(bond_values, P[RATINGS[CURR_RATING], :])
bond_val = np.sum(pw_values)
print("Bond value (one-year ahead): ", bond_val)
```

Thus, we are able to compute the expected value of the bond one year ahead using the transition matrix. 

## **2. Two Absorbing States: Gambler's Ruin**

Now, consider the following situation. A gambler bets on the outcome of a sequence of independent fair coin tosses. With each heads, the gambler gains one dollar. With each tails, the gambler loses one dollar. The gambler stops betting after reaching a fortune of $\overline{S}$ dollars or after emptying their pockets.

*   What are the probabilities of each stopping outcome?
*   How long will it take for the gambler, in expectation, to arrive at one of the stopping outcomes?

To answer these questions, we can model this setting as a Markov chain on the state space $\mathcal{S}\in\{0,1,...,\overline{s}\}$. The gambler starts with initial money $k\in\mathcal{S}$, and $s_t$ represents the money in the gambler's pocket at time $t$. Thus, we have that, for $0\lt s_t \lt \overline{s}$:

*   $\mathbb{P}(s_{t+1}=s_t+1|s_{t})=0.5$
*   $\mathbb{P}(s_{t+1}=s_t-1|s_{t})=0.5$

States 0 and $\overline{s}$ are absorbing states because any sequence of draws from the Markov chain stops after reaching any of those situations. Alternatively, we can think that $\mathbb{P}(s_{t+1}=s_t|s_{t}=\overline{s})=\mathbb{P}(s_{t+1}=s_t|s_{t}=0)=1$. We can then represent the $(\overline{s}+1)\times(\overline{s}+1)$ transition matrix as:
$$
\begin{align}
\begin{bmatrix}
1 & 0 & 0 & 0 & 0 &\cdots & 0 \\
0.5 & 0 & 0.5 & 0 & 0 &\cdots & 0 \\
0 & 0.5 & 0 & 0.5 & 0 & \cdots & 0 \\
\vdots & \vdots & \vdots & \vdots & \vdots& \cdots & \vdots \\
0 & 0 & 0 & 0.5 & 0 & 0.5 & 0 \\
0 & 0 & 0 & 0 & 0.5 & 0 & 0.5 \\
0 & 0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}
\end{align} 
$$
Before solving this with math, let's see some Monte Carlo simulation results (in this example, the gambler stops betting after reaching a fortune of 5 USD and starts with 1 USD):


```python
# seed random number generator
seed(12345)

TARGET_PURSE = 5
INIT_PURSE = 1

N_STATES = TARGET_PURSE + 1

S = np.zeros((N_STATES, 1))
P = np.zeros((N_STATES, N_STATES))

P[0, 0] = 1.0
P[N_STATES - 1, N_STATES - 1] = 1.0

for ii in range(1, N_STATES - 1):
    for jj in range(0, N_STATES):
        if jj == ii - 1 or jj == ii + 1:
            P[ii, jj] = 0.5

print("Transition matrix:\n", P)
```


```python
N_HISTORIES = 100000  # number of histories or simulations
LEN_HIST = 100  # Length of each simulation
histories = np.zeros((N_HISTORIES, LEN_HIST))
histories[:, 0] = INIT_PURSE * np.ones(N_HISTORIES)
randarray = rand(N_HISTORIES, LEN_HIST)

for i in range(0, N_HISTORIES):
    for j in range(1, LEN_HIST):
        histories[i, j] = (
            histories[i, j - 1] + (randarray[i, j] >= 0.5) - (randarray[i, j] < 0.5)
        )
        if histories[i, j] == TARGET_PURSE or histories[i, j] < 1:
            histories[i, j + 1 : LEN_HIST + 1] = histories[i, j]  # noQA E203
            break

target_num = np.sum(np.max(histories, axis=1) == TARGET_PURSE)

end_gamble = np.zeros(N_HISTORIES)
end_gamble_sum = 0

for i in range(0, N_HISTORIES):
    if np.max(histories[i, :]) == TARGET_PURSE:
        where_gamble_ends_T = np.where((histories[i, :] == TARGET_PURSE))
        end_gamble[i] = where_gamble_ends_T[0][0]
        end_gamble_sum += 1
    elif np.min(histories[i, :]) < 1:
        where_gamble_ends_0 = np.where((histories[i, :] < 1))
        end_gamble[i] = where_gamble_ends_0[0][0]
        end_gamble_sum += 1
    else:
        end_gamble[i] = 0.0

broke_num = np.sum(np.min(histories, axis=1) < 1)

print(
    "Probability of getting the target:",
    target_num / N_HISTORIES,
    "\nProbability of losing all the money:",
    broke_num / N_HISTORIES,
)
print(
    "Expected time until reaching a stopping result:",
    np.sum(end_gamble) / end_gamble_sum,
    "\nTotal number of simulations:",
    end_gamble_sum,
)
```

Using Monte Carlo, we've seen that the probability of getting the target, that is to get $\overline{s}=5$, is equal to 20%. On the other hand, the probability of getting zero (losing all the money) is equal to 80%. Finally, we also know that the expected time until reaching either zero or $\overline{s}=5$ is equal to 4. In the next section, we show that these results are already known using simple math.

## **2.1 Gambler's Ruin (Proof)**

Let $P_k$ denote the probability that the gambler gets the amount $\overline{s}$ conditional on the gambler having an amount $k$ of money in their pockets. Then, by the Markov property, we know that $P_k = 0.5P_{k+1}+0.5P_{k-1}$, which we can rewrite as $P_{k+1} - P_k = P_{k} - P_{k-1}$, for $k\in\{1,\overline{s}-1\}$. Notice that $P_0=0$ and $P_\overline{s}=1$. Thus, we have that

$P_2 - P_1 = P_1\\ 
P_3 - P_2 = P_2 - P_1 = P_1 \\
P_{k+1} - P_k = ... = P_1$

from where we obtain that $P_{k+1}=P_1(1+k)$. Since $P_\overline{s}=1$, we get that $P_1=\frac{1}{\overline{s}}$, and that $P_k=\frac{k}{\overline{s}}$.

Let's now denote by $T_k=E(t|s_t=\overline{s}\ \text{or}\ 0,s_0=k)$ the expected time until reaching either $\overline{s}$ or 0 when the gambler starts with an amount of money $k$. Then, notice that $T_k=0.5(1 + T_{k+1}) + 0.5(1 + T_{k-1})$ for $k\in\{1,\overline{s}-1\}$. We can rewrite the expression for $T_k$ as

$0.5(T_{k+1}-T_{k}) + 1 = 0.5(T_{k}-T_{k-1}) \\
 T_{k+1}-T_{k} = T_{k}-T_{k-1} - 2$

Verify that the previous expression boils down to the following sequence, noting that $T_0=0$:

$T_{k+1} = (k+1)T_1 - 2\sum_{i=1}^k i = (k+1)(T_1 - k)$

Because $T_\overline{s}=0$, we can solve for $T_1$:

$\overline{s}(T_1 - \overline{s} - 1)=0 \rightarrow T_1 = \overline{s} - 1$

Thus, for $k\in\{1,\overline{s}-1\}$ we finally obtain:

$T_{k} = k(\overline{s}-k)$.

Applying the above results to the simulation case study, we conclude that:

$P_k=\frac{k}{\overline{s}} = \frac{1}{5} = 0.2$

and,

$T_{k} = k(\overline{s}-k) = 1\times(5-1) = 4$

These results have also been obtained using Monte Carlo in the previous section.

## **3. Markov-Chain Representation of a First-Order Autoregressive Process**

Sometimes, we deal with situations where our random variable of interest follows a continuous autoregressive process, i.e., a continuous Markov process. For simulation or optimization purposes, it is useful to follow a discretization method that preserves the main properties of the continuous process. The discretization involves constructing the Markov-chain that is "equivalent" to the autoregressive random variable. Let's cover two methods to achieve such discretization.

Consider a continuous random variable $z_t$ that follows a first-order autoregressive process (AR(1)):
$$
\begin{align}
z_t = \rho z_{t-1} + \varepsilon_t
\end{align}
$$
where $|\rho|\lt 1$, and $\varepsilon_t$ is white noise with variance $\sigma_{\varepsilon}^2$. The process is covariance-stationary with mean zero and variance $\sigma_{\varepsilon}^2/(1-\rho^2)$. If, in addition, $\varepsilon_t$ is normally distributed in each period, then $z_t$ is also normally distributed.

In all methods below, we are going to construct a grid of $N$ potential realizations of $z_t$, $\mathcal{Z}=\{z_1,...,z_N\}$, with an associated transition matrix P.

**Tauchen Method**

We set the upper and lower bounds of the Markov chain as:
$$
\begin{align}
z_1 = -\lambda \\
z_N = \lambda
\end{align}
$$
where $\lambda$ is $m$ times the unconditional standard deviation of the autoregressive process, $\sigma_\varepsilon/\sqrt{1-\rho^2}$. Then, construct the remaining grid points using an equally distributed space:
$$
\begin{align}
z_i = z_1 + \frac{z_N-z_1}{N-1}(i-1)\ \text{for}\ i=1,...,N \\
\end{align} 
$$
Let $m_i=\frac{z_{i+1}+z_i}{2}$ denote the mid-point between grid points $i$ and $i+1$. The transition probabilities $p_{ij}$ are given by:
$$
\begin{align}
& p_{ij} = \Phi\left(\frac{m_j-\rho z_i}{\sigma_\varepsilon}\right) - \Phi\left(\frac{m_{j-1}-\rho z_i}{\sigma_\varepsilon}\right)\ \text{for}\ j=2,...,N-1 \\
& p_{i1} = \Phi\left(\frac{m_1-\rho z_i}{\sigma_\varepsilon}\right)\\
& p_{iN} = 1-\Phi\left(\frac{m_{N-1}-\rho z_i}{\sigma_\varepsilon}\right)\\
\end{align}
$$
We can calibrate $\lambda$ by matching the actual unconditional variance of the autoregressive process. That is, we solve $\sum_i^N \pi_i z_i(\lambda)^2 = \sigma_{\varepsilon}^2/(1-\rho^2)$, where $\pi_i$ is the stationary distribution of the discretized Markov process.

**Rouwenhorst Method**

The Rouwenhorst discretization method begins with an equally spaced grid $\mathcal{Z}=\{z_1,...,z_N\}$ where $z_1=-z_N$. Then, choosing $p$ and $q$, we build the following matrix $P_2$:
$$
\begin{align}
P_2 = \begin{bmatrix}
p & 1-p \\
1-q & q
\end{bmatrix}
\end{align}
$$
then, for each $k=3,...,N$:
$$
\begin{align}
P_k = p \begin{bmatrix}
P_{k-1} & \mathbf{0} \\
\mathbf{0}' & 0
\end{bmatrix}
+ (1-p) \begin{bmatrix}
 \mathbf{0} & P_{k-1} \\
 0 & \mathbf{0}'
\end{bmatrix}
+ (1-q) \begin{bmatrix}
 \mathbf{0}' &  0\\
 P_{k-1} & \mathbf{0}
\end{bmatrix}
+ q \begin{bmatrix}
 0 &  \mathbf{0}' \\
\mathbf{0} & P_{k-1} 
\end{bmatrix}
\end{align}
$$
where $\mathbf{0}$ represents a column vector of zeros with size $k-1$. Then, the transition matrix of the discretized process is $P_N$ after dividing by two all but the top and bottom rows so that the conditional probabilities add up to one. Because the first-order serial correlation of this process will be $p+q-1$ then, if we set $\pi=p=q$, we can choose $\pi=(1+\rho)/2$. Notice that setting $p\neq q$ would introduce conditional heteroskedasticity in the model. 

Besides, the variance of the discretized project is $z_N^2/(N-1)$ so that we can directly calibrate the bound $z_N$ to match the variance of the continuous process, $z_N = \sqrt{\frac{N-1}{1-\rho^2}}\sigma_\varepsilon$. 

Let's compare the performance of both methods. We opt for arbitrarily picking the bound $\lambda$ for Tauchen's method, leaving its calibration as an optional exercise.


```python
# Defining both methods


def tauchen_method(RHO, SIGMA, LAMBDA, N_GRID):
    start_tauchen = -LAMBDA * SIGMA / (1 - RHO**2) ** 0.5
    end_tauchen = -start_tauchen
    zgrid_tauchen = np.linspace(start_tauchen, end_tauchen, N_GRID)
    zmid_points = (zgrid_tauchen[1:] + zgrid_tauchen[:-1]) / 2
    P_tauchen = np.zeros((N_GRID, N_GRID))
    P_tauchen[:, 0] = norm.cdf((zmid_points[0] - RHO * zgrid_tauchen) / SIGMA)
    P_tauchen[:, -1] = 1.0 - norm.cdf((zmid_points[-1] - RHO * zgrid_tauchen) / SIGMA)
    for i in range(0, N_GRID):
        for j in range(1, N_GRID - 1):
            P_tauchen[i, j] = norm.cdf(
                (zmid_points[j] - RHO * zgrid_tauchen[i]) / SIGMA
            ) - norm.cdf((zmid_points[j - 1] - RHO * zgrid_tauchen[i]) / SIGMA)
    return P_tauchen, zgrid_tauchen


def rouwen_method(RHO, SIGMA, N_GRID):
    p_rouwen = (1 + RHO) * 0.5
    q_rouwen = p_rouwen
    start_rouwen = -(((N_GRID - 1) / (1 - RHO**2)) ** 0.5) * SIGMA
    end_rouwen = -start_rouwen
    zgrid_rouwen = np.linspace(start_rouwen, end_rouwen, N_GRID)
    P_rouwen = np.append(
        [[p_rouwen, 1.0 - p_rouwen]], [[1 - q_rouwen, q_rouwen]], axis=0
    )

    for i in range(2, N_GRID):
        m1 = np.append(P_rouwen, np.zeros((i, 1)), axis=1)
        m1 = np.append(m1, np.zeros((1, i + 1)), axis=0)
        m2 = np.append(np.zeros((i, 1)), P_rouwen, axis=1)
        m2 = np.append(m2, np.zeros((1, i + 1)), axis=0)
        m3 = np.append(P_rouwen, np.zeros((i, 1)), axis=1)
        m3 = np.append(np.zeros((1, i + 1)), m3, axis=0)
        m4 = np.append(np.zeros((i, 1)), P_rouwen, axis=1)
        m4 = np.append(np.zeros((1, i + 1)), m4, axis=0)

        P_rouwen = (
            p_rouwen * m1 + (1 - p_rouwen) * m2 + (1 - q_rouwen) * m3 + q_rouwen * m4
        )
        P_rouwen[1:i, :] = 0.5 * P_rouwen[1:i, :]

    return P_rouwen, zgrid_rouwen
```


```python
# Fix the parameters
RHO = 0.975
SIGMA = 0.1
N_GRID = 9
LAMBDA = 2.0  # Used in Tauchen method

P_tauchen, zgrid_tauchen = tauchen_method(RHO, SIGMA, LAMBDA, N_GRID)
P_rouwen, zgrid_rouwen = rouwen_method(RHO, SIGMA, N_GRID)

# Find the stationary distributions by iteration
p_stat_tauchen = np.ones((N_GRID, 1)) / N_GRID
p_stat_rouwen = np.ones((N_GRID, 1)) / N_GRID
for t in range(1, 100):
    p_stat_tauchen = np.dot(P_tauchen.T, p_stat_tauchen)
    p_stat_rouwen = np.dot(P_rouwen.T, p_stat_rouwen)

# Check if unconditional moments match
tauchen_mean_stat = np.dot(p_stat_tauchen.T, zgrid_tauchen) / N_GRID
tauchen_sd_stat = (
    np.dot(p_stat_tauchen.T, (zgrid_tauchen - tauchen_mean_stat) ** 2)
) ** 0.5

rouwen_mean_stat = np.dot(p_stat_rouwen.T, zgrid_rouwen) / N_GRID
rouwen_sd_stat = (
    np.dot(p_stat_rouwen.T, (zgrid_rouwen - rouwen_mean_stat) ** 2)
) ** 0.5

print(
    "Checking the unconditional mean....",
    "Tauchen Mean:",
    tauchen_mean_stat,
    "Rouwen Mean:",
    rouwen_mean_stat,
    "Mean:",
    0,
)
print(
    "Checking the unconditional sd....",
    "Tauchen sd:",
    tauchen_sd_stat,
    "Rouwen sd:",
    rouwen_sd_stat,
    "sd:",
    SIGMA / (1 - RHO**2) ** 0.5,
)
```

The next step is to simulate using both methods. We just have to choose the number of simulations (`LEN_HIST`) in the following script.


```python
# Monte Carlo simulations to compare performance
# seed random number generator
seed(12345)

LEN_HIST = 10000

histories_tauchen_st = np.zeros((LEN_HIST), np.int8)
histories_tauchen_z = np.zeros((LEN_HIST))
histories_tauchen_st[0] = 1
histories_tauchen_z[0] = zgrid_tauchen[histories_tauchen_st[0]]

histories_rouwen_st = np.zeros((LEN_HIST), np.int8)
histories_rouwen_z = np.zeros((LEN_HIST))
histories_rouwen_st[0] = 1
histories_rouwen_z[0] = zgrid_rouwen[histories_rouwen_st[0]]

randarray = rand(LEN_HIST)

for j in range(1, LEN_HIST):
    for r in range(0, N_GRID):
        if randarray[j] < np.cumsum(P_tauchen[histories_tauchen_st[j - 1], :])[r]:
            histories_tauchen_z[j] = zgrid_tauchen[r]
            histories_tauchen_st[j] = r
            break
    for r in range(0, N_GRID):
        if randarray[j] < np.cumsum(P_rouwen[histories_rouwen_st[j - 1], :])[r]:
            histories_rouwen_z[j] = zgrid_rouwen[r]
            histories_rouwen_st[j] = r
            break

plt.subplot(1, 2, 1)
plt.plot(histories_tauchen_z[:])
plt.plot(histories_rouwen_z[:])
plt.title("Tauchen vs. Rouwenhorst")

fig = plt.gcf()
fig.set_size_inches(16, 5)
plt.show()
```

Finally, we want to measure the performance of the simulations. In other words, we are going to compute the mean, variance, and autocorrelation coefficient of both series and compare them with the real ones. Notice that we add a burn-in period (`T_EXCLUDE`), meaning an extra piece that we add at the start of the time series while simulating but throw away later. When you simulate a time series, the first part you simulate will not follow your chosen model and must be discarded.


```python
# Compute mean, variance, and autocorrelation coefficient of both series
# Exclude the initial periods (Burn-in)
T_EXCLUDE = 100
tauchen_mean = np.mean(histories_tauchen_z[T_EXCLUDE:LEN_HIST])
rouwen_mean = np.mean(histories_rouwen_z[T_EXCLUDE:LEN_HIST])

print("Real mean:", 0, "Tauchen mean:", tauchen_mean, "Rouwen mean:", rouwen_mean)

tauchen_sd = np.std(histories_tauchen_z[T_EXCLUDE:LEN_HIST])
rouwen_sd = np.std(histories_rouwen_z[T_EXCLUDE:LEN_HIST])

print(
    "Real sd:",
    SIGMA / (1 - RHO**2) ** 0.5,
    "Tauchen sd:",
    tauchen_sd,
    "Rouwen_sd:",
    rouwen_sd,
)

tauchen_cov = np.cov(
    histories_tauchen_z[T_EXCLUDE : LEN_HIST - 1],  # noQA E203
    histories_tauchen_z[T_EXCLUDE + 1 : LEN_HIST],  # noQA E203
)
rouwen_cov = np.cov(
    histories_rouwen_z[T_EXCLUDE : LEN_HIST - 1],  # noQA E203
    histories_rouwen_z[T_EXCLUDE + 1 : LEN_HIST],  # noQA E203
)

tauchen_rho = tauchen_cov[0, 1] / tauchen_cov[0, 0]
rouwen_rho = rouwen_cov[0, 1] / rouwen_cov[0, 0]

print("Real rho:", RHO, "Tauchen rho:", tauchen_rho, "Rouwen rho:", rouwen_rho)
```

Of course, we can improve the results by using more simulations, playing with the "grid size" in both methods and/or with "LAMBDA" in Tauchen's method.

The above algorithms are already available, for instance, in the following Python library: https://quanteconpy.readthedocs.io/en/latest/markov/approximation.html#



## **4. Conclusion**

In this lesson, we have worked through the concept of the Markov Chain and its application to a simple bond valuation case study. In addition, we have introduced the gambler's ruin problem and the Tauchen and Rouwenhorst discretization methods. In the next lesson, we will start with the Hidden Markov Model. 

See you there!

---
Copyright 2025 WorldQuant University. This
content is licensed solely for personal use. Redistribution or
publication of this material is strictly prohibited.

