## STOCHASTIC MODELING
MODULE 4 | LESSON 1


---

# **MARKOV CHAINS**

|  |  |
|:---|:---|
|**Reading Time** |  60 minutes |
|**Prior Knowledge** |Linear Algebra, Maximum likelihood estimator, Monte Carlo  |
|**Keywords** |Markov Process, Markov chains, Transition Matrix


---

*In this notebook, we introduce Markov chains in a reinforcement learning framework with Python. This lecture represents the first building block in order to understand more complex reinforcement learning methods. Make sure you understand this properly because it is going to lay the foundation for future work.*

## **1. REINFORCEMENT LEARNING**

### **1.1 Introduction to Reinforcement Learning**

Machine learning provides automated methods that can detect patterns
in data and use them to achieve some tasks. Within ML methods, reinforcement learning (RL) is the task of learning how a decision-maker should take sequences of actions in order to optimize the cumulative rewards.

### **1.2 The RL Setting**

The general RL problem is formalized as a discrete time stochastic control problem. At each time step $t$, the environment is represented by a *state* $s_t\in\mathcal{S}$, where $\mathcal{S}$ represents the state space, i.e., all the possible situations where the agent can fall. Given $s_t$, the agent must take an action $a_t\in\mathcal{A}$ that implies (i) a reward $r_t\in\mathcal{R}$ for the agent, and (ii) a transition to state $s_{t+1}\in\mathcal{S}$.

Beginning at time $t=0$, the RL problem boils down to finding a sequence of actions $\{a_0,a_1,...,a_t,...\}$ that maximizes the cumulative rewards for the agent.

### **1.3 The Markov Property**
We are going to consider the stochastic control process with the Markovian property.

**Definition**. A discrete time stochastic control process is Markovian (i.e., it has the Markov property) if
*   $\mathbb{P}(s_{t+1}|s_{t},a_{t})=\mathbb{P}(s_{t+1}|s_{t},a_{t},...,s_{0},a_{0})$
*   $\mathbb{P}(r_{t}|s_{t},a_{t})=\mathbb{P}(r_{t}|s_{t},a_{t},...,s_{0},a_{0})$

The Markov property means that the future of the process only
depends on the current observation, and the agent has no interest in looking at the full history. Before studying a fully fledged case of Reinforcement Learning, we are going to focus on the properties of random process with the Markov property and some specific finance applications.

## **2. MARKOV CHAINS**

If the state space consists of countably many states, the Markov process is called a Markov chain, formally defined as follows:

A sequence of random variables $(S_0,S_1,S_2,...)$ is a Markov chain with state space $\mathcal{S}$ and transition matrix $P$ if, for all $t\geq 0$ and all sequences $(s_0,s_1,s_2,...,s_t,s_{t+1})$, we have that $\mathbb{P}(S_{t+1}|s_{t},s_{t-1},...,s_{1},s_{0})=\mathbb{P}(S_{t+1}|s_{t})$.

### **2.1 Homogeneous Markov Chains**

We say that a Markov chain with state space $\mathcal{S}=\{s_1,...,s_N\}$ is homogeneous if $\mathbb{P}(S_{t+1}=s_i | S_{t}=s_j)$ is constant for all $t$ and $(i,j)$. For homogeneous Markov chains, we can specify $\mathbb{P}(S_{t+1}=s_j | S_{t}=s_i) = p_{ij}$ and, then, the matrix $P(p_{ij})$ is the transition matrix for $S_t$, where it must satisfy that $\sum_i p_{ij} = 1$ for all $i$. 

A Markov chain's transition matrix P is a stochastic matrix, i.e., a square matrix of non-negative terms in which the elements in each row sum to one.

*Note: Some authors opt to define the transition matrix $P$ as the matrix whose entries $p_{ij}$ denote the probability that the state changes from $j$ to $i$. We prefer the notation with the convention above as the most intuitive.*

We have described how the one-step-ahead realizations of a Markov chain depend on its current realizations. How do the further-ahead realizations of the chain depend also on current realizations? For instance, notice that
$$
\begin{align}
\mathbb{P}(s_{t+2}=s_j | s_{t}=s_i) & = \sum_{k=1}^N \mathbb{P}(s_{t+2}=s_j | s_{t+1}=s_k, s_{t+1}=s_i)\mathbb{P}(s_{t+1}=s_k | s_{t}=s_i) \\
& = \sum_{k=1}^N \mathbb{P}(s_{t+2}=s_j | s_{t+1}=s_k)\mathbb{P}(s_{t+1}=s_k | s_{t+1}=s_i) \\
& =  \sum_{k=1}^N p_{kj}p_{ik}
\end{align}
$$
Notice that this result yields the $(i,j)^{th}$ position of the squared transition matrix, $P^2$.

Following the same logic, letting $p_{ij}^n$ denote the $(i,j)^{th}$ position of matrix $P^n$:
$$
\begin{align}
\mathbb{P}(s_{t+m}=s_j | s_{t+n}=s_i) = p_{ij}^{m+n} = \sum_{k=1}^N p_{kj}^n p_{ik}^m
\end{align}
$$

### **2.2 Example: A Two-state Homogeneous Markov Chain**

Let $\mathcal{S}=\{s_1,s_2\}$, with $s_1\lt s_2$, $\mathbb{P}(S_{t+1}=s_1|s_{1}) = p$, and $\mathbb{P}(S_{t+1}=s_2|s_{2})= q$. Then, we can represent the transition matrix for this chain as
$$
\begin{align}
P = \begin{bmatrix}
p & 1-p \\
1-q & q
\end{bmatrix}
\end{align}
$$
If we multiply the transition matrix by the vector that represents the state space of the process, we obtain the expectation of the next period's state, conditional on each state. In our simple 2-state case:
$$
\begin{align}
\begin{bmatrix}
p & 1-p \\
1-q & q
\end{bmatrix}
\begin{bmatrix}
s_1 \\
s_2
\end{bmatrix}
=
\begin{bmatrix}
E(s_{t+1}|s_t = s_1) \\
E(s_{t+1}|s_t = s_2)
\end{bmatrix}
\end{align}
$$
Suppose that a process $X_t$ is such that $X_t = X_{t-1} + s_t$, where $s_t$ is the Markov process above. Let's simulate the process for some arbitrary parameters.


```python
# LIBRARIES WE USE IN THE NOTEBOOK
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand, seed
```


```python
N_STATES = 2
S = np.zeros((2, 1))
P = np.zeros((2, 2))

S[0] = -1
S[1] = 1

P[0, 0] = 0.55
P[0, 1] = 1.0 - P[0, 0]
P[1, 1] = 0.55
P[1, 0] = 1.0 - P[1, 1]

# Display the state vector
print("State vector:\n", S)
# Display the transition matrix
print("Transition matrix:\n", P)
# Display the conditional mean vector
S_condmean = np.dot(P, S)
print("Conditional mean vector:\n", S_condmean)
```


```python
# SAMPLING FROM THE MARKOV CHAIN
# Simulate a sequence from the chain
# seed random number generator
np.set_printoptions(precision=3, suppress=True)
seed(12345)

LEN_HIST = 50
states = np.zeros((LEN_HIST, 2), np.int8)
Xarray = np.zeros(LEN_HIST)
Psim = np.zeros((2, 2))
S_DICT = dict(
    [
        (
            "1",
            0,
        ),
        ("2", 1),
    ]
)
S_VAL = dict(
    [
        (
            "1",
            -1,
        ),
        ("2", 1),
    ]
)

print(S_VAL["1"])

# Generate sequence of uniform random numbers
randarray = rand(LEN_HIST)
# Initialize process s_0, say at state 1 (0 in Python's vector notation)
states[0, :] = (S_DICT["1"], S_VAL["1"])
Xarray[0] = 75  # states[0]

for tt in range(1, LEN_HIST):
    if P[states[tt - 1, 0], states[tt - 1, 0]] > randarray[tt]:
        states[tt, :] = states[tt - 1, :]
    else:
        if states[tt - 1, 0] == S_DICT["1"]:
            states[tt, :] = [S_DICT["2"], S_VAL["2"]]
        else:
            states[tt, :] = [S_DICT["1"], S_VAL["1"]]
    Xarray[tt] = Xarray[tt - 1] + states[tt, 1]
    Psim[states[tt - 1, 0], states[tt, 0]] = (
        1.0 + Psim[states[tt - 1, 0], states[tt, 0]]
    )

# Plot the evolution of the X_t and s_t
plt.subplot(1, 2, 1)
plt.plot(Xarray)
plt.title("X_t")

plt.subplot(1, 2, 2)
plt.plot(states[:, 1])
plt.title("S_t")

fig = plt.gcf()
fig.set_size_inches(16, 5)
plt.show()
```

### **2.3 Estimating Markov Transition Probabilities from Transition Data**

Suppose that we observe the realizations of a Markov chain and wish to estimate the probabilities of the transition matrix $P$. If $n_{ij}$ is the number of times that we observe a change from state $i$ to state $j$ and $N$ is the total number of states, then the estimated transition probabilities can be computed as:
$$
\begin{align}
\widehat{p}_{ij} = \frac{n_{ij}}{\sum_{k=1}^N n_{ik}}
\end{align}
$$

One can show that this expression corresponds to the Maximum Likelihood Estimator for $p_{ij}$. Let's verify that the Monte Carlo exercise yields the correct transition matrix.


```python
# Compute estimated transition matrix from the Monte Carlo exercise
Pest = (Psim.T / np.sum(Psim, axis=1)).T

print(Pest)
```

Take into account that in order to increase the accuracy of the estimation, it is important to increase the number of Monte Carlo simulations ("LEN_HIST" in the code above).

### **2.4 Stationary Distribution of a Markov Chain**

Consider the matrix $P^n$, whose $(i,j)^{th}$ element is $p_{ij}^n = \mathbb{P}(s_{t+n}=s_j | s_{t}=s_i)$. If memory of the past dies out with increasing $n$, then we would expect the dependence of $p_{ij}^n$ on both n and i to disappear as $n→\infty$. This means that $P^n$ should converge to a limit as $n→\infty$ and that for each column $j$, its elements should all converge toward the same value $\pi_j$, and all its rows should be identical.

Notice that $p_{ij}^{n+1} = \sum_{k=1}^N p_{kj} p_{ik}^n$. Assuming convergence, this means that the limiting distribution as $n→\infty$ satisfies $\pi_{j}^{n+1} = \sum_{k=1}^N p_{kj} \pi_{k}$. In vector form:
$$
\begin{align}
\pi = \pi P ;\ \text{where}\ \pi=\{\pi_1,...,\pi_N\}, \pi_k\geq 0\  \text{and}\ \sum_k\pi_k = 1
\end{align}
$$
If the row-vector $\pi$ is taken as the initial probability distribution of the Markov chain, then $\pi = \pi P = \pi P^2= \pi P^3=...$, and we say that the chain is in steady state.

Under what conditions does convergence take place? Under what conditions is $\pi$ unique? While we are going to omit the details, convergence is essentially assured by the fact that $P$ is a stochastic matrix whose largest eigenvalue is one. Uniqueness is guaranteed under the property ergodicity. This means that, first, there is positive probability of traveling from any state to any other. Second, it means that the states are not partitioned into sets such that all state transitions occur cyclically from one set to another.


```python
# Compute estimated stationary distribution from the Monte Carlo exercise
Pstat = [1.0 - np.sum(states[:, 0]) / LEN_HIST, np.sum(states[:, 0]) / LEN_HIST]

print(Pstat)

# Compute the stationary distribution from the transition matrix.
# We can follow two methods:
# 1. Matrix multiplication
P2 = np.matmul(P, P)
print(P2)
P3 = np.matmul(P2, P)
print(P3)
P10 = np.linalg.matrix_power(P, 10)
print(P10)
P40 = np.linalg.matrix_power(P, 40)
print(P40)

# Sequential multiplication of a vector
p_stat = 0.5 * np.ones((2))
for t in range(1, 100):
    p_stat = np.dot(P.T, p_stat)

print(p_stat)

# Check this results with the matrix Pstat above (both methods should give the "same" results)
```

Again, you should be able to increase the accuracy of "Pstat" by increasing the number of Monte Carlo simulations.

**Absorbing Markov chains**

Absorbing Markov chains play a prominent role in Finance, for example, by identifying default as an absorbing state.

If a Markov chain is homogeneous, we say that state $i$ is absorbing if $p_{ii}=1$. That is, once the Markov chain $S_t$ enters an absorbing state, it stays there forever.

## **3. Application to Credit Ratings**

Credit ratings provide investors invaluable information to assess firms' abilities to meet their debt obligations, i.e., the probability of default. Ratings are published periodically and change from time to time to reflect firms' changes in credit risk. It is useful to use a Markov chain model to describe the dynamics of a firm's credit rating.

Consider a homogeneous Markov chain $S_t$ defined on the state space $\mathcal{S}=\{1,2,...,D-1,D\}$ with transition matrix $P$. State 1 represents the highest credit class, state 2 the second highest, and so on, while state $D-1$ refers to the lowest credit class, and state $D$ represents default.

The table below shows the actual transition probabilities for credit ratings published by Standard & Poor's ($NR$ stands for "rating withdrawn").<span style='color: transparent; font-size:1%'>All rights reserved WQU WorldQuant University QQQQ</span>

![SP Transition matrix 2021.png](data:image/png;base64,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)

In the next example, we consider the transition matrix as given (see above) and we are gonna simulate how long it takes a firm to default starting with some current rating. 


```python
# seed random number generator
seed(12345)

np.set_printoptions(precision=3, suppress=True)

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

# Normalize transition matrix, ignoring NR type
P = P0[:, 0 : P0.shape[1] - 1]  # noQA E203
P = (P.T / np.sum(P, axis=1)).T

# Verify where the ratings process converges given these estimates
P10 = np.linalg.matrix_power(P, 10)
print(P10)

P40 = np.linalg.matrix_power(P, 200)
print(P40)

# SAMPLING FROM THE MARKOV CHAIN
# Simulate how long it takes a firm to default starting with some current rating
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
CURR_RATING = "CCC"

N_HISTORIES = 1000
LEN_HIST = 100
histories = np.zeros((N_HISTORIES, LEN_HIST), np.int8)
histories[:, 0] = RATINGS[CURR_RATING]
randarray = rand(N_HISTORIES, LEN_HIST)

default_time = np.zeros(N_HISTORIES)
default_sum = 0

for i in range(0, N_HISTORIES):
    for j in range(1, LEN_HIST):
        for r in RATINGS:
            if randarray[i, j] < np.cumsum(P[histories[i, j - 1], :])[RATINGS[r]]:
                histories[i, j] = RATINGS[r]
                break
        if histories[i, j] == RATINGS["D"]:
            break
    # Compute the average time to default
    if np.max(histories[i, :]) == RATINGS["D"]:
        where_default = np.where((histories[i, :] == RATINGS["D"]))
        default_time[i] = where_default[0][0]
        default_sum += 1
    else:
        default_time[i] = 0.0

print("Default time:", np.sum(default_time) / default_sum)
```

## **4. Conclusion**

In this lesson, we have worked through the concept of a Markov Chain and its application to a simple credit rating case study. In the next lesson, we will apply these concepts to bond valuation.

See you there!

---
Copyright 2025 WorldQuant University. This
content is licensed solely for personal use. Redistribution or
publication of this material is strictly prohibited.

