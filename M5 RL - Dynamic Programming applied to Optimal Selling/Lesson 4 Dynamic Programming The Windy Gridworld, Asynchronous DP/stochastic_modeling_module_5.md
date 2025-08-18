## STOCHASTIC MODELING
MODULE 5 | LESSON 4


---

# **REINFORCEMENT LEARNING: Windy Gridworld**

|  |  |
|:---|:---|
|**Reading Time** |  75min |
|**Prior Knowledge** |Markov process, gridworld, q-function  |
|**Keywords** |Reinforcement Learning, Windy gridworld


---

*In this notebook, we adapt the algorithms of the previous lesson to include some stochastic factors that influence the transitions. Make sure you understand this properly because it is going to lay the foundation for the next modules.*<span style='color: transparent; font-size:1%'>All rights reserved WQU WorldQuant University QQQQ</span>

## **1. Windy Gridworld**

In the previous lesson example, the agent had full control of the transitions from one state to another. We can easily adapt the algorithms to include some stochastic factors that influence the transitions. Let's analyze the solution to the gridworld model where we introduce a simple stochastic transition across cells.

Consider the gridworld model solved in the last lesson, but now there is a variable crosswind running downward through the grid. Suppose that at each state, there exists some probability $p_{down}$ that the next transition is a "down" move due to windy conditions, regardless of the actual decision of the agent and the state.    

Because the underlying problem has changed, we need to redefine a new policy evaluation function that takes into account the probability of a downward move.


```python
# Import libraries to use in this notebook
import time

import matplotlib.pyplot as plt
import numpy as np

# SOLVING THE GRIDWORLD EXAMPLE.
# Setting up the model parameters and transitions

N = 16
GRID_WIDTH = np.sqrt(N)
S_GRID = np.linspace(1, N - 2, N - 2)
A = 4  # number of actions

# Build an array that indicates, for each state, the destination cells
# from moving up, down, right, or left
destinations = np.zeros((N, A), dtype=np.int8)
destinations[N - 1, :] = (N - 1) * np.ones((A), dtype=np.int8)
for ss in range(1, N - 1):
    # determine row of position ss in the grid
    row_ss = np.floor(ss / GRID_WIDTH) + 1

    destinations[ss, 0] = (ss - GRID_WIDTH) * (ss - GRID_WIDTH >= 0) + ss * (
        ss - GRID_WIDTH < 0
    )
    destinations[ss, 1] = (ss + GRID_WIDTH) * (ss + GRID_WIDTH <= N - 1) + ss * (
        ss + GRID_WIDTH > N - 1
    )
    destinations[ss, 2] = (ss + 1) * (ss < row_ss * GRID_WIDTH - 1) + ss * (
        ss >= row_ss * GRID_WIDTH - 1
    )
    destinations[ss, 3] = (ss - 1) * (ss > (row_ss - 1) * GRID_WIDTH) + ss * (
        ss <= (row_ss - 1) * GRID_WIDTH
    )


def policy_improvement(value, nactions):
    """
    Return the improved policy given a value function
    """
    nstates = value.shape[0]
    policy = np.zeros((nstates, nactions))
    for ss in range(1, nstates - 1):
        policy_opt = np.zeros((nactions))
        val0 = -np.Inf
        for aa in range(0, nactions):
            val1 = -1 + value[destinations[ss, aa]]
            if val1 > val0:
                aa_opt = aa
                val0 = val1
        policy_opt[aa_opt] = 1.0
        policy[ss, :] = policy_opt

    return policy


# Define policy evaluation function in windy gridworld
def policy_evaluation_windy(policy, pwind, tol, iten):
    """
    Return the value function given a policy choice and the probability of
    windy conditions pwind
    """
    nstates, nactions = policy.shape[0], policy.shape[1]
    value = np.zeros((nstates))
    for ite in range(1, iten):
        value_old = value.copy()
        value = np.zeros((nstates))
        for ss in range(1, nstates - 1):
            valdown = value_old[destinations[ss, 1]]  # value if there is wind
            value[ss] = pwind * valdown + (1 - pwind) * np.dot(
                policy[ss, :], -np.ones(nactions) + value_old[destinations[ss, :]]
            )
        if np.max(np.absolute(np.subtract(value_old, value))) < tol:
            break

    return value, ite
```

Study the block of code below, which solves the model with policy iteration, and how we introduce the windy conditions in the environment. We can observe how the $Q$-function is now asymmetric, given that the agent leans against the wind when making upward moves, leading to some preference for downward moves. 

Explore how the optimal choices change for different values of $p_{down}$ or under different wind directions.


```python
# Set probability of down movement
PDOWN = 0.1

# Assume an initial policy that is random
policy = np.ones((N, A)) / A

value = np.zeros((N))

ITEMAX = 1000
ITETOL = 1e-4

for ite0 in range(ITEMAX):  # noQA E203
    # Policy evaluation, given a policy
    value, ite = policy_evaluation_windy(policy, PDOWN, ITETOL, ITEMAX)
    # Policy improvement, given a value
    policy_old = policy.copy()
    policy = policy_improvement(value, A)
    if np.max(np.absolute(np.subtract(policy_old, policy))) < ITETOL:  # noQA E203
        break

    print(ite0 + 1, ite + 1)

plt.subplot(1, 2, 1)
plt.plot(S_GRID, value[1 : N - 1])  # noQA E203
plt.title("Value function", fontsize="x-large")
plt.xlabel("Cell", fontsize="x-large")

plt.subplot(1, 2, 2)
plt.plot(S_GRID, policy[1 : N - 1, :])  # noQA E203
plt.title("Policies", fontsize="x-large")
plt.xlabel("Cell", fontsize="x-large")
plt.legend(["Up", "Down", "Right", "Left"])

fig = plt.gcf()
fig.set_size_inches(16, 6)
plt.show()
```

## **2. Asynchronous Dynamic Programming**

A major drawback to the DP methods above is that they involve operations over the entire state-space. If the dimensionality of the states is very large, then even one evaluation can be computationally expensive. This is known as the *curse of dimensionality*.

Asynchronous DP algorithms are iterative algorithms that update the values of states in any order, using whatever values of other states happen to be available. The values of some states may be updated several times before the values of others are updated once. For convergence, the asynchronous algorithm must still update the values of all the states at some point in the computation. Asynchronous DP algorithms allow great flexibility in selecting states to update.

For example, synchronous value iteration waits until all states are updated in order to obtain a new value function. That is, to obtain $v^{(1)}(s)$ we use the value function $v^{(0)}(s')$ obtained in the previous iteration. A version of asynchronous DP, named "in-place" value iteration, exploits at each iteration whatever new updates of $v^{(1)}(s')$ exist at the time of updating to estimate $v^{(1)}(s)$. That is, in every "sweep" or loop that goes through the state space we compute $v^{(1)}(s)$ using new updates $v^{(1)}(s')$ available in other states.

Another alternative is to update a selection of states that generated the highest updating error in previous steps of the iteration. This is called "prioritized sweeping." We can maintain a queue of a certain number of states that receive priority when updating, reducing the computational burden of each sweep through the state-space.

Another method, which we will cover in Module 7, is to use "Real-Time DP" that updates the optimization objects according to the experience obtained by the agent in the environment, leaving aside "irrelevant" states. 

The following example extends the investor problem we developed at the beginning of this module. The example is meant to provide a view on the large amount of computing time that might be needed to solve a seemingly simple model and how asynchronous DP with "in-place" value iteration may help.


## **3. The Investor's Selling Timing Problem with Two Assets**

As in the first example of this module, consider the situation of an investor that owns a financial asset, Asset A, whose dividends $s_t\geq 0$ fluctuate randomly according to a Markov process that takes values in $\{s_1,...,s_N\}$, with $s_1=0$ being an absorbing state (say, bankruptcy and liquidation), and $s_i<s_{i+1}$ for $i=1,2,...,N$.

The dividend Markov process is defined with the transition matrix we introduced above:
$$
\begin{align}
& \mathbb{P}(s_i|s_{i-1}) = \mathbb{P}(s_i|s_{i+1}) = 0.5\ \text{for}\ i=2,...,N-1 \\
& \mathbb{P}(s_{1}|s_{1}) = \mathbb{P}(s_{N-1}|s_{N}) = 1
\end{align}
$$
The agent can make trades on another asset, Asset B, whose dividends follow the same distribution as above, and are independent of Asset 0's dividends. We now assume that implementing any trading decision implies a transaction cost $C$ for the investor.  

We assume that the investor currently holds one of the assets, say Asset A. At each time step, the investor must choose among (i) holding the asset or (ii) selling the asset at a price $\ell$ and buying Asset B at a price $k$, or (iii) selling the asset at a price $\ell$ and doing nothing else. Thus, the action space is $\mathcal{A}=\{hold,sell\&buy,sell\}$. The state space is a three-dimensional array that includes the dividend realizations of each asset and which asset the investor is holding, which we denote by the variable $pos$. We assume that $pos$ takes value of zero when the investor holds Asset A and takes the value of one otherwise. Thus, a realization of the state is the triple $(s_1,s_2,pos)$. 

The rewards for the investor are $r(s_1,s_2,pos)=s_1(1-pos) + s_2 pos$ when the investor holds, and we need to take into account the term  $\ell - k - 2C$ when the manager sells the asset and purchases the alternative, and $\ell - C$ when the manager chooses to liquidate the position in one of the assets without purchasing the other.

Let's find the optimal solution to this problem through value iteration.


First, we set up the elements of the model. We assume that the dividend realizations take place in the range $[0,30]$. We create a list of states for each potential combination of dividend realizations. We set the transaction cost parameter to 1 and the discounting parameter to 0.75. We also set the available purchase and selling prices for each security at the levels across assets and across trading choices. 

Using the terminology from options trading, the investor owns a "straddle" on both assets. That is, the investor owns a put and a call on the same asset with the same exercise price. The expiration time is not explicitly modeled; it is implicitly assumed that the options have no maturity or they are continuously rolled over. 


```python
# SOLVING THE INVESTOR'S PROBLEM WITH TWO ASSETS
# Set up the states and Markov process
S_MIN = 0
S_MAX = 30
S_MAX2 = 30
N = S_MAX + 1
N2 = S_MAX2 + 1

# List of states
STATES = [(s0, s1) for s0 in range(S_MIN, S_MAX + 1) for s1 in range(0, S_MAX2 + 1)]

COST = 1

GAMMA = 0.75

SELLING_PRICE = [50, 50]
BUYING_PRICE = [50, 50]
```

Second, we build the transition matrix. This might be the most cumbersome part of setting up DP problems with large state spaces. In this case, the assumption of independence between the dividends of the assets makes our life easy. Specifically, we just need to multiply the probability of each transition in those states with non-zero probability.

For this purpose, we build a transition matrix with four dimensions where the first two represent the current realizations of both dividends and the last two are the future combinations of dividends. Thus, each entry of the array tells us the probability of transitioning from a state $(s_1,s_2)$ to state $(s_1',s_2')$.


```python
# Build the transition matrix
P = np.zeros((N, N2, N, N2))
for ss0 in STATES:
    for ss1 in STATES:
        pr0 = 0
        pr1 = 0
        if ss0[0] == 0:
            if ss1[0] == 0:  # Liquidation is an absorbing state
                pr0 = 1
        elif ss0[0] == S_MAX and ss1[0] == S_MAX - 1:  # Dividend bounds back
            pr0 = 1
        elif (
            ss1[0] == ss0[0] + 1 or ss1[0] == ss0[0] - 1
        ):  # Transitions from remaining states
            pr0 = 0.5

        if pr0 > 0:  # The joint prob. will be zero otherwise, we skip if pr0 = 0
            if ss0[1] == 0:
                if ss1[1] == 0:  # Liquidation is an absorbing state
                    pr1 = 1
            elif ss0[1] == S_MAX2 and ss1[1] == S_MAX2 - 1:  # Dividend bounds back
                pr1 = 1
            elif (
                ss1[1] == ss0[1] + 1 or ss1[1] == ss0[1] - 1
            ):  # Transitions from remaining states
                pr1 = 0.5

        P[ss0[0], ss0[1], ss1[0], ss1[1]] = pr0 * pr1
```

Next, we are going to define a function that performs the value iteration algorithm given the inputs of the model. Notice how value iteration navigates through each current state, taking into account the potential reward of each action, and then sums the expected continuation values across all future states.

We are going to compare the performance of asymmetric DP, so first, we state the standard DP problem.


```python
# Define the value iteration function


def value_iteration_investor(states, nstates, ptrans, sell_price, buy_price, tol, iten):
    """
    Returns the value and policy functions from value iteration
    """
    value = np.zeros((nstates[0], nstates[1], 3))
    for ite in range(1, iten):
        value_old = value.copy()
        value = np.zeros((nstates[0], nstates[1], 3))
        policy = np.zeros((nstates[0], nstates[1], 3))
        for ss in states:
            for pos in range(2):
                val = np.zeros(3)
                # Value of Selling and Purchasing the other asset (initialize)
                val[1] = sell_price[pos] - buy_price[pos] - 2 * COST
                # Value of Selling Permanently
                val[2] = sell_price[pos] - COST
                for ss_prime in states:
                    p = ptrans[ss[0], ss[1], ss_prime[0], ss_prime[1]]
                    if p > 0:
                        # Value of Holding
                        val[0] += p * (
                            ss_prime[0] * (1 - pos)
                            + ss_prime[1] * pos
                            + GAMMA * value_old[ss_prime[0], ss_prime[1], pos]
                        )
                        # Value of Selling and Purchasing the other asset
                        val[1] += p * (
                            ss_prime[0] * pos
                            + ss_prime[1] * (1 - pos)
                            + GAMMA * value_old[ss_prime[0], ss_prime[1], 1 - pos]
                        )
                value[ss[0], ss[1], pos] = np.max(val)
                policy[ss[0], ss[1], pos] = np.argmax(val)
        delta = np.max(np.absolute(np.subtract(value_old, value)))
        print("Iteration {}: delta = {:.4f}".format(ite, delta))
        if delta < tol:
            break

    return value, policy, ite
```

The block of code below defines the function that performs value iteration in an asynchronous manner. Notice that we do not initialize to zeros the new guess for the value function, but rather, we use new updates whenever they are available. In other words, notice that the new updates exploit the array "value", instead of the array "value_old" as in the function above.


```python
# Define the value iteration function with asynchronous DP


def value_iteration_investor_asyn(
    states, nstates, ptrans, sell_price, buy_price, tol, iten
):
    """
    Returns the value and policy functions from value iteration using
    asynchronous, "in-place", dynamic programming
    """
    value = np.zeros(
        (nstates[0], nstates[1], 3)
    )  # Value function that will be updated within the iteration
    for ite in range(1, iten):
        value_old = value.copy()  # Value function inherited from previous iteration
        policy = np.zeros((nstates[0], nstates[1], 3))
        for ss in states:
            for pos in range(2):
                val = np.zeros(3)
                # Value of Selling and Purchasing the other asset (initialize)
                val[1] = sell_price[pos] - buy_price[pos] - 2 * COST
                # Value of Selling Permanently
                val[2] = sell_price[pos] - COST
                for ss_prime in states:
                    p = ptrans[ss[0], ss[1], ss_prime[0], ss_prime[1]]
                    if p > 0:
                        # Value of Holding
                        val[0] += p * (
                            ss_prime[0] * (1 - pos)
                            + ss_prime[1] * pos
                            + GAMMA * value[ss_prime[0], ss_prime[1], pos]
                        )
                        # Value of Selling and Purchasing the other asset
                        val[1] += p * (
                            ss_prime[0] * pos
                            + ss_prime[1] * (1 - pos)
                            + GAMMA * value[ss_prime[0], ss_prime[1], 1 - pos]
                        )
                value[ss[0], ss[1], pos] = np.max(val)
                policy[ss[0], ss[1], pos] = np.argmax(val)
        delta = np.max(np.absolute(np.subtract(value_old, value)))
        print("Iteration {}: delta = {:.4f}".format(ite, delta))
        if delta < tol:
            break

    return value, policy, ite
```

Let's now compare the computation performance of each method, using the same convergence criteria.


```python
# Classical value iteration
ITETOL = 1e-4
ITEMAX = 500

start_time = time.time()

value, policy, ite = value_iteration_investor(
    STATES, [N, N2], P, SELLING_PRICE, BUYING_PRICE, ITETOL, ITEMAX
)

print("\n--- {:.2f} seconds ---".format(time.time() - start_time))

# Asynchronous value iteration

start_time = time.time()

value, policy, ite = value_iteration_investor_asyn(
    STATES, [N, N2], P, SELLING_PRICE, BUYING_PRICE, ITETOL, ITEMAX
)

print("\n--- {:.2f} seconds ---".format(time.time() - start_time))
```

Asynchronous value iteration considerably outperforms the standard procedure. This example illustrates how to speed up the convergence by providing the algorithm with new updates of the value function as soon as it appears in each iteration.

Finally, let's plot the optimal outcomes of the iteration algorithms. Because the functions are three-dimensional, we can build separate contour plots when the investor holds each asset. Notice that the problem is symmetric across both assets, so the optimal policies are also symmetric. Notice how the value of an asset for the investor depends on what is the dividend realization of the other asset, although both assets have independent payoffs. This dependence would disappear by ruling out the ability of the manager to switch from one asset to the other. How can we achieve that by changing the parameters in the model above? We leave this as an exercise.

The optimal policies prescribe when to exercise the put and call options that underlie this setup. Trivially, the options are exercised, *action 1*, when the dividend realizations in the holding asset are low and in the other are high. Otherwise, the investor holds, *action 0*. The sole exercise of the put option, *action 2*, takes place only when both assets' dividends are low so that they are close to liquidation.


```python
fig, ax = plt.subplots(1, 1)
cp = ax.contourf(value[:, :, 0])
fig.colorbar(cp)  # Add a colorbar to a plot
ax.set_title("Value of holding Asset A")
ax.set_xlabel("Dividend Asset B")
ax.set_ylabel("Dividend Asset A")
plt.show()

fig, ax = plt.subplots(1, 1)
cp = ax.contourf(value[:, :, 1])
fig.colorbar(cp)  # Add a colorbar to a plot
ax.set_title("Value of holding Asset B")
ax.set_xlabel("Dividend Asset B")
ax.set_ylabel("Dividend Asset A")
plt.show()

fig, ax = plt.subplots(1, 1)
cp = ax.contourf(policy[:, :, 0])
fig.colorbar(cp)  # Add a colorbar to a plot
ax.set_title("Optimal policy when holding Asset A")
ax.set_xlabel("Dividend Asset B")
ax.set_ylabel("Dividend Asset A")
plt.show()

fig, ax = plt.subplots(1, 1)
cp = ax.contourf(policy[:, :, 1])
fig.colorbar(cp)  # Add a colorbar to a plot
ax.set_title("Optimal policy when holding Asset B")
ax.set_xlabel("Dividend Asset B")
ax.set_ylabel("Dividend Asset A")
plt.show()
```

*Note: The problem above is a re-adaptation of a financial setup of Jack's Rental Car Problem in Chapter 4 of Sutton and Barto's book. A Python implementation of the solution to the problem appears in https://gist.github.com/pat-coady/71bbca1a2f64d96bb923ea979cf9b358*

**A final note on the efficiency of Dynamic Programming**

DP may not be practical for very large problems but is actually quite efficient. The worst-case computational time that DP methods take to find an optimal policy is polynomial in the number of states and actions. In this sense, DP is exponentially faster than any direct search in the policy space because direct search would have to exhaustively examine each policy to provide the same guarantee.

In practice, current computers can efficiently solve problems with millions of states using DP methods. Both policy iteration and value iteration are widely used; which method is better depends on the application at hand. Also, note that each dynamic programming problem has its own specificities and, although sharing the general characteristics stated above, require a careful formulation.

In the examples above, we have dealt with discrete spaces of states and actions. This considerably simplifies the exposition and coding. If any of the spaces takes values in a continuous set, for computation we need to rely on function approximation and interpolation to obtain the optimal policies at each point. These methodologies fall out of the scope of this course. 

## **4. Conclusion** 

In this lesson, we have worked through the example of the windy gridworld. In the next module, we will introduce the problem of multi-armed bandits with application to investment strategies.

See you there!

---
Copyright 2025 WorldQuant University. This
content is licensed solely for personal use. Redistribution or
publication of this material is strictly prohibited.

