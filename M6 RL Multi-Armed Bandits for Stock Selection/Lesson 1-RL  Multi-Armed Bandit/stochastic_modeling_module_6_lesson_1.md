## STOCHASTIC MODELING
MODULE 6 | LESSON 1


---

# **MULTI-ARMED BANDITS: Theory**

|  |  |
|:---|:---|
|**Reading Time** |  30 min |
|**Prior Knowledge** |Markov process, Optimization, Monte Carlo  |
|**Keywords** |Reinforcement learning, multi-armed bandits


---

*In this notebook, we introduce the theory necessary to understand the next lesson on the stationary k-bandit problem. This lecture represents the first building block in understanding more complex reinforcement learning methods. Notice that this lesson is only theory (**there is no Python code in this notebook**). Make sure you understand this properly because it is going to lay the foundation for future work.*<span style='color: transparent; font-size:1%'>All rights reserved WQU WorldQuant University QQQQ</span>

In this module, we are going to analyze one technique that deals with one of the essential ingredients for RL algorithms, the lack of knowledge or full modelization of the transitions between states. This means that the agent learns from the optimal actions to take just from experience. In the previous module, Dynamic Programming assumed perfect knowledge of how the environment transitions from one state to another. In many cases, it is almost unfeasible to obtain precise estimates of the transitions. Thus, in RL algorithms, the agent learns from repeated interactions with the environment. The most straightforward application of learning from unknown environments is Armed Bandits. In the last module, we will combine all our accumulated knowledge to develop fully fledged RL algorithms and problems.

## **1. Multi-Armed Bandits**

Consider the following learning situation. You repeatedly face a choice among $k$ different options or actions. After each action, you receive a reward from a probability distribution that depends on the action you selected. The objective is to maximize the expected total reward over some time period where you select some actions.

This type of environment is labeled as the "$k$-armed bandit" problem, analogous to a slot machine, or "one-armed bandit", with $k$ levers. Each action selection represents a play of one of the machine's levers, and the rewards are the associated payoffs. Through repeated action selections, our aim is to maximize our rewards by concentrating the actions on the best levers.

Another analogy is that of an investor choosing between stocks. Each action is the selection of a stock, and each reward is the return from the investment. Through experience, the investor may be able to learn which stocks tend to generate higher returns and select them appropriately. At the end of the module, we develop an example that uses real data with that environment in mind. 

From a finance-theoretical standpoint, it is important to bear in mind that developing an extraordinary profit-making ability by learning from the past represents either (i) a violation to the efficient market hypothesis, or (ii) that the investor is exposed to some source of systematic risk that generates the returns. Keeping this in mind, we are going to illustrate how this methodology can be implemented.

In our $k$-armed bandit problem, each of the $k$ actions has an expected or mean reward given that that action is selected: this is the *value* of that action. We denote the action selected at time $t$ as $a_t$, and the corresponding reward as $r_t$. Then, the value of an arbitrary action $a$, denoted $Q^*(a)$, is the expected reward given that $a$ is selected:
$$
\begin{align}
Q^*(a) = E\{r_t | a_t = a\}
\end{align}
$$
If you knew the value of each action, then it would be trivial to solve the $k$-armed bandit problem: you would always select the lever that yields the highest value. However, we never know the action values with certainty, although we may have more or less precise estimates. We denote the estimated value of action a at time step t as $Q_t(a)$. We would like $Q_t(a)$ to be close to $Q^*(a)$. One natural way to define it is $Q_t(a)$:
$$
\begin{align}
Q_t(a) = \frac{\sum_{s=1}^{t-1} r_t \mathbf{1}(a_s = a)}{\sum_{s=1}^{t-1} \mathbf{1}(a_s = a)}
\end{align}
$$
That is, $Q_t(a)$ represents the average reward obtained when $a$ was chosen. The simplest action selection rule is to select one of the actions with the highest estimated value; that is, we follow a *greedy* policy. If there is more than one greedy action, then a selection is made among them in some arbitrary way, perhaps randomly. We write this *greedy* action selection method as
$$
\begin{align}
A_t = \arg\underset{a}{\max} Q_t(a)
\end{align}
$$
Greedy action selection always exploits current knowledge to maximize immediate reward, but reduces the extent of exploration to learn about the reward of other alternatives. A simple alternative is to behave
greedily most of the time, but with some probability $\varepsilon$,  select randomly from among all the actions with equal probability, independently of the action-value estimates. We call methods using this near-greedy action selection rule *$\varepsilon$-greedy* methods.

It is easy to devise incremental formulas for updating averages with small, constant computation required to process each new reward. That is, we do not need to keep track of all the history of rewards. Given the estimated reward $Q_t(a)$ and and an observed reward after choosing $a$, $r_t$, the new average of all rewards can be computed as:
$$
\begin{align}
Q_t(a) = Q_{t-1}(a) + \frac{1}{N_t(a)}[r_t - Q_{t-1}(a)]
\end{align}
$$
where $N_t(a)$ measures the number of times that $a$ has been chosen up to and including time step $t$. This averaging method is appropriate for stationary problems, that is, when the reward probabilities do not change over time. We often encounter reinforcement learning problems that are effectively nonstationary: the underlying reward process changes over time. This is the most likely case in financial applications. In such cases, it makes sense to give more weight to recent rewards than to distant rewards. One of the most popular ways of doing this is to use a constant step-size parameter $\alpha\in(0,1)$:
$$
\begin{align}
Q_t(a) = Q_{t-1}(a) + \alpha[r_t - Q_{t-1}(a)]
\end{align}
$$
Notice that if $n$ denotes the number of times that action $a$ has been taken, we can re-express the function $Q(a)$ as just a weighted average of the past rewards obtained from choosing $a$ and a potential initial estimate $Q_0(a)$ (which might be zero in many instances):
$$
\begin{align}
Q_n(a) & = \alpha r_n + (1-\alpha)Q_{n-1}(a) \\
& = \alpha r_n + (1-\alpha)[\alpha r_{n-1} + (1-\alpha)Q_{t-2}(a)] \\
& = \alpha r_n + (1-\alpha)\alpha r_{t-1} + (1-\alpha)^2\alpha r_{n-2} + ... +(1-\alpha)^{n-1}\alpha r_{1} + (1-\alpha)^nQ_{0}(a)] \\
& =  (1-\alpha)^nQ_{0}(a) + \alpha\sum_{i=1}^n (1-\alpha)^{n-i}r_{i}
\end{align}
$$
Notice that the weight given to each reward $r_i$ declines as the number of new draws from choosing action $a$ increases.

## **2. Conclusion**

In this lesson, we have introduced multi-armed bandit models. In the next lesson, we will see how we can apply these models in practice using Python.

See you there!

---
Copyright 2025 WorldQuant University. This
content is licensed solely for personal use. Redistribution or
publication of this material is strictly prohibited.

