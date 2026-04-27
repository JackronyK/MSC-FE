# Formulas for Kupiec and Christoffersen Tests

Based on the Ramos-Pérez et al. (2021) paper, here are the key formulas for VaR backtesting validation:

## Kupiec Test (Unconditional Coverage Test)

The Kupiec test evaluates whether the observed frequency of VaR exceedances matches the expected frequency at the chosen confidence level.

**Test Statistic:**
$$LR_{uc} = -2 \ln\left[\frac{(1-p)^{T-N} p^N}{(1-\hat{p})^{T-N} \hat{p}^N}\right]$$

**Where:**
- $T$ = total number of observations in the test period
- $N$ = number of VaR exceedances (violations where actual loss > VaR)
- $p$ = expected failure rate (e.g., $p = 0.005$ for 99.5% VaR per Solvency II)
- $\hat{p} = N/T$ = observed failure rate

**Distribution:** Under $H_0$: $LR_{uc} \sim \chi^2(1)$

**Null Hypothesis:** The model produces the correct unconditional coverage (exceedance rate equals $1-\alpha$)

---

## Christoffersen Test (Conditional Coverage Test)

The Christoffersen test jointly tests unconditional coverage AND independence of exceedances (no clustering).

**Combined Test Statistic:**
$$LR_{cc} = LR_{uc} + LR_{ind}$$

**Independence Component ($LR_{ind}$):**
$$LR_{ind} = -2 \ln\left[\frac{(1-\pi)^{n_{00}+n_{10}} \pi^{n_{01}+n_{11}}}{(1-\pi_{01})^{n_{00}} \pi_{01}^{n_{01}} (1-\pi_{11})^{n_{10}} \pi_{11}^{n_{11}}}\right]$$

**Where:**
- $n_{ij}$ = number of transitions from state $i$ to state $j$ (where $i,j \in \{0,1\}$, with 1 = violation, 0 = no violation)
- $\pi = (n_{01} + n_{11})/(n_{00} + n_{01} + n_{10} + n_{11})$ = unconditional probability of violation
- $\pi_{01} = n_{01}/(n_{00} + n_{01})$ = probability of violation following a non-violation
- $\pi_{11} = n_{11}/(n_{10} + n_{11})$ = probability of violation following a violation

**Distribution:** Under $H_0$: $LR_{cc} \sim \chi^2(2)$

**Null Hypothesis:** Exceedances are independently distributed with correct unconditional coverage

---

# Data Splitting Strategy (Ramos-Pérez et al., 2021)

## 1. Training Data

| Component | Specification |
|-----------|--------------|
| **Date Range** | January 1, 2008 – December 31, 2015 |
| **Purpose** | Hyperparameter optimization (selecting optimal dropout level θ, architecture configuration) |
| **Rolling Window Size** | 650 trading days (fixed sample length) |
| **Forecast Horizon** | 1 day ahead |
| **Optimization Criterion** | Minimize RMSE on training period forecasts |

The rolling window approach means that for each forecast date $t$, the model is fitted using observations from $[t-650, t-1]$, then predicts volatility for day $t$.

## 2. Testing Data

| Component | Specification |
|-----------|--------------|
| **Date Range** | January 1, 2016 – December 31, 2020 |
| **Purpose** | Out-of-sample performance evaluation; final model comparison |
| **Rolling Window Size** | 650 trading days (same as training) |
| **Forecast Horizon** | 1 day ahead |
| **Evaluation Metrics** | RMSE, MAE, Kupiec test, Christoffersen test |

**Key Point:** The window size remains constant at 650 days throughout both training and testing—only the time period shifts forward.

## 3. How Hybrid Methodologies Optimize Data Usage

Hybrid models (e.g., T-GARCH, MTL-GARCH) combine:

```
Traditional GARCH-family models + Deep Learning Layers
```

**Optimization benefits:**

| Benefit | Explanation |
|---------|-------------|
| **Parameter efficiency** | Transformer/Multi-Transformer layers capture complex temporal dependencies with fewer parameters than pure deep networks, reducing overfitting risk on limited financial data |
| **Complementary strengths** | GARCH models handle volatility clustering and leverage effects; neural layers capture non-linear patterns GARCH cannot model |
| **Robustness to regime changes** | Hybrid models adapt better to structural breaks (e.g., COVID-19 volatility spike in 2020) because neural components learn flexible representations |
| **Reduced retraining frequency** | Once optimal architecture is selected on training data, the same configuration generalizes to testing period without re-optimization |

From the paper: *"The optimum configuration of the models is obtained by applying the rolling window approach and selecting the configuration which minimizes the error (RMSE) in the period going from 1 January 2008 to 31 December 2015."*

## 4. Why Validation Is Important

| Reason | Practical Impact |
|--------|-----------------|
| **Prevents overfitting** | Financial time series have low signal-to-noise ratio; validation ensures selected hyperparameters generalize beyond in-sample fit |
| **Hyperparameter selection** | Dropout rate (θ), number of attention heads, LSTM units—all require empirical tuning on unseen data |
| **Model comparison fairness** | All models (benchmark and proposed) use identical validation protocol, enabling apples-to-apples comparison |
| **Risk measure reliability** | VaR backtesting (Kupiec/Christoffersen) on validation data identifies models that produce statistically valid risk estimates |
| **Regulatory compliance** | Solvency II/Basel frameworks require demonstrable out-of-sample performance; validation provides audit trail |

The paper demonstrates this: models optimized on 2008-2015 data were then evaluated on 2016-2020 data, with MTL-GARCH achieving the lowest RMSE (0.0038) and passing both backtesting tests (p-values > 0.05).

## 5. Applying This to Your Own Backtesting Project: Rolling Window Guidelines

### Step-by-Step Framework:

```
1. DEFINE YOUR UNIVERSE
   ├─ Asset class (equities, FX, crypto, multi-asset)
   ├─ Frequency (daily, hourly, minute)
   └─ Target horizon (1-day VaR, 10-day ES, etc.)

2. SELECT WINDOW SIZE (n)
   ├─ Rule of thumb: n ≥ 250 × forecast horizon (for daily data)
   ├─ Consider: 
   │  • Volatility regime stability (shorter windows adapt faster)
   │  • Parameter estimation needs (GARCH needs ~500+ obs)
   │  • Computational budget (larger n = slower retraining)
   └─ Empirical test: try n ∈ {250, 500, 750, 1000} and compare RMSE stability

3. SPLIT TEMPORALLY (not randomly!)
   ├─ Training: earliest 60-70% of data → hyperparameter tuning
   ├─ Validation: middle 10-20% → model selection
   └─ Testing: latest 20-30% → final performance report
   └─ Ensure no look-ahead bias: each forecast uses only past data

4. IMPLEMENT ROLLING FORECAST
   for t in test_period:
       train_window = [t-n, t-1]
       fit_model(data[train_window])
       predict_volatility(t)
       record_actual_return(t)
       compute_VaR_breach(actual, predicted)

5. EVALUATE COMPREHENSIVELY
   ├─ Point forecast accuracy: RMSE, MAE
   ├─ Risk measure validity: Kupiec (LR_uc), Christoffersen (LR_cc)
   ├─ Economic significance: capital requirements, P&L impact
   └─ Robustness: repeat across multiple assets/sub-periods

6. SENSITIVITY ANALYSIS
   ├─ Vary window size ±25%: does performance degrade gracefully?
   ├─ Test different start dates: is model sensitive to initial conditions?
   └─ Stress test: how does model perform during known crises?
```

### Practical Tips from the Paper:

| Challenge | Recommendation |
|-----------|---------------|
| **Small sample size** | Use simpler architectures; prioritize GARCH+FF over complex Transformers if n < 500 |
| **Non-stationarity** | Implement adaptive windowing: shorten window when volatility regime shifts detected |
| **Computational cost** | Pre-compute GARCH parameters; only retrain neural components in hybrid models |
| **Model selection uncertainty** | Use ensemble of top-3 configurations from validation rather than single "best" model |
| **Regulatory reporting** | Document all window choices, retraining frequency, and backtesting results for audit |

### Example Configuration for Daily Equity VaR:

```python
# Pseudocode inspired by Ramos-Pérez implementation
config = {
    'window_size': 650,           # ~2.5 years of daily data
    'forecast_horizon': 1,        # 1-day ahead
    'training_period': '2008-01-01 to 2015-12-31',
    'testing_period': '2016-01-01 to 2020-12-31',
    'retrain_frequency': 'daily', # rolling window: retrain every new observation
    'optimization_metric': 'RMSE',
    'backtesting_tests': ['Kupiec_99.5%', 'Christoffersen_99.5%']
}
```

**Final Recommendation:** Start with the paper's 650-day window as a baseline. If your asset exhibits higher volatility clustering or structural breaks, consider shorter windows (400-500 days) with more frequent retraining. Always validate that your chosen window produces stable Kupiec/Christoffersen p-values across multiple sub-periods before deploying to production.