# Case 1: Different mu, same sigma
mod_case1 = sm.tsa.MarkovRegression(y, k_regimes=2, trend='c', switching_variance=False)
res_case1 = mod_case1.fit()

# Case 2: Same mu, different sigma
mod_case2 = sm.tsa.MarkovRegression(y, k_regimes=2, trend='c', switching_variance=True, switching_trend=False)
res_case2 = mod_case2.fit()

# Case 3: Different mu, different sigma
mod_case3 = sm.tsa.MarkovRegression(y, k_regimes=2, trend='c', switching_variance=True)
res_case3 = mod_case3.fit()