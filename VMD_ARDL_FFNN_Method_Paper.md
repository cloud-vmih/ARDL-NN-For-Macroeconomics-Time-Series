# A VMD-ARDL-FFNN Framework for Time-Series Forecasting: Methodology, Forecast-Safe Inputs, and an Export Forecasting Experiment

## Abstract

This paper presents a hybrid methodology for nonlinear time-series forecasting based on Variational Mode Decomposition (VMD), Autoregressive Distributed Lag (ARDL) lag selection, and Feedforward Neural Networks (FFNNs). The objective is methodological: the export dataset used in the experiment is treated as an empirical illustration rather than the central subject of the paper. The proposed framework decomposes a target time series and its exogenous predictors into frequency-oriented components, selects forecast-safe autoregressive and distributed lags using ARDL principles, and trains FFNNs to learn nonlinear mappings from lagged inputs to future values. The final forecast is obtained by reconstructing the component forecasts. The paper formalizes the model inputs, equations, training procedure, validation strategy, and leakage-control design. A key feature of the implementation is walk-forward VMD caching: decomposition for validation and testing is performed only with information available at each forecast origin, avoiding the common error of decomposing the full series before forecasting. In the illustrative experiment on monthly Vietnamese exports to the United States, the VMD-ARDL-FFNN model improves test-set RMSE, MAE, MAPE, and directional accuracy relative to a direct ARDL-FFNN baseline, although validation results favor the simpler baseline. These findings suggest that VMD-ARDL-FFNN is best understood as a general hybrid forecasting architecture whose value depends on careful lag construction, out-of-sample validation, and leakage-aware decomposition.

**Keywords:** time-series forecasting; VMD; ARDL; FFNN; neural network; lag selection; signal decomposition; hybrid forecasting

## 1. Introduction

Time-series forecasting requires models that can represent persistence, delayed effects, nonstationarity, nonlinear interactions, and changing signal structure. Classical econometric models such as autoregressive and distributed-lag models are attractive because their inputs are explicit and interpretable. However, many real-world time series contain nonlinear patterns that are difficult to express with a purely linear specification. Neural networks address this weakness by learning flexible nonlinear functions, but they can overfit and become difficult to interpret when the input lag structure is not controlled.

Hybrid forecasting methods attempt to combine these strengths. In this paper, ARDL is used not mainly as a final linear forecasting equation, but as a principled lag-selection mechanism. FFNN is used as the nonlinear prediction engine. VMD is used as a signal decomposition layer that separates a time series into components before prediction. The combined architecture can be summarized as:

```text
raw time series -> stationarity transform -> VMD decomposition -> ARDL lag selection -> FFNN prediction -> reconstruction
```

The main methodological problem is not simply how to combine these tools, but how to combine them without look-ahead bias. Decomposition methods are especially risky in forecasting because full-sample decomposition uses future information when constructing past components. The framework therefore applies VMD in a walk-forward manner for validation and testing. At each forecast origin, only historical observations available before the prediction point are used.

This paper explains the full VMD-ARDL-FFNN method, including its mathematical form, input matrix construction, model training, validation, and forecast reconstruction. The empirical export forecasting task is presented only as an experiment demonstrating how the framework operates on a real monthly dataset.

## 2. Related Literature

The ARDL tradition provides a flexible way to model a dependent variable using its own lags and lagged explanatory variables. Pesaran and Shin introduced ARDL modeling for cointegration analysis, and Pesaran, Shin, and Smith later developed the bounds-testing framework for level relationships. Although the present framework focuses on forecasting rather than cointegration testing, ARDL remains useful because it gives a disciplined structure for autoregressive and distributed lag inputs.

Feedforward neural networks are supported by the universal approximation literature. Cybenko showed that neural networks with sigmoid activation can approximate continuous functions, while Hornik, Stinchcombe, and White established that multilayer feedforward networks are universal approximators under broad conditions. Rumelhart, Hinton, and Williams popularized backpropagation as a practical learning algorithm. These results justify the use of FFNNs as nonlinear function approximators, while also implying that practical performance still depends on architecture, regularization, optimization, and validation.

Hybrid linear-nonlinear time-series forecasting is also well established. Zhang proposed a hybrid ARIMA-neural network model to combine linear and nonlinear modeling. This idea motivates the present ARDL-FFNN design: ARDL contributes lag discipline and linear time-series structure, while FFNN learns the nonlinear mapping from selected lagged inputs.

VMD was proposed by Dragomiretskiy and Zosso as a variational method for decomposing a signal into modes with compact frequency support. It is related in spirit to earlier decomposition approaches such as empirical mode decomposition, but VMD is formulated as an optimization problem. In forecasting, decomposition can reduce complexity by allowing separate models to handle different components, but it must be implemented carefully to avoid future-information leakage.

## 3. Methodology

### 3.1 Problem Definition

Let the forecasting target be a univariate time series:

```math
y_t,\quad t = 1,2,\ldots,T
```

and let the exogenous predictor vector be:

```math
\mathbf{x}_t = (x_{1,t}, x_{2,t}, \ldots, x_{m,t})^\top.
```

The one-step-ahead forecasting objective is:

```math
\hat{y}_{t} = f(\mathcal{I}_{t-1}),
```

where \(\mathcal{I}_{t-1}\) is the information set available before time \(t\). A forecast-safe model must not use \(y_t\), \(\mathbf{x}_t\), or any future values when predicting \(y_t\), unless the forecasting setting explicitly assumes that contemporaneous exogenous values are known. In this implementation, exogenous lag 0 is excluded, so all predictors are lagged.

### 3.2 Stationarity and Scale Transformation

Before lag modeling, each variable may be transformed to improve stationarity. If log transformation is enabled and the series is positive, the base series is:

```math
z_t = \log(y_t).
```

If first differencing is selected, the model-scale target is:

```math
\Delta z_t = z_t - z_{t-1}.
```

The same principle is applied to exogenous variables. Stationarity decisions are made using the training sample only. This is important because selecting transformations using the whole dataset would leak validation and test information into model design.

When forecasts are produced in differenced log scale, they are reconstructed to level scale by reversing the differencing and log transformations:

```math
\hat{z}_t = z_{t-1} + \widehat{\Delta z}_t,
```

```math
\hat{y}_t = \exp(\hat{z}_t).
```

If no log transformation is used, the inverse mapping is adjusted accordingly.

### 3.3 Variational Mode Decomposition

VMD decomposes a signal \(s(t)\) into \(K\) modes:

```math
s(t) \approx \sum_{k=1}^{K} u_k(t),
```

where each mode \(u_k(t)\) is estimated to have compact bandwidth around a center frequency \(\omega_k\). The VMD optimization problem can be written conceptually as:

```math
\min_{\{u_k\}, \{\omega_k\}}
\sum_{k=1}^{K}
\left\|
\partial_t
\left[
\left(\delta(t) + \frac{j}{\pi t}\right) * u_k(t)
\right]
e^{-j\omega_k t}
\right\|_2^2
```

subject to:

```math
\sum_{k=1}^{K} u_k(t) = s(t).
```

In the implemented forecasting framework, VMD is applied to the transformed target and exogenous variables. The model also keeps a residual component:

```math
r(t) = s(t) - \sum_{k=1}^{K} u_k(t).
```

Thus, the effective component set is:

```math
\mathcal{C} = \{1,2,\ldots,K,\text{RES}\}.
```

For each component \(c \in \mathcal{C}\), the target component is denoted \(y_t^{(c)}\), and the exogenous component vector is denoted:

```math
\mathbf{x}_t^{(c)} = (x_{1,t}^{(c)}, x_{2,t}^{(c)}, \ldots, x_{m,t}^{(c)})^\top.
```

### 3.4 ARDL Lag Selection

The ARDL structure provides the lag candidates used as FFNN inputs. A general ARDL form for component \(c\) is:

```math
y_t^{(c)}
= \alpha^{(c)}
+ \sum_{i \in P_c} \phi_i^{(c)} y_{t-i}^{(c)}
+ \sum_{j=1}^{m} \sum_{\ell \in Q_{j,c}} \beta_{j,\ell}^{(c)} x_{j,t-\ell}^{(c)}
+ \varepsilon_t^{(c)}.
```

Here:

```text
P_c = selected autoregressive lags for the target component
Q_{j,c} = selected distributed lags for exogenous variable j in component c
```

The target lag candidates are generated from ACF/PACF information on the training set and screened on validation data. Exogenous lags are selected using ARDL information criteria such as AIC or BIC. The implementation searches lags up to a maximum target lag and maximum exogenous lag. Exogenous lag 0 is not allowed:

```math
\ell \ge 1.
```

This restriction makes the design forecast-safe because the model never uses same-period exogenous values to predict the target unless those values are known in advance.

### 3.5 FFNN Forecasting Model

After ARDL selects the lag sets, each component is converted into a supervised learning problem. For component \(c\), define the FFNN input vector:

```math
\mathbf{h}_t^{(c)}
=
\left[
\{y_{t-i}^{(c)} : i \in P_c\},
\{x_{j,t-\ell}^{(c)} : j=1,\ldots,m;\ \ell \in Q_{j,c}\}
\right].
```

The FFNN estimates a nonlinear mapping:

```math
\hat{y}_t^{(c)} = g_c(\mathbf{h}_t^{(c)}; \Theta_c),
```

where \(\Theta_c\) contains the network weights and biases. For an FFNN with \(L\) hidden layers:

```math
\mathbf{a}^{(0)} = \mathbf{h}_t^{(c)},
```

```math
\mathbf{a}^{(\ell)} =
\sigma\left(
\mathbf{W}^{(\ell)}\mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)}
\right),
\quad \ell = 1,\ldots,L,
```

```math
\hat{y}_t^{(c)} =
\mathbf{W}^{(L+1)}\mathbf{a}^{(L)} + b^{(L+1)}.
```

The activation function \(\sigma(\cdot)\) can be tanh or ReLU. In the implementation, the FFNN is trained with standardized inputs and standardized targets. The objective is regularized mean squared error:

```math
\min_{\Theta_c}
\frac{1}{N}
\sum_{t}
\left(y_t^{(c)} - \hat{y}_t^{(c)}\right)^2
+ \lambda \|\Theta_c\|_2^2.
```

The hyperparameters include hidden-layer architecture, regularization strength, random seed, activation function, learning rate, and maximum iterations.

### 3.6 Forecast Reconstruction

For the direct no-VMD baseline, the model outputs a single transformed forecast:

```math
\hat{y}_t = g(\mathbf{h}_t; \Theta).
```

For the VMD model, each component has its own prediction:

```math
\hat{y}_t^{(1)}, \hat{y}_t^{(2)}, \ldots, \hat{y}_t^{(K)}, \hat{y}_t^{(\text{RES})}.
```

The reconstructed forecast is:

```math
\hat{y}_t
=
\sum_{c \in \mathcal{C}} \hat{y}_t^{(c)}.
```

If the model operates on differenced or log-transformed scale, this reconstructed value is then mapped back to the original level scale.

### 3.7 Walk-Forward VMD and Leakage Control

The most important operational rule is that VMD must not be fitted once on the full dataset before forecasting. Full-sample decomposition would allow future observations to influence past component values. This would contaminate validation and test evaluation.

The proposed approach uses walk-forward VMD caching:

1. For each forecast origin \(t\), build the history available before \(t\).
2. Apply VMD only to that historical window.
3. Extract the lagged component values needed for \(\mathbf{h}_t^{(c)}\).
4. Predict each component at \(t\).
5. Reconstruct the aggregate forecast.
6. After the forecast is made, append the actual value at \(t\) to the history before moving to \(t+1\).

For component-level validation, actual component labels are also generated without future leakage. At origin \(t\), the decomposition used for the actual component includes the current actual row only for labeling that row, not for constructing predictors before the forecast is made.

### 3.8 Model Selection and Evaluation

The framework uses three chronological segments:

```text
training -> validation -> test
```

The training set is used for stationarity decisions, ARDL lag selection, VMD training decomposition, and FFNN fitting. The validation set is used for choosing lag specifications and hyperparameters. The test set is used only once, after the final configuration is selected.

The main evaluation metrics are:

```math
RMSE =
\sqrt{
\frac{1}{n}
\sum_{t=1}^{n}
(y_t - \hat{y}_t)^2
},
```

```math
MAE =
\frac{1}{n}
\sum_{t=1}^{n}
|y_t - \hat{y}_t|,
```

```math
MAPE =
\frac{100}{n}
\sum_{t=1}^{n}
\left|
\frac{y_t - \hat{y}_t}{y_t}
\right|,
```

and directional accuracy:

```math
DA =
\frac{100}{n-1}
\sum_{t=2}^{n}
\mathbf{1}
\left[
\operatorname{sign}(y_t-y_{t-1})
=
\operatorname{sign}(\hat{y}_t-\hat{y}_{t-1})
\right].
```

## 4. Algorithm

The full VMD-ARDL-FFNN procedure is:

```text
Input:
  target series y_t
  exogenous series X_t
  maximum target lag p_max
  maximum exogenous lag q_max
  number of VMD modes K
  FFNN hyperparameter grid

Step 1:
  Sort the data chronologically and split into train, validation, and test.

Step 2:
  Run train-only stationarity screening and apply the selected transformations.

Step 3:
  For the VMD model, decompose the training series into K modes plus residual.

Step 4:
  For each component:
    generate target lag candidates from ACF/PACF;
    select target lags using validation screening;
    select exogenous lags using ARDL information criteria;
    build forecast-safe lagged input matrices.

Step 5:
  Train FFNN candidates for each component.

Step 6:
  Evaluate component candidates on walk-forward validation cache.

Step 7:
  Select the component-model combination that minimizes reconstructed validation RMSE.

Step 8:
  Refit selected models using train plus validation data.

Step 9:
  Forecast the test period using walk-forward VMD cache.

Output:
  component forecasts
  reconstructed forecasts
  RMSE, MAE, MAPE, and directional accuracy
```

## 5. Empirical Experiment

The method is illustrated using a monthly dataset with 219 observations from January 2008 to March 2026. The target in this experiment is Vietnam's exports to the United States. The exogenous variables are U.S. personal consumption expenditures, U.S. retail sales, U.S. consumer sentiment, Vietnam's imports from China, and the USD/VND exchange rate.

This dataset is not the theoretical focus of the paper. Its role is to test whether the proposed forecasting architecture can be implemented on a realistic macroeconomic time series with trend, volatility, and external predictors.

After first differencing, the effective split contains 152 training observations, 33 validation observations, and 33 test observations. All variables are transformed based on train-only stationarity screening. The baseline model is a direct ARDL-FFNN without VMD. The hybrid model uses cached VMD decomposition with six modes plus a residual component.

## 6. Experimental Results

The selected direct ARDL-FFNN baseline uses target lag 1 and lagged exogenous inputs from U.S. retail sales, PCE, U.S. sentiment, imports from China, and USD/VND. The selected FFNN architecture uses tanh activation and a small hidden architecture, consistent with the limited sample size.

The selected VMD-ARDL-FFNN model trains separate FFNNs for the six VMD modes and residual component. Different components select different lag structures, showing that the decomposition layer separates dynamics with distinct temporal dependencies.

**Table 1. Level-Scale Forecast Performance in the Export Experiment**

| Split | Model | RMSE | MAE | MAPE | Directional Accuracy |
|---|---:|---:|---:|---:|---:|
| Validation | ARDL-FFNN | 1,068,381 | 732,262 | 9.45% | 62.50% |
| Validation | VMD-ARDL-FFNN | 1,139,667 | 844,307 | 10.81% | 65.63% |
| Test | ARDL-FFNN | 1,377,791 | 989,822 | 9.38% | 56.25% |
| Test | VMD-ARDL-FFNN | 1,290,032 | 925,696 | 9.12% | 71.88% |

The validation period favors the simpler ARDL-FFNN in error metrics, while VMD-ARDL-FFNN gives slightly better directional accuracy. On the test period, VMD-ARDL-FFNN improves RMSE by 6.37%, MAE by 6.48%, MAPE by 2.88%, and directional accuracy by 15.63 percentage points. This result supports the usefulness of the decomposition layer in the final out-of-sample period, but it also shows that VMD should be selected empirically rather than assumed to dominate.

## 7. Discussion

The proposed framework has three main methodological advantages. First, ARDL constrains the FFNN input space. This is important because neural networks can overfit quickly when many lagged variables are included without selection. ARDL lag selection gives a statistically motivated way to form compact lag matrices.

Second, FFNN adds nonlinear approximation capacity. A purely linear ARDL equation assumes additive linear effects of lagged variables. In many time series, the effect of one lag may depend on another lag, or the response may change across regimes. FFNN can learn such nonlinear interactions while still using interpretable lagged inputs.

Third, VMD can separate the forecasting task into component-level subproblems. Low-frequency components may capture trend-like behavior, while higher-frequency components may capture short-term fluctuations. Modeling these components separately can improve prediction when the aggregate series mixes dynamics with different frequencies.

The main risk is leakage. If decomposition is applied to the entire dataset before train-validation-test splitting, the resulting components are not valid forecasting inputs. The walk-forward cache used in this framework is therefore not a minor implementation detail; it is required for credible evaluation.

The empirical results show that VMD is beneficial on the test period but weaker on validation errors. This mixed result is methodologically useful because it prevents overclaiming. VMD-ARDL-FFNN should be viewed as a candidate architecture whose performance must be evaluated with clean chronological validation and final holdout testing.

## 8. Conclusion

This paper proposed a VMD-ARDL-FFNN framework for time-series forecasting. The method uses VMD for component decomposition, ARDL for forecast-safe lag selection, and FFNN for nonlinear prediction. The reconstructed forecast is the sum of component forecasts, transformed back to the original scale when needed.

The central methodological contribution is the integration of these three tools under a leakage-aware validation design. The export forecasting experiment demonstrates that the framework can improve test-set forecasting performance relative to a direct ARDL-FFNN baseline, especially in directional accuracy. However, validation results show that decomposition is not automatically superior. Future work should test the framework across multiple datasets, rolling-origin splits, VMD mode counts, and benchmark models.

## References

Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems, 2*, 303-314. https://doi.org/10.1007/BF02551274

Dragomiretskiy, K., & Zosso, D. (2014). Variational mode decomposition. *IEEE Transactions on Signal Processing, 62*(3), 531-544. https://doi.org/10.1109/TSP.2013.2288675

Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer feedforward networks are universal approximators. *Neural Networks, 2*(5), 359-366. https://doi.org/10.1016/0893-6080(89)90020-8

Huang, N. E., Shen, Z., Long, S. R., Wu, M. C., Shih, H. H., Zheng, Q., Yen, N. C., Tung, C. C., & Liu, H. H. (1998). The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis. *Proceedings of the Royal Society A, 454*(1971), 903-995. https://doi.org/10.1098/rspa.1998.0193

Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. https://otexts.com/fpp3/

Pesaran, M. H., & Shin, Y. (1999). An autoregressive distributed-lag modelling approach to cointegration analysis. In S. Strom (Ed.), *Econometrics and Economic Theory in the 20th Century: The Ragnar Frisch Centennial Symposium* (pp. 371-413). Cambridge University Press. https://doi.org/10.1017/CCOL521633230.011

Pesaran, M. H., Shin, Y., & Smith, R. J. (2001). Bounds testing approaches to the analysis of level relationships. *Journal of Applied Econometrics, 16*(3), 289-326. https://doi.org/10.1002/jae.616

Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature, 323*, 533-536. https://doi.org/10.1038/323533a0

Zhang, G. P. (2003). Time series forecasting using a hybrid ARIMA and neural network model. *Neurocomputing, 50*, 159-175. https://doi.org/10.1016/S0925-2312(01)00702-0
