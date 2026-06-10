# Forecasting Vietnam's Exports to the United States with ARDL-Guided Feedforward Neural Networks and Variational Mode Decomposition

## Abstract

Accurate export forecasting is important for trade planning, inventory decisions, and macroeconomic monitoring, especially when external demand, exchange rates, and import inputs move unevenly over time. This study develops and evaluates a hybrid forecasting framework for monthly Vietnamese exports to the United States using autoregressive distributed lag (ARDL) lag selection and feedforward neural networks (FFNNs). The baseline model uses ARDL-selected target and exogenous lags as inputs to an FFNN. A second model extends this framework with Variational Mode Decomposition (VMD), where the target and predictors are decomposed into components, each component is forecast separately, and component forecasts are reconstructed into the final export forecast. The dataset contains 219 monthly observations from January 2008 to March 2026. Export_US is forecast using U.S. retail sales, personal consumption expenditures, U.S. consumer sentiment, Vietnam's imports from China, and the USD/VND exchange rate. To reduce look-ahead bias, the experiment uses chronological train-validation-test splits, train-only stationarity screening, forecast-safe exogenous lags, validation-based model selection, and final test evaluation only after model selection. On the level-scale test set, the cached VMD-ARDL-FFNN reduces RMSE from 1.378 million to 1.290 million, reduces MAE from 0.990 million to 0.926 million, lowers MAPE from 9.38% to 9.12%, and improves directional accuracy from 56.25% to 71.88%. However, VMD performs worse on level-scale validation errors, indicating that decomposition can improve out-of-sample directional performance but does not uniformly dominate the simpler ARDL-FFNN baseline. The findings support the value of combining econometric lag discipline with nonlinear learning while highlighting the need for leakage-aware decomposition in time-series forecasting.

**Keywords:** export forecasting; ARDL; feedforward neural network; Variational Mode Decomposition; Vietnam; time series; hybrid model

## 1. Introduction

Export forecasting is a practical problem for both firms and policymakers. For export-oriented economies, future shipment values affect production schedules, inventory decisions, foreign-exchange planning, and assessments of external demand. Vietnam's exports to the United States are especially relevant because the United States is one of Vietnam's major destination markets, while export performance is influenced by demand-side conditions, exchange-rate movements, and supply-chain inputs. Monthly export data are also difficult to forecast because they contain trend, seasonality, shocks, nonlinear adjustment, and structural breaks.

Traditional econometric models offer interpretability and disciplined lag selection. The autoregressive distributed lag (ARDL) framework is useful because it models the current target using its own lags and lagged explanatory variables. This structure fits export forecasting because trade flows respond to both past export momentum and delayed changes in retail demand, consumption, consumer sentiment, exchange rates, and import inputs. However, linear ARDL models may be too restrictive when the relationships among exports and macroeconomic indicators are nonlinear or regime-dependent.

Artificial neural networks provide a complementary strength. Feedforward neural networks (FFNNs) can approximate nonlinear functions and interactions among lagged variables without imposing a fixed linear response. At the same time, neural networks can overfit when the input space is poorly controlled, especially in small monthly datasets. A practical solution is to combine ARDL lag selection with FFNN estimation: ARDL is used to identify forecast-relevant lag structures, and the FFNN learns nonlinear relationships from those lagged inputs.

Recent hybrid forecasting approaches also use signal decomposition before prediction. Variational Mode Decomposition (VMD) separates a time series into smoother intrinsic modes and a residual component. In principle, decomposition can simplify the learning problem by allowing separate models to capture low-frequency trend, medium-frequency cycles, and high-frequency noise. In practice, decomposition can easily create look-ahead bias if validation or test periods are decomposed together with future observations. Therefore, a leakage-aware VMD design is necessary for credible forecasting evaluation.

This study addresses the following research objective: to evaluate whether a leakage-aware VMD-ARDL-FFNN model improves monthly forecasts of Vietnam's exports to the United States compared with a direct ARDL-FFNN baseline. The paper contributes a reproducible forecasting pipeline that combines econometric lag selection, nonlinear FFNN prediction, train-only stationarity decisions, chronological validation, and walk-forward VMD caching. The main empirical question is not only whether VMD improves forecast errors, but also whether the improvement is stable across validation and test periods.

## 2. Materials and Methods

### 2.1 Data and Variables

The empirical dataset contains 219 monthly observations from January 2008 to March 2026. The target variable is `Export_US`, representing Vietnam's exports to the United States. Five explanatory variables are used: U.S. retail sales (`US_Retail`), U.S. personal consumption expenditures (`PCE`), U.S. consumer sentiment (`US_Sentiment`), Vietnam's imports from China (`Import_CN`), and the USD/VND exchange rate (`USD_VND`). These variables were selected to represent U.S. demand conditions, consumer expectations, production-input links, and exchange-rate conditions.

The original dataset also includes `EU_RETAIL`, `Export_EU`, and `EUR_VND`, but the reported experiment focuses on exports to the United States and therefore uses U.S.-related demand indicators plus import and exchange-rate variables. The dataset is sorted by `TIME_PERIOD` and converted to monthly frequency. All model decisions are made using chronological splits rather than random shuffling.

**Table 1. Dataset Overview**

| Item | Description |
|---|---|
| Frequency | Monthly |
| Full sample | January 2008 to March 2026 |
| Observations | 219 |
| Target | Vietnam exports to the United States (`Export_US`) |
| Predictors | `PCE`, `US_Retail`, `US_Sentiment`, `Import_CN`, `USD_VND` |
| Mean Export_US | 4.777 million |
| Minimum Export_US | 0.582 million |
| Maximum Export_US | 15.191 million |

Because all selected level variables are positive, the pipeline supports log transformation. The reported results are evaluated on both the transformed model scale and the reconstructed level scale. The main interpretation uses the level scale because it is expressed in the original export units.

### 2.2 Chronological Splitting and Stationarity Treatment

The data are split in time order into training, validation, and test sets. After first-difference transformation, the effective chronological split is 152 observations for training, 33 for validation, and 33 for testing. The training period runs from February 2008 to September 2020, the validation period from October 2020 to June 2023, and the test period from July 2023 to March 2026.

Stationarity screening is performed only on the training data using Augmented Dickey-Fuller (ADF) and KPSS tests. This prevents the transformation decision from using validation or test information. All selected variables are treated with first differences in the reported pipeline. For the USD/VND exchange rate, the level tests are mixed and the first-difference KPSS result remains weaker than the other variables, so the pipeline marks it as a likely higher-integration-risk variable while still applying first differences for consistency.

**Table 2. Stationarity Screening Summary**

| Variable | Level decision | Diff1 decision | Transform used |
|---|---|---|---|
| Export_US | Non-stationary | Stationary | diff1 |
| US_Retail | Non-stationary | Stationary | diff1 |
| PCE | Non-stationary | Stationary | diff1 |
| US_Sentiment | Non-stationary | Stationary | diff1 |
| Import_CN | Non-stationary | Stationary | diff1 |
| USD_VND | Mixed/unclear | Mixed/unclear | diff1 |

### 2.3 Baseline ARDL-FFNN Model

The baseline model forecasts the transformed export series directly. First, candidate autoregressive lags are generated from ACF/PACF information on the training data. Candidate target-lag sets are screened on the validation period. Exogenous lags are selected using ARDL information criteria over forecast-safe lag ranges. Exogenous lag 0 is excluded because same-period explanatory variables would not be available in a true forecast setting.

For each selected lag specification, a supervised learning matrix is built. The inputs include lagged target values and lagged exogenous variables, and the output is the current transformed target value. The FFNN is implemented with a multilayer perceptron regressor. Inputs and outputs are standardized inside the model wrapper, predictions are inverse-transformed back to model scale, and final results are reconstructed to level scale.

The FFNN search uses validation performance to choose among lag specifications, hidden-layer architectures, regularization values, random seeds, and activation functions. The final model is selected by validation RMSE, with MAE and MAPE used as secondary criteria. After model selection, the winning configuration is refit on training plus validation data and evaluated on the held-out test period.

### 2.4 VMD-ARDL-FFNN Model

The VMD-ARDL-FFNN model extends the baseline by decomposing the transformed target and predictors into VMD components. Each component is modeled using the same ARDL-guided FFNN strategy. Component forecasts are then summed to reconstruct the final target forecast.

The key methodological issue is avoiding leakage. If VMD were applied once to the whole series, validation and test decompositions would use future observations, making the forecast evaluation overly optimistic. The project avoids this by using a cached walk-forward origin design. At each validation or test timestamp, VMD is applied only to the historical information available before that forecast origin. The actual current row is added only after the forecast origin has been processed. Component-level actual values are also generated in a walk-forward manner so that component candidates are scored against the correct component target rather than against the aggregate target.

The reported cached VMD output contains seven reconstructed components, corresponding to six VMD modes plus a residual component. For each component, ARDL selects forecast-safe lags and an FFNN is trained. Component candidates are first evaluated at the component level, then combinations of component candidates are evaluated on reconstructed validation forecasts. The selected component models are refit before final test evaluation.

### 2.5 Evaluation Metrics

Forecast accuracy is evaluated using RMSE, MAE, MAPE, and directional accuracy. RMSE penalizes larger errors more heavily, MAE reports average absolute error, MAPE expresses average proportional error, and directional accuracy measures whether the predicted change has the same sign as the actual change. The level-scale metrics are emphasized because they are directly interpretable in export-value units.

## 3. Results

### 3.1 Data Characteristics and Diagnostics

Export_US grows strongly across the sample, from values below 1 million in early 2008 to several months above 10 million in the 2023-2026 test period. The target is volatile, with a mean of 4.777 million and a standard deviation of 3.688 million. This scale shift motivates using transformations and chronological evaluation rather than random cross-validation.

The Granger-causality diagnostic on the training set indicates that U.S. retail sales has statistically significant predictive content for Export_US at lag 2 (F = 3.445, p = 0.0345). PCE, U.S. sentiment, imports from China, and USD/VND are not significant at the 5% level in this diagnostic. These nonsignificant results are still retained in the nonlinear forecasting framework because the FFNN may use them through nonlinear interactions and because ARDL lag selection can down-weight weak lag structures.

### 3.2 Selected Model Structures

The selected no-VMD baseline uses target lag 1 and a set of lagged exogenous inputs: U.S. retail sales lag 1, PCE lag 1, U.S. sentiment lags 1-2, imports from China lags 1-2, and USD/VND lags 1-4. Its selected FFNN architecture uses two hidden layers of three neurons each, tanh activation, regularization alpha of 0.01, and seed 7.

The selected cached VMD model contains separate FFNNs for `VMD1` through `VMD6` and the residual component. The component models generally select short autoregressive lags such as lag 1, with several components also using lag 2, lag 4, or lag 6. This indicates that decomposed export dynamics contain both short-run persistence and medium-frequency lag effects. All selected component models in the available best-model file use tanh activation, while hidden-layer sizes vary by component.

### 3.3 Forecast Accuracy

Table 3 compares the baseline ARDL-FFNN and cached VMD-ARDL-FFNN on the reconstructed level scale. On the validation period, the no-VMD baseline has lower RMSE, MAE, and MAPE. The cached VMD model has slightly higher directional accuracy. On the test period, the cached VMD model improves all reported error metrics and substantially improves directional accuracy.

**Table 3. Level-Scale Forecast Performance**

| Split | Model | RMSE | MAE | MAPE | Directional accuracy |
|---|---:|---:|---:|---:|---:|
| Validation | ARDL-FFNN | 1,068,381 | 732,262 | 9.45% | 62.50% |
| Validation | Cached VMD-ARDL-FFNN | 1,139,667 | 844,307 | 10.81% | 65.63% |
| Test | ARDL-FFNN | 1,377,791 | 989,822 | 9.38% | 56.25% |
| Test | Cached VMD-ARDL-FFNN | 1,290,032 | 925,696 | 9.12% | 71.88% |

Relative to the baseline, cached VMD worsens validation RMSE by 6.67% and validation MAE by 15.30%. On the held-out test set, however, cached VMD improves RMSE by 6.37%, MAE by 6.48%, and MAPE by 2.88%. The largest gain is directional accuracy, which increases by 15.63 percentage points on the test period.

These results suggest that decomposition is not uniformly beneficial, but it can improve robustness in the final out-of-sample window. The difference between validation and test results also suggests that the validation period may contain dynamics that favor a simpler direct model, while the later test period benefits from component-wise modeling.

## 4. Discussion

The empirical findings support three main interpretations. First, ARDL-guided lag selection is useful as a discipline for neural-network forecasting. Rather than feeding a large and arbitrary lag matrix into the FFNN, the pipeline uses ACF/PACF target-lag screening and ARDL information criteria to construct forecast-safe lag specifications. This is especially important in a dataset with only monthly observations, where overfitting risk is high.

Second, the VMD extension provides mixed but promising evidence. Its validation errors are worse than those of the direct ARDL-FFNN baseline, but its test errors and directional accuracy are better. This pattern argues against a simplistic claim that decomposition always improves forecasting. Instead, VMD appears useful when the decomposed components capture signal structures that generalize to the test period. Because component forecasts are summed, errors can also accumulate across components, especially when the number of modes is high or individual component signals are weak.

Third, the leakage-aware design is central to the credibility of the comparison. Decomposition methods can accidentally use future information if validation or test observations are included in the decomposition window before forecasting. The cached origin design used here is more conservative: each forecast origin only sees available history. This makes the reported VMD results more realistic than a full-sample decomposition evaluation, although it also makes the VMD pipeline computationally more expensive.

The diagnostic results also provide economic context. U.S. retail sales is the only predictor with significant Granger-causal evidence at the 5% level in the training set, which is consistent with the idea that U.S. demand conditions are directly linked to Vietnamese exports to the United States. Other variables are not statistically significant in the linear Granger test, but they may still contribute through delayed, nonlinear, or interaction effects in the FFNN. The exchange-rate variable is diagnostically more difficult because its stationarity results are mixed, which may weaken its reliability as a direct predictor.

This study has several limitations. The available dataset contains only 219 monthly observations, and the effective training sample after transformation is smaller. Neural networks can therefore be sensitive to architecture, random seed, and validation-window composition. The metrics files in the project also contain appended results from multiple runs, so the paper emphasizes the exported comparison table and best-model files rather than treating every appended row as an independent experiment. Finally, the study evaluates one target market and one selected feature set; results may differ for exports to other regions or for alternative macroeconomic predictors.

Future work should test multiple VMD mode counts in separate clean output folders, select the number of modes only by reconstructed validation performance, and then evaluate a single locked configuration on the test set. It would also be useful to benchmark against classical ARDL, ARIMA/SARIMA, tree-based models, LSTM/GRU models, and simple seasonal naive forecasts. A rolling-origin backtest with several validation-test windows would provide stronger evidence about whether the VMD advantage is stable or specific to the July 2023-March 2026 test period.

## 5. Conclusion

This paper evaluated a hybrid forecasting framework for Vietnam's monthly exports to the United States. The baseline ARDL-FFNN model uses econometric lag selection to define forecast-safe neural-network inputs, while the VMD-ARDL-FFNN model adds leakage-aware decomposition and component-wise forecasting. The direct ARDL-FFNN baseline performs better on validation errors, but the cached VMD-ARDL-FFNN performs better on the final test period, reducing level-scale RMSE, MAE, and MAPE and improving directional accuracy from 56.25% to 71.88%.

The main conclusion is that VMD can improve out-of-sample export forecasting when it is implemented with walk-forward decomposition, but it should be treated as an empirically tested modeling choice rather than an automatic improvement. The strongest methodological contribution of the project is the combination of ARDL lag discipline, nonlinear FFNN estimation, chronological model selection, and leakage-aware VMD evaluation.

## References

Dragomiretskiy, K., & Zosso, D. (2014). Variational mode decomposition. *IEEE Transactions on Signal Processing, 62*(3), 531-544.

Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.

Pesaran, M. H., Shin, Y., & Smith, R. J. (2001). Bounds testing approaches to the analysis of level relationships. *Journal of Applied Econometrics, 16*(3), 289-326.

Zhang, G. P. (2003). Time series forecasting using a hybrid ARIMA and neural network model. *Neurocomputing, 50*, 159-175.

Busse, C., & August, E. (2021). How to write and publish a research paper for a peer-reviewed journal. *Journal of Cancer Education, 36*, 909-913.
