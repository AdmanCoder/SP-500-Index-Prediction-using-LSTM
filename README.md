Hey!
This is the repository for my ECO 723 course project. 
The goal of this project was to replicate and extend a 2022 academic paper on forecasting the S&P 500.
Instead of just throwing a standard neural network at raw price data, I pulled in 10 different macroeconomic and technical indicators (like the VIX, interest rates, and unemployment) and used a Hidden Markov Model (HMM) to classify broader market regimes to give the deep learning model actual economic context.


Project Summary
Here is the high-level breakdown of what this project accomplished:

Objective:
To optimise S&P 500 forecasts using macro & technical features to improve reliability in volatile markets.

Approach:
Analyzed 5k+ trading days across 10 macro and technical indicators, integrating HMM-classified regimes.
Deployed SHAP analysis for interpretability, identifying VIX as the most important feature for forecasting.

Impact:
Mitigated overfitting via a Single-layer LSTM, outperforming multi-layer models by 18.8% in RMSE.

Repository Structure
 `raw_data.csv`: The core 20-year daily dataset containing the S&P 500 prices and all our macroeconomic variables.
 `replicate_lstm_stock.py`: The baseline deep learning script that benchmarks the Single-layer vs Multi-layer LSTM architectures.
 `replicate_hmm_shap.py`: The advanced extension script that calculates the HMM market regimes and runs the SHAP model interpretability.

 
