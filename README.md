# Volta-Inflow-Prediction
The 2023 Akosombo spillage affected 36,000+ people. Developing an LSTM model to predict Lake Volta inflows 7-30 days ahead using rainfall and climate indices, comparing performance against traditional regression methods.


# LSTM-Based Prediction of Lake Volta Inflows

**A project to forecast Lake Volta's inflow 7-30 days in advance to mitigate flood risk, inspired by the 2023 Akosombo spillage.**

## Project Summary
This repository contains the code and analysis for developing and comparing machine learning models (MLR, ARIMA, LSTM) to predict hydrological inflows. The final LSTM model successfully outperformed baseline methods, demonstrating the effectiveness of neural networks for this task.

## Key Results
The final LSTM model achieved a superior performance on the test set compared to the baseline Multiple Linear Regression model.

| Model | RMSE (m³/s) | NSE |
|---|---|---|
| MLR | 132.81 | 0.73 |
| **LSTM** | **125.36** | **0.76** |

![Observed vs. Predicted Plot](figures/final_comparison_plot.png)


## How to Run This Project
1.  **Clone the repository:** `git clone [your-repo-url]`
2.  **Set up the environment:** `conda env create -f environment.yml` 
3.  **Run the notebooks:** Start with `01-Data-Cleaning.ipynb` and proceed in order.

## Project Structure
-   `/notebooks`: Contains the step-by-step Jupyter Notebooks for data processing and modeling.
-   `/data`: Contains the raw and processed data.
-   `dashboard.py`: A Streamlit app to deploy the final model.
