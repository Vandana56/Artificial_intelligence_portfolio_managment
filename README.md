# AI Portfolio Management System
https://artificialintelligenceportfoliomanagment-e96eglscqrexymihbkvnt.streamlit.app
An interactive web application that helps investors analyse a stock portfolio, understand its risk, forecast short-term price direction using machine learning, and find a better allocation of their money.

Built with **Python** and **Streamlit**, with a clean separation between the frontend (`app.py`) and the analysis logic (`backend/`).


# Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Key Metrics Explained](#key-metrics-explained)
- [Future Improvements](#future-improvements)
- [Disclaimer](#disclaimer)
- [Author](#author)

---

## Overview

Managing a portfolio means answering three questions: *How is it performing? How risky is it? What should I do next?*

This project answers all three in one dashboard. The user selects stocks and a time period, and the app loads historical market data, calculates performance and risk measures, trains a machine learning model to predict the next 5 days of price direction, and suggests an optimal allocation across the selected stocks.



## Features

The app is organised into four tabs:

| Tab | What it does |
|-----|--------------|
| **Overview** | Shows portfolio summary, price history, and returns of the selected stocks, along with the data source and number of trading days analysed. |
| **Risk** | Calculates Sharpe ratio, Value at Risk (VaR), and maximum drawdown to show how risky the portfolio is. |
| **AI Forecast** | Uses an XGBoost model to predict the 5-day price direction (up or down) for each stock. |
| **Optimizer** | Suggests the best allocation (weights) across stocks to balance return and risk. |

Other highlights:

- Interactive charts and clean, easy-to-read layout
- Number inputs and controls for customising the analysis
- Modular backend that is easy to extend with new models or metrics



## How It Works

1. **Data loading**: `data_loader.py` fetches historical price data for the chosen stocks and prepares it (cleaning, returns calculation).
2. **Risk analysis**: `risk.py` computes risk metrics from the daily returns.
3. **Forecasting**: `models.py` builds features from price history and trains an XGBoost classifier to predict the direction of the next 5 days.
4. **Optimization**: `optimizer.py` finds portfolio weights that give the best balance between expected return and risk.
5. **Service layer**: `service.py` exposes a single `run_analysis` function that ties everything together and returns the results to the app.
6. **Frontend**: `app.py` takes user input, calls the backend, and displays results in the four tabs.


## Tech Stack

- **Language:** Python 3.12
- **Frontend:** Streamlit
- **Machine Learning:** XGBoost, Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Visualization:** Altair, Matplotlib



## Project Structure

 | Path | Description |
|-------|-------------|
`ai_portfolio/` | Main project folder |
`app.py` | Streamlit frontend (UI and the four tabs) |
 `backend/`  | All analysis logic |
 `backend/__init__.py` | Makes `backend` a Python package |
`backend/data_loader.py`  | Fetches and prepares stock data |
 `backend/models.py`  | Machine learning forecasting model (XGBoost) |
 `backend/risk.py`  | Risk metrics (Sharpe, VaR, drawdown) |
 `backend/optimizer.py` | Portfolio allocation optimizer |
 `backend/service.py`| `run_analysis` connects backend to frontend |
 `requirements.txt`  | Python dependencies |
`.gitignore`  | Files Git should ignore (for example `venv/`) |
| `README.md` | Project documentation |

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/Vandana56/Artificial_intelligence_portfolio_managment.git
cd Artificial_intelligence_portfolio_managment
```

**2. Create a virtual environment**

```bash
python -m venv venv
```

**3. Activate it**

- Windows (PowerShell):
  ```bash
  venv\Scripts\activate
  ```
- macOS / Linux:
  ```bash
  source venv/bin/activate
  ```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Usage

Start the app:

```bash
python -m streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

1. Choose the stocks and time period.
2. Run the analysis.
3. Explore the **Overview**, **Risk**, **AI Forecast**, and **Optimizer** tabs.


## Key Metrics Explained

- **Sharpe Ratio:** return earned per unit of risk. Higher is better.
- **Value at Risk (VaR):** the maximum loss expected over a period at a given confidence level (for example, 95%).
- **Maximum Drawdown:** the largest drop from a peak to a low point, showing the worst loss an investor would have experienced.
- **5-Day Forecast:** the model's prediction of whether a stock's price will move up or down over the next 5 trading days.


## Future Improvements

- Add more ML models and compare their accuracy
- Support for more asset types (ETFs, crypto)
- Backtesting of the optimized portfolio
- Downloadable reports (PDF/CSV)
- Deployment on Streamlit Community Cloud


## Disclaimer

This project is for **educational purposes only**. Predictions and allocations are not financial advice. Always do your own research before investing.



**Vandana**
GitHub: [@Vandana56](https://github.com/Vandana56)
