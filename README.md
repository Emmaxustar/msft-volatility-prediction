# MSFT Volatility Prediction

This project predicts Microsoft (MSFT) stock volatility using historical market data from Yahoo Finance.

## Data

The data is downloaded using `yfinance` for the following tickers:

- MSFT
- SPY
- QQQ
- XLK
- ^VIX

## Features

The project uses MSFT returns, lagged returns, rolling volatility, market index returns, VIX features, sector ETF features, and relative performance features.

## Targets

The project predicts three volatility horizons:

- Daily volatility: next-day absolute return
- Weekly volatility: next 5-trading-day realized volatility
- Monthly volatility: next 21-trading-day realized volatility

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
