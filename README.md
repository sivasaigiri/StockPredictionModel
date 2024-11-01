# Google Stock Price Prediction using LSTM

This project is a deep learning-based approach to predict the future stock prices of Google using a Long Short-Term Memory (LSTM) model. LSTM networks are a type of recurrent neural network (RNN) that are particularly effective for time series forecasting. This project uses historical stock data to train an LSTM model to forecast future closing prices.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [License](#license)

## Introduction

Stock price prediction is a challenging task due to the complex and dynamic nature of the stock market. Using machine learning and deep learning methods, we can attempt to analyze past stock prices to forecast future trends. This project aims to leverage LSTM to model the temporal dependencies in stock price movements and make predictions for Google’s stock.

## Dataset

The dataset used for training and testing the model includes Google stock prices. It consists of the following columns:
- **Date**: The date of the stock price
- **Open**: The opening price on that date
- **High**: The highest price on that date
- **Low**: The lowest price on that date
- **Close**: The closing price on that date
- **Volume**: The trading volume on that date

Data can be obtained from [Yahoo Finance](https://finance.yahoo.com/) or other financial data providers.

## Model Architecture

The model is built using a single-layer LSTM followed by dense layers. LSTM is suitable for time series analysis as it can capture long-term dependencies in sequential data.

### Architecture:
1. **LSTM Layer**: To process the sequential input data and capture time dependencies.
2. **Dropout Layer**: To prevent overfitting by randomly dropping units during training.
3. **Dense Layer**: Fully connected layer to output the final prediction.

### Hyperparameters:
- **Sequence Length**: Number of days used as input for predicting the next day’s price.
- **Batch Size**: Size of the data batches.
- **Epochs**: Number of training cycles.
- **Learning Rate**: Rate at which the model updates parameters during training.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Google-Stock-Prediction-LSTM.git
    cd Google-Stock-Prediction-LSTM
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Preprocess the data**:
   - Download and clean the data.
   - Normalize the values for efficient training.

2. **Train the model**:
   ```python
   python train.py
