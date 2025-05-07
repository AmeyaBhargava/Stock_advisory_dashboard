import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional, Conv1D, MaxPooling1D, Attention, RepeatVector, Flatten
import tensorflow as tf

# --- Constants ---
WINDOW = 1  # Adjust window size to 1 for your datasets
EPOCHS = 10
BATCH_SIZE = 16

# --- Load Historical Data from CSV ---
def load_data_from_csv(path, interval):
    df = pd.read_csv(path)
    
    # Strip spaces in column names
    df.columns = df.columns.str.strip()
    
    # Parse 'Date' as datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Sort by Date and set as index
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    
    # Ensure the dataset has enough rows for the window size
    if len(df) < WINDOW:
        raise ValueError(f"Dataset for {interval} interval has fewer than {WINDOW} rows. Please provide more data.")
    
    print(f"Loaded data for {interval} with {len(df)} rows.")
    
    return df[['Close']].astype(float)

# --- Preprocess for LSTM ---
def prepare_data(df, window):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i, 0])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y), scaler

# --- LSTM Variants ---
def vanilla_lstm(input_shape):
    model = Sequential([
        LSTM(50, input_shape=input_shape),
        Dense(1)
    ])
    return model

def stacked_lstm(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    return model

def bidirectional_lstm(input_shape):
    model = Sequential([
        Bidirectional(LSTM(50), input_shape=input_shape),
        Dense(1)
    ])
    return model

def cnn_lstm(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        LSTM(50),
        Dense(1)
    ])
    return model

def attention_lstm(input_shape):
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(50, return_sequences=True)(inputs)
    attention = Attention()([lstm_out, lstm_out])
    flat = Flatten()(attention)
    output = Dense(1)(flat)
    return Model(inputs, output)

def seq2seq_lstm(input_shape):
    inputs = Input(shape=input_shape)
    encoded = LSTM(100)(inputs)
    repeated = RepeatVector(1)(encoded)
    decoded = LSTM(50, return_sequences=False)(repeated)
    output = Dense(1)(decoded)
    return Model(inputs, output)

# --- Training + Prediction ---
def train_model(model_fn, X_train, y_train, X_test, scaler):
    model = model_fn((X_train.shape[1], 1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    
    pred_scaled = model.predict(X_test)
    pred = scaler.inverse_transform(pred_scaled)
    return pred[0][0], model

# --- Option Pricing using Black-Scholes Formula ---
def black_scholes(S, K, T, r, sigma, option_type='call'):
    from scipy.stats import norm
    d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        option_price = (S * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == 'put':
        option_price = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))
    return option_price

# --- Ensemble of Top 3 Models ---
def ensemble_prediction(predictions):
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    best_3 = sorted_predictions[:3]
    ensemble_avg = np.mean([x[1] for x in best_3])
    return ensemble_avg, best_3

# --- Plotting Prediction Lines ---
def plot_predictions(df, predictions, interval):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Close'], label="Actual Prices", color='blue')
    
    for name, pred in predictions.items():
        plt.plot(df.index[-len(pred):], pred, label=f"Predicted {name}")
    
    plt.title(f"Stock Price Predictions for {interval} Interval")
    plt.xlabel("Date")
    plt.ylabel("Price (₹)")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Master Run Function ---
def run_all_models_with_csv(csv_path, interval):
    df = load_data_from_csv(csv_path, interval)
    
    # Check if there's enough data to train
    if len(df) <= 1:
        print(f"Not enough data for training in {interval} interval.")
        return

    X, y, scaler = prepare_data(df, WINDOW)
    
    # Handle reshaping for single row (WINDOW=1)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    X_train, y_train = X[:-1], y[:-1]
    X_test = X[-1:].reshape(1, WINDOW, 1)

    models = {
        "Vanilla LSTM": vanilla_lstm,
        "Stacked LSTM": stacked_lstm,
        "Bidirectional LSTM": bidirectional_lstm,
        "CNN-LSTM": cnn_lstm,
        "Attention LSTM": attention_lstm,
        "Seq2Seq LSTM": seq2seq_lstm
    }

    predictions = {}
    for name, model_fn in models.items():
        print(f"Training {name}...")
        try:
            pred, _ = train_model(model_fn, X_train, y_train, X_test, scaler)
            predictions[name] = pred
        except Exception as e:
            print(f"{name} failed: {e}")

    actual_price = df['Close'].iloc[-1]

    print(f"\n🔍 Actual Close Price for {interval}: ₹{actual_price:.2f}\n")
    print("📊 Model Performance Benchmark:\n")
    print("{:<25} {:<15} {:<15} {:<15} {:<10}".format("Model", "Predicted", "Actual", "Abs Error", "MAPE (%)"))
    print("-" * 80)

    for name, pred in predictions.items():
        error = abs(actual_price - pred)
        mape = (error / actual_price) * 100
        print(f"{name:<25} ₹{pred:<14.2f} ₹{actual_price:<14.2f} ₹{error:<14.2f} {mape:<10.2f}")

    # Ensemble model prediction
    ensemble_avg, best_3 = ensemble_prediction(predictions)
    print(f"\nEnsemble of Best 3 Models:\n")
    print(f"Predicted Price (Ensemble): ₹{ensemble_avg:.2f}")
    print(f"Best 3 Models: {', '.join([x[0] for x in best_3])}")

    # Plot predictions
    plot_predictions(df, predictions, interval)

# --- Run ---
if __name__ == "__main__":
    intervals = ["daily", "weekly", "monthly"]
    for interval in intervals:
        csv_path = f"nifty50_{interval}.csv"  # Modify to actual file paths
        run_all_models_with_csv(csv_path, interval)