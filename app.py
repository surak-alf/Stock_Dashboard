import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.data_loader import load_data
from utils.preprocess import preprocess_data
from models.lstm_model import build_lstm_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

st.title('Stock Market Prediction Dashboard')

# Sidebar for user input
st.sidebar.header('User Input')
ticker = st.sidebar.selectbox('Select Ticker', ['baba', 'jnj', 'nvda', 'tm'])
seq_length = st.sidebar.slider('Sequence Length', 30, 120, 60)
epochs = st.sidebar.slider('Epochs', 10, 100, 50)
batch_size = st.sidebar.slider('Batch Size', 16, 128, 32)

# Load data
data = load_data(ticker)
st.subheader(f'{ticker.upper()} Processed Data')
st.write(data)

# Preprocess data
X, y, scaler = preprocess_data(data, seq_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build and train the LSTM model
model = build_lstm_model((X_train.shape[1], 1))
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                    epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=1)

# Plot training & validation loss values
st.subheader('Model Loss')
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
st.pyplot(plt)

# Predict on test data
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the results
st.subheader('Predictions vs Actual')
plt.figure(figsize=(10, 6))
plt.plot(y_test_original, color='blue', label='Actual Stock Price')
plt.plot(predictions, color='red', label='Predicted Stock Price')
plt.title(f'{ticker.upper()} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(plt)