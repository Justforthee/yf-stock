import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from data.yahoo_finance import get_stock_data, get_real_time_price
from preprocessing.feature_engineering import engineer_features, prepare_data
from models.short_term_model import ShortTermModel
from models.long_term_model import LongTermModel
import config
import os
import joblib

app = dash.Dash(__name__)

# Initialize models and feature columns
short_term_model = None
long_term_model = None
feature_columns = None

# Try to load pre-trained models and feature columns
models_loaded = False

if os.path.exists('short_term_model.joblib') and os.path.exists('long_term_model.joblib') and os.path.exists('feature_columns.joblib'):
    try:
        short_term_model = ShortTermModel.load('short_term_model.joblib')
        long_term_model = LongTermModel.load('long_term_model.joblib')
        feature_columns = joblib.load('feature_columns.joblib')
        models_loaded = True
        print("Pre-trained models and feature columns loaded successfully.")
    except Exception as e:
        print(f"Error loading pre-trained models or feature columns: {str(e)}")
        print("Please run main.py to train new models.")
else:
    print("Pre-trained models or feature columns not found. Please run main.py to train the models first.")

app.layout = html.Div([
    html.H1("Stock Price Prediction Dashboard"),
    dcc.Input(id="ticker-input", type="text", placeholder="Enter stock ticker"),
    html.Button("Submit", id="submit-button", n_clicks=0),
    dcc.Graph(id="stock-chart"),
    html.Div(id="prediction-output")
])

app.clientside_callback(
    """
    function(n_clicks, value) {
        if (n_clicks > 0 || value) {
            return n_clicks + 1;
        }
        return dash_clientside.no_update;
    }
    """,
    Output("submit-button", "n_clicks"),
    Input("submit-button", "n_clicks"),
    Input("ticker-input", "n_submit")
)

@app.callback(
    [Output("stock-chart", "figure"),
     Output("prediction-output", "children")],
    [Input("submit-button", "n_clicks")],
    [State("ticker-input", "value")],
    prevent_initial_call=True
)
def update_chart(n_clicks, ticker):
    empty_fig = go.Figure()
    empty_fig.update_layout(title="No data to display")

    if not ticker:
        return empty_fig, "Please enter a stock ticker."
    
    if not models_loaded:
        return empty_fig, "Models not loaded. Please run main.py to train the models first."
    
    try:
        # Fetch stock data
        df = get_stock_data(ticker)
        
        if df.empty:
            return empty_fig, f"No data available for ticker {ticker}"
        
        # Engineer features
        df_features = engineer_features(df)
        
        # Prepare features for prediction
        X_short, _ = prepare_data(df_features.iloc[-1:], 'Target_Short', is_classification=True, feature_columns=feature_columns)
        X_long, _ = prepare_data(df_features.iloc[-1:], 'Target_Long', is_classification=False, feature_columns=feature_columns)
        
        # Make predictions
        short_term_pred = short_term_model.predict(X_short)
        long_term_pred = long_term_model.predict(X_long)
        
        # Create chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'],
                                     name="Stock Price"))
        
        # Add long-term prediction to the chart
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30)
        future_prices = df['Close'].iloc[-1] * (1 + long_term_pred[0] * np.arange(1, 31))
        fig.add_trace(go.Scatter(x=future_dates, y=future_prices, mode='lines', name='30-day Forecast'))
        
        fig.update_layout(title=f"{ticker} Stock Price and Forecast",
                          xaxis_title="Date",
                          yaxis_title="Price",
                          height=600)
        
        # Add link to Yahoo Finance
        yahoo_link = f"https://finance.yahoo.com/quote/{ticker}"
        
        # Prepare prediction output
        current_price = df['Close'].iloc[-1]
        short_term_direction = "Up" if short_term_pred[0] == 1 else "Down"
        
        # Use short-term prediction for next day's price estimate
        next_day_change = 0.01 if short_term_pred[0] == 1 else -0.01  # Assuming 1% change
        next_day_pred = current_price * (1 + next_day_change)
        
        long_term_end_price = future_prices[-1]
        
        prediction_text = [
            html.P(f"Current Price: ${current_price:.2f}"),
            html.P(f"Short-term prediction (next day): Price likely to go {short_term_direction}"),
            html.P(f"Estimated next day price: ${next_day_pred:.2f}"),
            html.P(f"Long-term prediction (after 30 days): ${long_term_end_price:.2f}"),
            html.A("View on Yahoo Finance", href=yahoo_link, target="_blank")
        ]
        
        return fig, prediction_text
    except ValueError as e:
        # Return an empty figure and a specific error message for ValueError
        error_message = f"An error occurred: {str(e)}"
        return empty_fig, html.P(error_message, style={'color': 'red'})
    except Exception as e:
        # Return an empty figure and a generic error message
        error_message = f"An unexpected error occurred: {str(e)}"
        return empty_fig, html.P(error_message, style={'color': 'red'})
    
if __name__ == '__main__':
    app.run_server(debug=True)