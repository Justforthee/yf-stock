import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from data.yahoo_finance import get_stock_data, get_real_time_price
from preprocessing.feature_engineering import engineer_features
from models.short_term_model import ShortTermModel
from models.long_term_model import LongTermModel
import config
import os

app = dash.Dash(__name__)

# Initialize models
short_term_model = ShortTermModel()
long_term_model = LongTermModel()

# Try to load pre-trained models
models_loaded = False
if os.path.exists('short_term_model.joblib') and os.path.exists('long_term_model.joblib'):
    try:
        short_term_model.load('short_term_model.joblib')
        long_term_model.load('long_term_model.joblib')
        models_loaded = True
        print("Pre-trained models loaded successfully.")
    except Exception as e:
        print(f"Error loading pre-trained models: {str(e)}")
        print("Please run main.py to train new models.")
else:
    print("Pre-trained models not found. Please run main.py to train the models first.")

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
    if not ticker:
        return dash.no_update, dash.no_update
    
    if not models_loaded:
        return {}, "Models not loaded. Please run main.py to train the models first."
    
    try:
        # Fetch stock data
        df = get_stock_data(ticker)
        
        # Engineer features
        df_features = engineer_features(df)
        
        # Prepare features for prediction
        X = df_features.drop(['Target'], axis=1)
        
        # Make predictions
        short_term_pred = short_term_model.predict(X.iloc[-1:])
        long_term_pred = long_term_model.predict(X.iloc[-1:])
        
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
        fig.add_trace(go.Scatter(x=future_dates, y=long_term_pred[0], mode='lines', name='30-day Forecast'))
        
        fig.update_layout(title=f"{ticker} Stock Price and Forecast",
                          xaxis_title="Date",
                          yaxis_title="Price",
                          height=600)
        
        # Add link to Yahoo Finance
        yahoo_link = f"https://finance.yahoo.com/quote/{ticker}"
        
        # Prepare prediction output
        current_price = df['Close'].iloc[-1]
        short_term_direction = "Up" if short_term_pred[0] == 1 else "Down"
        next_day_pred = current_price * (1.01 if short_term_pred[0] == 1 else 0.99)  # Assuming 1% change
        long_term_end_price = long_term_pred[0][-1]
        
        prediction_text = [
            html.P(f"Current Price: ${current_price:.2f}"),
            html.P(f"Short-term prediction (next day): Price likely to go {short_term_direction}"),
            html.P(f"Estimated next day price: ${next_day_pred:.2f}"),
            html.P(f"Long-term prediction (after 30 days): ${long_term_end_price:.2f}"),
            html.A("View on Yahoo Finance", href=yahoo_link, target="_blank")
        ]
        
        return fig, prediction_text
    except ValueError as e:
        # Return an empty figure and the error message
        return go.Figure(), html.P(str(e), style={'color': 'red'})
    except Exception as e:
        # Return an empty figure and a generic error message
        return go.Figure(), html.P(f"An error occurred: {str(e)}", style={'color': 'red'})

if __name__ == '__main__':
    app.run_server(debug=True)