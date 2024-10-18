# Stock Price Prediction Dashboard

This project is a stock price prediction application powered by machine learning. It provides short-term and long-term predictions for stock prices using historical data and technical indicators.

## Features

- Real-time stock data fetching from Yahoo Finance
- Short-term (next day) price movement prediction
- Long-term (30-day) price forecast
- Interactive web dashboard for visualizing stock data and predictions
- Automated feature engineering and model training

## Technical Stack

- **Data Retrieval**: Yahoo Finance API
- **Data Processing**: Pandas, NumPy
- **Feature Engineering**: TA-Lib (Technical Analysis Library)
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Web Dashboard**: Dash (Python framework for building analytical web applications)

## Project Structure

- `data/`: Contains scripts for fetching stock data
- `preprocessing/`: Includes feature engineering and data preparation scripts
- `models/`: Defines short-term and long-term prediction models
- `dashboard/`: Contains the Dash application for the web interface
- `main.py`: Entry point for training models and running the dashboard

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/stock-prediction-dashboard.git
   cd stock-prediction-dashboard
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the main script to train the models and start the dashboard:
   ```
   python main.py
   ```

5. Open a web browser and navigate to `http://localhost:8050` to access the dashboard.

## Usage

1. Enter a stock ticker symbol in the input field on the dashboard.
2. Click the "Submit" button or press Enter.
3. The application will fetch the latest stock data, make predictions, and display the results.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This application is for educational and research purposes only. It should not be considered as financial advice. Always do your own research and consult with a qualified financial advisor before making investment decisions.
