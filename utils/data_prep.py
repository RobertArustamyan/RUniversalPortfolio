from datetime import datetime, timedelta
import yfinance as yf


def prepare_stock_data(stocks, train_days=730, test_days=365, end_date=None):
    if end_date is None:
        end_date = datetime.now()

    # Calculate start dates
    test_start = end_date - timedelta(days=test_days)
    train_start = test_start - timedelta(days=train_days)

    print(f"Downloading data for {len(stocks)} stocks")
    print(f"Train period: {train_start.date()} to {test_start.date()} ({train_days} calendar days)")
    print(f"Test period: {test_start.date()} to {end_date.date()} ({test_days} calendar days)")

    # Download data
    data = yf.download(stocks, start=train_start, end=end_date, auto_adjust=True)['Close']

    if len(stocks) == 1:
        data = data.to_frame(name=stocks[0])

    # Calculate price relatives
    price_relatives_df = (data / data.shift(1)).dropna()

    # Split train and test
    train_mask = price_relatives_df.index < test_start
    train_price_relatives_df = price_relatives_df[train_mask]
    test_price_relatives_df = price_relatives_df[~train_mask]

    # Convert to numpy arrays
    train_price_relatives = train_price_relatives_df.values
    test_price_relatives = test_price_relatives_df.values

    print(f"\nTrading days:")
    print(f"Train: {len(train_price_relatives)} days")
    print(f"Test: {len(test_price_relatives)} days")

    # Validation
    if len(train_price_relatives) == 0:
        raise ValueError("Training set is empty.")
    if len(test_price_relatives) == 0:
        raise ValueError("Test set is empty.")

    return {
        'train_price_relatives': train_price_relatives,
        'test_price_relatives': test_price_relatives,
        'train_dates': train_price_relatives_df.index,
        'test_dates': test_price_relatives_df.index,
        'stock_names': stocks,
        'data': data,
        'split_date': test_start
    }
