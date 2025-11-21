from datetime import datetime, timedelta

import numpy as np
import yfinance as yf

from benchmarks.portfolio.compare_strategies import compare_strategies
from algorithms.portfolio.barrons import Barrons

if __name__ == '__main__':
    # Settings for saving results
    save_plot = False
    path = '../../plots/portfolio'
    fig_name = 'barrons_4stocks_2train_1test'

    # Flag to control wealth continuation
    continue_wealth_from_training = True

    stocks = ["AAPL", "TSLA", "MSFT", "IBM", "ORCL", "AMZN","INTC", "AMD"]

    # Define time periods
    end_date = datetime.now()
    test_start = end_date - timedelta(days=365)  # Last year for testing
    train_start = test_start - timedelta(days=730)  # 2 years before

    print("Downloading stock data")
    data = yf.download(stocks, start=train_start, end=end_date, auto_adjust=True)['Close']
    price_relatives = (data / data.shift(1)).dropna().values

    # Split into train and test
    train_end_idx = len(data[data.index < test_start]) - 1
    train_price_relatives = price_relatives[:train_end_idx]
    test_price_relatives = price_relatives[train_end_idx:]

    print(f"\nTrain period: {len(train_price_relatives)} days")
    print(f"Test period: {len(test_price_relatives)} days")

    # Portfolio Initialization
    # Note: T should be set to the expected number of trading rounds
    total_rounds = len(train_price_relatives) + len(test_price_relatives)

    portfolio = Barrons(
        n_stocks=len(stocks),
        beta=0.01,  # Between 0 and 0.5
        eta=0.9,  # Between 0 and 1
        T=total_rounds  # Total expected rounds
    )

    print("\nTraining\n")

    train_results = portfolio.simulate_trading(train_price_relatives, verbose=True)
    train_wealth = train_results['final_wealth']

    print("\nTesting\n")


    # Initialize test wealth based on flag
    if continue_wealth_from_training:
        test_daily_wealth = [train_results['daily_wealth'][-1]]
        test_wealth_current = train_results['daily_wealth'][-1]
        print(f"Continuing from training wealth: {test_wealth_current:.4f}")
    else:
        test_daily_wealth = [1.0]
        test_wealth_current = 1.0
        print("Starting fresh with wealth = 1.0")

    for day, price_rel in enumerate(test_price_relatives):
        current_portfolio = portfolio.x_t.copy()
        daily_return = np.dot(current_portfolio, price_rel)
        test_wealth_current *= daily_return
        test_daily_wealth.append(test_wealth_current)
        portfolio.update(price_rel)

        if (day + 1) % 10 == 0 or day == 0:
            print(f"Test Day {day + 1}: Wealth = {test_wealth_current:.4f}, " +
                  f"Portfolio = [{', '.join([f'{x:.3f}' for x in current_portfolio])}]")


    # Prepare results for comparison
    if continue_wealth_from_training:
        # Combine train and test wealth
        combined_daily_wealth = np.concatenate([
            train_results['daily_wealth'][:-1],
            test_daily_wealth
        ])
        barrons_results = {
            'strategy': 'Barrons',
            'daily_wealth': combined_daily_wealth,
            'final_wealth': test_wealth_current
        }
        price_relatives_for_plot = np.vstack([train_price_relatives, test_price_relatives])
    else:
        # Test only
        barrons_results = {
            'strategy': 'Barrons (trained on 2yr)',
            'daily_wealth': np.array(test_daily_wealth),
            'final_wealth': test_wealth_current
        }
        price_relatives_for_plot = test_price_relatives

    results = compare_strategies(
        price_relatives_for_plot,
        stock_names=stocks,
        show_plot=True,
        save_path=f'{path}/{fig_name}.png' if save_plot else None,
        crp_weights=np.ones(len(stocks)) / len(stocks),
        additional_results=barrons_results
    )