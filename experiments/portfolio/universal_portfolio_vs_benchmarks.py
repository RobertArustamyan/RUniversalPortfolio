from datetime import datetime, timedelta

import numpy as np
import yfinance as yf

from benchmarks.portfolio.compare_strategies import compare_strategies
from algorithms.portfolio.r_universal import RUniversalPortfolio


if __name__ == '__main__':
    # Settings for saving results
    save_plot = True
    path = '../../plots/portfolio'
    fig_name = 'portfolio_comparison'

    stocks = ["AAPL", "TSLA", "NVDA"]

    # Define time periods
    end_date = datetime.now()
    test_start = end_date - timedelta(days=30)  # Last month for testing
    train_start = test_start - timedelta(days=400)  # One year before

    print("Downloading stock data")
    data = yf.download(stocks, start=train_start, end=end_date, auto_adjust=True)['Close']
    price_relatives = (data / data.shift(1)).dropna().values

    # Split into train and test
    train_end_idx = len(data[data.index < test_start]) - 1
    train_price_relatives = price_relatives[:train_end_idx]
    test_price_relatives = price_relatives[train_end_idx:]

    print(f"\nTrain period: {len(train_price_relatives)}")
    print(f"Test period: {len(test_price_relatives)}")

    # Portfolio Initialization
    portfolio = RUniversalPortfolio(
        n_stocks=len(stocks),
        m=250,
        S=8000,
        delta=0.01,
        delta_0=1e-6,
        use_damping=False,
        use_parallel=True,
        n_processes=6
    )

    train_results = portfolio.simulate_trading(train_price_relatives, verbose=True)
    train_wealth = train_results['final_wealth']

    print("\nTesting\n")

    # test_wealth = 1.0
    # for day, price_rel in enumerate(test_price_relatives):
    #     current_portfolio = portfolio.get_portfolio()
    #     daily_return = np.dot(current_portfolio, price_rel)
    #     test_wealth *= daily_return
    #     portfolio.update(price_rel)
    #
    #     print(f"Test Day {day + 1}: Wealth = {test_wealth:.4f}, " +
    #           f"Portfolio = [{', '.join([f'{x:.3f}' for x in current_portfolio])}]")

    test_daily_wealth = [train_results['daily_wealth'][-1]]  # Start from final train wealth
    test_wealth_current = train_results['daily_wealth'][-1]

    for day, price_rel in enumerate(test_price_relatives):
        current_portfolio = portfolio.get_portfolio()
        daily_return = np.dot(current_portfolio, price_rel)
        test_wealth_current *= daily_return
        test_daily_wealth.append(test_wealth_current)
        portfolio.update(price_rel)

        print(f"Test Day {day + 1}: Wealth = {test_wealth_current:.4f}, " +
              f"Portfolio = [{', '.join([f'{x:.3f}' for x in current_portfolio])}]")

    combined_daily_wealth = np.concatenate([
        train_results['daily_wealth'][:-1],
        test_daily_wealth
    ])

    runiversal_results = {
        'strategy': 'R-Universal',
        'daily_wealth': combined_daily_wealth,
        'final_wealth': test_wealth_current
    }

    all_price_relatives = np.vstack([train_price_relatives, test_price_relatives])
    results = compare_strategies(
        all_price_relatives,
        stock_names=stocks,
        show_plot=True,
        save_path=f'{path}/{fig_name}.png' if save_plot else None,
        crp_weights=[0.33, 0.33, 0.33],
        additional_results=runiversal_results
    )
