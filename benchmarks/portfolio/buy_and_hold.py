"""
Buy and hold strategy
Buy equal amount of each stock at the beginning and hold forever
"""
import numpy as np

class BuyAndHold():
    def __init__(self, initial_weights=None):
        self.initial_weights = initial_weights
        self.daily_wealth = []
        self.daily_portfolios = []

    def run(self, price_relatives):
        price_relatives = np.array(price_relatives)
        n_days, n_stocks = price_relatives.shape

        if self.initial_weights is None:
            self.initial_weights = np.ones(n_stocks) / n_stocks
        else:
            if len(self.initial_weights) != n_stocks:
                raise ValueError(f"Weights length {len(self.initial_weights)} != n_stocks {n_stocks}")

            initial_weights = np.array(self.initial_weights)
            self.initial_weights = initial_weights / np.sum(initial_weights)

        # Start with wealth 1 and given (or uniformly) portfolio
        self.daily_wealth = [1.0]
        self.daily_portfolios = [self.initial_weights.copy()]

        portfolio_values = self.initial_weights.copy()

        for day in range(n_days):
            portfolio_values = portfolio_values * price_relatives[day]

            total_wealth = np.sum(portfolio_values)
            self.daily_wealth.append(total_wealth)

            current_weights = portfolio_values / total_wealth
            self.daily_portfolios.append(current_weights.copy())

        return {
            'strategy': 'Buy and Hold',
            'final_wealth': self.daily_wealth[-1],
            'daily_wealth': np.array(self.daily_wealth),
            'daily_portfolios': np.array(self.daily_portfolios),
            'n_days': n_days,
            'initial_weights': self.initial_weights
        }


def run_buy_and_hold(price_relatives, initial_weights=None):
    bah = BuyAndHold(initial_weights=initial_weights)
    return bah.run(price_relatives)