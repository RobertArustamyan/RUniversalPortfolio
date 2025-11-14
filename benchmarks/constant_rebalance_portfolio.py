"""
Constant Rebalanced Portfolio (CRP) Strategy
Fixed portfolio weights by rebalancing at the end of each day
"""

import numpy as np


class ConstantRebalancedPortfolio:
    def __init__(self, weights):
        self.weights = np.array(weights)
        self.weights = self.weights / np.sum(self.weights)
        self.daily_wealth = []
        self.daily_portfolios = []

    def run(self, price_relatives):
        price_relatives = np.array(price_relatives)
        n_days, n_stocks = price_relatives.shape

        if self.weights is None:
            self.weights = np.ones(n_stocks) / n_stocks

        if len(self.weights) != n_stocks:
            raise ValueError(f"Weights length {len(self.weights)} != n_stocks {n_stocks}")

        self.daily_wealth = [1.0]
        self.daily_portfolios = [self.weights.copy()]

        current_wealth = 1.0

        for day in range(n_days):
            daily_return = np.dot(self.weights, price_relatives[day])
            current_wealth *= daily_return

            self.daily_wealth.append(current_wealth)

            self.daily_portfolios.append(self.weights.copy())

        return {
            'strategy': 'Constant Rebalanced Portfolio',
            'final_wealth': self.daily_wealth[-1],
            'daily_wealth': np.array(self.daily_wealth),
            'daily_portfolios': np.array(self.daily_portfolios),
            'n_days': n_days,
            'weights': self.weights
        }


def run_crp(price_relatives, weights=None):
    if weights is None:
        n_stocks = price_relatives.shape[1]
        weights = np.ones(n_stocks) / n_stocks

    crp = ConstantRebalancedPortfolio(weights=weights)
    return crp.run(price_relatives)