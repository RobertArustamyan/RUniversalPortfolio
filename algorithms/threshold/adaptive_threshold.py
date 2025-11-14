import numpy as np

class AdaptiveThresholdTrader:
    def __init__(self, lookback_window: int = 20, percentile_buy: float = 20,
                 percentile_sell: float = 80, adaptation_rate: float = 0.1):
        self.lookback_window = lookback_window
        self.percentile_buy = percentile_buy
        self.percentile_sell = percentile_sell
        self.adaptation_rate = adaptation_rate

        self.holding = False
        self.wealth = 1.0
        self.shares = 0
        self.entry_price = 0

        self.buy_threshold = None
        self.sell_threshold = None

        self.price_history = []
        self.price_changes = []

    def learn_initial_thresholds(self, historical_prices):
        """Learn initial buy-sell thresholds from historical data"""
        # Calculate price changes (percentage changes)
        price_changes = np.diff(historical_prices) / historical_prices[:-1]

        price_changes = price_changes[~np.isnan(price_changes)]
        price_changes = price_changes[~np.isinf(price_changes)]

        # Set thresholds based on percentiles
        self.buy_threshold = np.percentile(price_changes, self.percentile_buy)
        self.sell_threshold = np.percentile(price_changes, self.percentile_sell)

        # Initialize history
        self.price_history = list(historical_prices)
        self.price_changes = list(price_changes)

        print(f"Initial thresholds learned:")
        print(f"Buy: {self.buy_threshold:.4f}")
        print(f"Sell: {self.sell_threshold:.4f}")

    def update_thresholds(self, new_price_change):
        # Adaptive updating
        self.price_changes.append(new_price_change)

        # Keeping recent values
        if len(self.price_changes) > self.lookback_window:
            self.price_changes = self.price_changes[-self.lookback_window:]


        recent_changes = np.array(self.price_changes)
        new_buy = np.percentile(recent_changes, self.percentile_buy)
        new_sell = np.percentile(recent_changes, self.percentile_sell)

        # Blend old and new thresholds
        self.buy_threshold = ((1 - self.adaptation_rate) * self.buy_threshold +
                              self.adaptation_rate * new_buy)
        self.sell_threshold = ((1 - self.adaptation_rate) * self.sell_threshold +
                               self.adaptation_rate * new_sell)

    def make_decision(self, current_price):
        if len(self.price_history) == 0:
            return "hold"

        prev_price = self.price_history[-1]
        price_change = (current_price - prev_price) / prev_price

        # Update thresholds with new information
        self.update_thresholds(price_change)

        decision = "hold"

        # Buy logic
        if not self.holding and price_change <= self.buy_threshold:
            self.holding = True
            self.entry_price = current_price
            self.shares = self.wealth / current_price
            decision = "buy"

        # Sell logic
        elif self.holding and price_change >= self.sell_threshold:
            self.wealth = self.shares * current_price
            self.shares = 0
            self.holding = False
            decision = "sell"

        self.price_history.append(current_price)

        return decision

    def get_current_wealth(self, current_price):
        if self.holding:
            return self.shares * current_price
        return self.wealth

    def trade_online(self, future_prices):
        daily_wealth = []
        decisions = []

        for price in future_prices:
            decision = self.make_decision(price)
            decisions.append(decision)
            current_wealth = self.get_current_wealth(price)
            daily_wealth.append(current_wealth)

        if self.holding:
            final_price = future_prices[-1]
            self.wealth = self.shares * final_price
            self.holding = False

        return self.wealth, np.array(daily_wealth), decisions