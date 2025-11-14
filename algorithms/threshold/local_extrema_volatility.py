import numpy as np

class LocalExtremaVolatilityTrader:
    def __init__(self, lookback_window: int = 20, volatility_window: int = 10,
                 buy_zscore: float = -1.0, sell_zscore: float = 1.0,
                 min_extrema_samples: int = 5):
        """
        Trader that learns thresholds from actual local extrema and normalizes by volatility

        Parameters:
        - lookback_window: How far back to look for extrema patterns
        - volatility_window: Window for calculating rolling volatility
        - buy_zscore: Z-score threshold for buying (negative = drop)
        - sell_zscore: Z-score threshold for selling (positive = rise)
        - min_extrema_samples: Minimum extrema needed before trading
        """
        self.lookback_window = lookback_window
        self.volatility_window = volatility_window
        self.buy_zscore = buy_zscore
        self.sell_zscore = sell_zscore
        self.min_extrema_samples = min_extrema_samples

        self.holding = False
        self.wealth = 1.0
        self.shares = 0
        self.entry_price = 0

        self.buy_threshold = None
        self.sell_threshold = None

        self.price_history = []
        self.local_minima = []
        self.local_maxima = []
        self.minima_drops = []  # Price changes leading to minima
        self.maxima_rises = []  # Price changes leading to maxima

    def _detect_local_extrema(self, prices):
        minima_indices = []
        maxima_indices = []

        for i in range(1, len(prices) - 1):
            # Local minimums
            if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
                minima_indices.append(i)
            # Local maximums
            elif prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
                maxima_indices.append(i)

        return minima_indices, maxima_indices

    def _calculate_rolling_volatility(self, prices) :
        """Calculate rolling standard deviation of returns"""
        if len(prices) < 2:
            return np.array([0.01])

        returns = np.diff(prices) / prices[:-1]
        volatilities = []

        for i in range(len(returns)):
            start_idx = max(0, i - self.volatility_window + 1)
            window_returns = returns[start_idx:i + 1]

            if len(window_returns) > 1:
                vol = np.std(window_returns)
                volatilities.append(vol if vol > 1e-6 else 0.01)
            else:
                volatilities.append(0.01)

        return np.array(volatilities)

    def learn_initial_thresholds(self, historical_prices):
        """Learn buy/sell thresholds from local extremums in historical data"""
        self.price_history = list(historical_prices)

        # Detect all local extrema
        minima_idx, maxima_idx = self._detect_local_extrema(historical_prices)

        # Calculate volatility at each point
        volatilities = self._calculate_rolling_volatility(historical_prices)

        # Analyze price changes leading to each minimum
        for idx in minima_idx:
            if idx > 0:
                price_change = (historical_prices[idx] - historical_prices[idx - 1]) / historical_prices[idx - 1]
                vol = volatilities[idx - 1] if idx - 1 < len(volatilities) else 0.01
                normalized_change = price_change / vol
                self.minima_drops.append(normalized_change)

        # Analyze price changes leading to each maximum
        for idx in maxima_idx:
            if idx > 0:
                price_change = (historical_prices[idx] - historical_prices[idx - 1]) / historical_prices[idx - 1]
                vol = volatilities[idx - 1] if idx - 1 < len(volatilities) else 0.01
                normalized_change = price_change / vol
                self.maxima_rises.append(normalized_change)

        # Set thresholds based on extrema distributions
        if len(self.minima_drops) >= self.min_extrema_samples:
            # Buy threshold
            self.buy_threshold = np.median(self.minima_drops)
        else:
            # Fallback to z-score based threshold
            self.buy_threshold = self.buy_zscore

        if len(self.maxima_rises) >= self.min_extrema_samples:
            # Sell threshold
            self.sell_threshold = np.median(self.maxima_rises)
        else:
            # Fallback to z-score based threshold
            self.sell_threshold = self.sell_zscore

        print(f"Initial thresholds learned from extrema:")
        print(f"Buy threshold: {self.buy_threshold:.4f}")
        print(f"Sell threshold: {self.sell_threshold:.4f}")

    def _get_current_volatility(self):
        """Calculate current volatility from recent price history"""
        if len(self.price_history) < 2:
            return 0.01

        recent_prices = np.array(self.price_history[-self.volatility_window:])
        if len(recent_prices) < 2:
            return 0.01

        returns = np.diff(recent_prices) / recent_prices[:-1]
        vol = np.std(returns)
        return vol if vol > 1e-6 else 0.01

    def _update_extrema_history(self, current_price):
        if len(self.price_history) < 3:
            return

        # Check last 3 prices to see if middle one was an extremum
        p1 = self.price_history[-3]
        p2 = self.price_history[-2]
        p3 = self.price_history[-1]

        vol = self._get_current_volatility()

        # chek if p2 was local minimum
        if p2 < p1 and p2 < p3:
            price_change = (p2 - p1) / p1
            normalized_change = price_change / vol
            self.minima_drops.append(normalized_change)

            # Keep only recent extrema
            if len(self.minima_drops) > self.lookback_window:
                self.minima_drops = self.minima_drops[-self.lookback_window:]

            # Update buy threshold
            if len(self.minima_drops) >= self.min_extrema_samples:
                self.buy_threshold = np.median(self.minima_drops)

        # chek if p2 was local maximum
        elif p2 > p1 and p2 > p3:
            price_change = (p2 - p1) / p1
            normalized_change = price_change / vol
            self.maxima_rises.append(normalized_change)

            # Keep only recent extrema
            if len(self.maxima_rises) > self.lookback_window:
                self.maxima_rises = self.maxima_rises[-self.lookback_window:]

            # Update sell threshold
            if len(self.maxima_rises) >= self.min_extrema_samples:
                self.sell_threshold = np.median(self.maxima_rises)

    def make_decision(self, current_price):
        if len(self.price_history) == 0:
            self.price_history.append(current_price)
            return "hold"

        prev_price = self.price_history[-1]
        price_change = (current_price - prev_price) / prev_price

        vol = self._get_current_volatility()
        normalized_change = price_change / vol

        decision = "hold"

        # Buy logic
        if not self.holding and normalized_change <= self.buy_threshold:
            self.holding = True
            self.entry_price = current_price
            self.shares = self.wealth / current_price
            decision = "buy"

        # Sell logic
        elif self.holding and normalized_change >= self.sell_threshold:
            self.wealth = self.shares * current_price
            self.shares = 0
            self.holding = False
            decision = "sell"

        # updating history and checking for new extremas
        self.price_history.append(current_price)
        self._update_extrema_history(current_price)

        return decision

    def get_current_wealth(self, current_price):
        if self.holding:
            return self.shares * current_price
        return self.wealth

    def trade_online(self, future_prices):
        """
        trade as new prices come one by one
        """
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

    def get_extrema_statistics(self):
        return {
            'minima_drops': self.minima_drops.copy(),
            'maxima_rises': self.maxima_rises.copy(),
            'buy_threshold': self.buy_threshold,
            'sell_threshold': self.sell_threshold,
            'n_minima': len(self.minima_drops),
            'n_maxima': len(self.maxima_rises)
        }