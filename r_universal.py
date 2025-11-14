import numpy as np
from scipy.optimize import brentq
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from benchmarks.compare_strategies import compare_strategies


class RUniversalPortfolio:
    def __init__(self, n_stocks, epsilon=None, eta=None, T=None, A=100,
                 m=None, S=None, delta=None, delta_0=None, use_damping=True,
                 n_processes=None, use_parallel=False):
        '''
        R-UNIVERSAL Portfolio Algorithm

        Two modes of operation:

        Mode 1 - Theoretical parameters (provide epsilon, eta, T):
            Calculates m, S, delta, delta_0 according to Theorem 2

        Mode 2 - Manual parameters (provide m, S, delta, delta_0):
            Uses your specified values directly

        :param n_stocks: Number of stocks
        :param epsilon: Performance approximation parameter (Mode 1)
        :param eta: Probability of failure (Mode 1)
        :param T: Number of trading days (Mode 1)
        :param A: Constant from paper (default: 100)
        :param m: Number of samples per day (Mode 2)
        :param S: Number of random walk steps per sample (Mode 2)
        :param delta: Grid spacing (Mode 2)
        :param delta_0: Minimum coordinate value (Mode 2)
        :param use_damping: If True, use Q_t (with damping); if False, use P_t
        :param n_processes: Number of parallel processes (None = auto-detect)
        :param use_parallel: Enable multiprocessing (recommended for m > 100)
        '''

        self.n = n_stocks
        self.use_damping = use_damping
        self.use_parallel = use_parallel
        self.n_processes = n_processes if n_processes else cpu_count()
        self.t = 0
        self.current_portfolio = np.ones(n_stocks) / n_stocks
        self.price_relatives_history = []

        # Find mode (either manual or theoretical)
        manual_params_provided = all(x is not None for x in [m, S, delta, delta_0])
        theoretical_params_provided = all(x is not None for x in [epsilon, eta, T])

        if manual_params_provided:
            self.mode = "manual"
            self.m = m
            self.S = S
            self.delta = delta
            self.delta_0 = delta_0
            # Keeping values (can be None)
            self.epsilon = epsilon
            self.eta = eta
            self.T = T
            self.A = A

        elif theoretical_params_provided:
            self.mode = "theoretical"
            self.epsilon = epsilon
            self.eta = eta
            self.T = T
            self.A = A
            self._calculate_theoretical_params()

        else:
            raise ValueError(
                "Must provide either:\n"
                "  - Theoretical params: epsilon, eta, T\n"
                "  - Manual params: m, S, delta, delta_0"
            )

        self._print_config()

    def _calculate_theoretical_params(self):
        """Calculate parameters according to Theorem 2"""
        n = self.n
        T = self.T
        epsilon = self.epsilon
        eta = self.eta
        A = self.A

        # delta_0 <= epsilon / (8*n*T*(n+T)^2)
        self.delta_0 = epsilon / (8 * n * T * (n + T) ** 2)

        # Solve: delta * log(1/delta) = epsilon*delta_0 / (A*(n+T)^2)
        target_value = (epsilon * self.delta_0) / (A * (n + T) ** 2)
        self.delta = self._solve_for_delta(target_value)

        # m >= 64*T^2*(n+T)*ln(nT/eta) / epsilon^2
        m_raw = 64 * T ** 2 * (n + T) * np.log(n * T / eta) / epsilon ** 2
        self.m = int(np.ceil(min(m_raw, 1e9)))  # Cap at 1 billion

        # S >= A*n/delta^2 * log((n+T)/(epsilon*delta))
        if self.delta > 0:
            S_raw = A * n / self.delta ** 2 * np.log((n + T) / (epsilon * self.delta))
            self.S = int(np.ceil(min(S_raw, 1e9)))  # Cap at 1 billion
        else:
            self.S = 10000

    @staticmethod
    def _solve_for_delta(target_value):
        """
        Solve: delta * log(1/delta) = target_value
        Using scipy's Brent method
        """
        if target_value <= 0:
            return 1e-6

        # Check if target is achievable
        max_value = 1.0 / np.e  # Maximum at delta = 1/e
        if target_value > max_value:
            print(f"⚠️  Target {target_value:.2e} > max {max_value:.4f}, using fallback")
            return 1.0 / np.e

        if target_value < 1e-15:
            print(f"⚠️  Target {target_value:.2e} too small, using practical fallback")
            return 0.01

        def equation(delta):
            if delta <= 0:
                return float('inf')
            return delta * np.log(1.0 / delta) - target_value

        try:
            solution = brentq(equation, 1e-10, 0.368, xtol=1e-15)

            # Verify
            check = solution * np.log(1.0 / solution)
            if abs(check - target_value) / target_value > 0.1:
                print(f"⚠️  Solution verification failed, using fallback")
                return min(0.01, np.sqrt(target_value * 10))

            return solution
        except:
            return min(0.01, np.sqrt(target_value * 10))

    def _print_config(self):
        print(f"R-UNIVERSAL Portfolio Configuration")
        print(f"Mode: {self.mode}")
        print(f"Stocks: {self.n}")
        print(f"Damping: {'ENABLED (Q_t)' if self.use_damping else 'DISABLED (P_t)'}")
        print(f"Parallel: {'YES ({} processes)'.format(self.n_processes) if self.use_parallel else 'NO'}")

        if self.mode == "theoretical":
            print(f"\nTheoretical Parameters:")
            print(f"  epsilon = {self.epsilon}")
            print(f"  eta = {self.eta}")
            print(f"  T = {self.T}")
            print(f"  A = {self.A}")

        print(f"\nAlgorithm Parameters:")
        print(f"  delta_0 = {self.delta_0:.6e}")
        print(f"  delta = {self.delta:.6e}")
        print(f"  m = {self.m:,}")
        print(f"  S = {self.S:,}")

        if self.m > 1e6 or self.S > 1e6:
            print(f"  ⚠️  Large computational requirements")

        if not self.use_parallel and self.m > 100:
            print(f"  💡 Consider use_parallel=True for m > 100")

    def P_t(self, portfolio):
        """
        Cumulative wealth function P_t(b)
        P_t(b) = PRODUCT(i=1 to t) b·x_i
        """
        if len(self.price_relatives_history) == 0:
            return 1.0

        wealth = 1.0
        for price_relatives in self.price_relatives_history:
            daily_return = np.dot(portfolio, price_relatives)
            wealth *= daily_return

        return wealth

    def Q_t(self, portfolio):
        """
        Damped wealth function Q_t(b)
        Q_t(b) = P_t(b) * min(exp((b_n - 2*delta_0)/(n*delta)), 1)

        Only used when use_damping=True
        """
        P_t_value = self.P_t(portfolio)

        if not self.use_damping:
            return P_t_value

        # Calculate damping factor
        b_n = portfolio[-1]
        damping_exp = (b_n - 2 * self.delta_0) / (self.n * self.delta)
        damping_factor = min(np.exp(damping_exp), 1.0)

        return P_t_value * damping_factor

    def _wealth_function(self, portfolio):
        """Choose between P_t and Q_t based on use_damping"""
        return self.Q_t(portfolio) if self.use_damping else self.P_t(portfolio)

    def _random_walk_step(self, r):
        """
        One step of Metropolis-Hastings random walk

        Acceptance: min(1, wealth_proposed / wealth_current)

        Two modes:
        1. Asymmetric (use_damping=True): Pick j from {0,...,n-2}, pair with last coord
        2. Symmetric (use_damping=False): Pick any two different coordinates
        """
        coords = np.random.choice(self.n, size=2, replace=False)
        i, j = coords[0], coords[1]

        if self.use_damping:
            i = -1  # Always pair with last coordinate

        X = np.random.choice([-1, 1])
        new_r = r.copy()
        new_r[j] += X * self.delta
        new_r[i] -= X * self.delta

        # Check constraints
        if new_r[j] >= self.delta_0 and new_r[i] >= self.delta_0:
            wealth_current = self._wealth_function(r)
            wealth_proposed = self._wealth_function(new_r)

            if wealth_current > 0 and wealth_proposed > 0:
                acceptance_prob = min(1.0, wealth_proposed / wealth_current)
                if np.random.random() < acceptance_prob:
                    return new_r

        return r

    def _sample_portfolio(self):
        """Generate one portfolio sample via S steps of random walk"""
        # Start at uniform distribution
        r = np.ones(self.n) / self.n

        # Snap to grid
        r = np.round(r / self.delta) * self.delta
        r = np.maximum(r, self.delta_0)
        r = r / np.sum(r)  # Renormalize

        # Perform S steps of random walk
        for _ in range(self.S):
            r = self._random_walk_step(r)

        return r

    def _sample_portfolio_wrapper(self, _):
        """Wrapper for multiprocessing (ignores dummy argument)"""
        return self._sample_portfolio()

    def update(self, price_relatives):
        """
        Update portfolio AFTER observing today's price relatives

        Uses multiprocessing if use_parallel=True

        :param price_relatives: Array[n] of price relatives (close/open)
        :return: Portfolio for NEXT day
        """
        self.price_relatives_history.append(price_relatives.copy())
        self.t += 1

        # Generate m samples (parallel or sequential)
        if self.use_parallel:
            with Pool(processes=self.n_processes) as pool:
                samples = pool.map(self._sample_portfolio_wrapper, range(self.m))
            samples = np.array(samples)
        else:
            samples = np.array([self._sample_portfolio() for _ in range(self.m)])

        # Average samples (approximates the integral in Definition 1)
        self.current_portfolio = np.mean(samples, axis=0)
        self.current_portfolio /= np.sum(self.current_portfolio)

        return self.current_portfolio.copy()

    def get_portfolio(self):
        """Get current portfolio for trading (BEFORE seeing today's prices)"""
        return self.current_portfolio.copy()

    def simulate_trading(self, price_relatives_sequence, verbose=True):
        """
        Simulate trading over a sequence of days

        :param price_relatives_sequence: List of price relative arrays
        :param verbose: Print progress
        :return: Dictionary with performance metrics
        """
        wealth = 1.0
        daily_wealth = [1.0]
        portfolios_used = []

        for day, price_relatives in enumerate(price_relatives_sequence):
            # Use current portfolio for today's trading
            portfolio = self.get_portfolio()
            portfolios_used.append(portfolio.copy())

            # Calculate today's return
            daily_return = np.dot(portfolio, price_relatives)
            wealth *= daily_return
            daily_wealth.append(wealth)

            # Update for next day
            self.update(price_relatives)

            if verbose and ((day + 1) % 10 == 0 or day == 0):
                print(f"Day {day + 1}: Wealth = {wealth:.4f}")

        return {
            'final_wealth': wealth,
            'daily_wealth': np.array(daily_wealth),
            'portfolios_used': np.array(portfolios_used),
            'num_days': len(price_relatives_sequence)
        }


if __name__ == '__main__':
    stocks = ["AAPL", "TSLA", "NVDA"]

    # Define time periods
    end_date = datetime.now()
    test_start = end_date - timedelta(days=30)  # Last month for testing
    train_start = test_start - timedelta(days=365)  # One year before

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
        train_results['daily_wealth'][:-1],  # Exclude last train day (it's first test day)
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
        save_path='portfolio_comparison.png',
        additional_results=runiversal_results
    )