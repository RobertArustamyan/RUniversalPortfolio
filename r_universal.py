import numpy as np
from scipy.optimize import brentq


class RUniversalPortfolio:
    def __init__(self, n_stocks, epsilon=None, eta=None, T=None, A=100,
                 m=None, S=None, delta=None, delta_0=None, use_damping=True):
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
        '''

        self.n = n_stocks
        self.use_damping = use_damping
        self.t = 0
        self.current_portfolio = np.ones(n_stocks) / n_stocks
        self.price_relatives_history = []

        # Determine which mode to use
        manual_params_provided = all(x is not None for x in [m, S, delta, delta_0])
        theoretical_params_provided = all(x is not None for x in [epsilon, eta, T])

        if manual_params_provided:
            # Mode 2: Use manual parameters
            self.mode = "manual"
            self.m = m
            self.S = S
            self.delta = delta
            self.delta_0 = delta_0
            self.epsilon = epsilon  # Store for reference (may be None)
            self.eta = eta
            self.T = T
            self.A = A

        elif theoretical_params_provided:
            # Mode 1: Calculate from theoretical bounds
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
        """Print configuration"""
        print(f"\n{'=' * 70}")
        print(f"R-UNIVERSAL Portfolio Configuration")
        print(f"{'=' * 70}")
        print(f"Mode:        {self.mode}")
        print(f"Stocks:      {self.n}")
        print(f"Damping:     {'ENABLED (Q_t)' if self.use_damping else 'DISABLED (P_t)'}")

        if self.mode == "theoretical":
            print(f"\nTheoretical Parameters:")
            print(f"  epsilon = {self.epsilon}")
            print(f"  eta     = {self.eta}")
            print(f"  T       = {self.T}")
            print(f"  A       = {self.A}")

        print(f"\nAlgorithm Parameters:")
        print(f"  delta_0 = {self.delta_0:.6e}")
        print(f"  delta   = {self.delta:.6e}")
        print(f"  m       = {self.m:,}")
        print(f"  S       = {self.S:,}")

        if self.m > 1e6 or self.S > 1e6:
            print(f"\n⚠️  WARNING: Parameters are very large!")
            print(f"   Consider using manual mode with smaller values")

        print(f"{'=' * 70}\n")

    def P_t(self, portfolio):
        """
        Cumulative wealth function P_t(b)
        P_t(b) = ∏(i=1 to t) b·x_i
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
        return self.Q_t(portfolio)

    def _random_walk_step(self, r):
        """
        One step of Metropolis-Hastings random walk

        With damping (use_damping=True):
            - Pick coordinate j from {0, ..., n-2}
            - Adjust j and coordinate n to maintain sum=1

        Without damping (use_damping=False):
            - Pick ANY two different coordinates i, j
            - Increase one, decrease the other
            This is more symmetric (as suggested in Section 7)
        """
        if self.use_damping:
            # Original algorithm: treat coordinate n specially
            j = np.random.randint(0, self.n - 1)
            X = np.random.choice([-1, 1])

            new_r = r.copy()
            new_r[j] += X * self.delta
            new_r[-1] -= X * self.delta

            # Check constraints
            if new_r[j] >= self.delta_0 and new_r[-1] >= self.delta_0:
                wealth_current = self._wealth_function(r)
                wealth_proposed = self._wealth_function(new_r)

                if wealth_proposed > 0:
                    acceptance_prob = min(1.0, wealth_current / wealth_proposed)
                    if np.random.random() < acceptance_prob:
                        return new_r
        else:
            # Without damping: pick ANY two coordinates (more symmetric)
            # Pick two different coordinates
            coords = np.random.choice(self.n, size=2, replace=False)
            i, j = coords[0], coords[1]
            X = np.random.choice([-1, 1])

            new_r = r.copy()
            new_r[i] += X * self.delta
            new_r[j] -= X * self.delta

            # Check constraints
            if new_r[i] >= self.delta_0 and new_r[j] >= self.delta_0:
                wealth_current = self._wealth_function(r)
                wealth_proposed = self._wealth_function(new_r)

                if wealth_proposed > 0:
                    acceptance_prob = min(1.0, wealth_current / wealth_proposed)
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
        r = r / np.sum(r)  # Renormalize to ensure sum = 1

        # Take S steps of random walk
        for _ in range(self.S):
            r = self._random_walk_step(r)

        return r

    def update(self, price_relatives):
        """
        Update portfolio based on today's price relatives

        :param price_relatives: Array[n] of price relatives (close/open)
        :return: New portfolio allocation
        """
        self.price_relatives_history.append(price_relatives.copy())
        self.t += 1

        # Generate m samples
        samples = np.array([self._sample_portfolio() for _ in range(self.m)])

        # Average samples (approximates the integral in Definition 1)
        self.current_portfolio = np.mean(samples, axis=0)
        self.current_portfolio /= np.sum(self.current_portfolio)

        return self.current_portfolio.copy()

    def get_portfolio(self):
        """Get current portfolio allocation"""
        return self.current_portfolio.copy()

    def get_cumulative_wealth(self):
        """Calculate cumulative wealth of current portfolio"""
        return self.P_t(self.current_portfolio)


if __name__ == '__main__':
    portfolio1 = RUniversalPortfolio(
        n_stocks=3,
        epsilon=0.2,
        eta=0.1,
        T=10,
        A=100,
        use_damping=True
    )

    portfolio2 = RUniversalPortfolio(
        n_stocks=3,
        m=100,
        S=1000,
        delta=0.01,
        delta_0=1e-6,
        use_damping=True
    )

    portfolio3 = RUniversalPortfolio(
        n_stocks=3,
        m=100,
        S=1000,
        delta=0.01,
        delta_0=1e-6,
        use_damping=False
    )