import numpy as np
from scipy.optimize import fsolve, brentq, minimize_scalar
from scipy.special import lambertw


class RUniversalPortfolio:
    def __init__(self, n_stocks, epsilon, eta, T):
        '''
        :param n_stocks: Number of stocks
        :param epsilon: Performance measure parameter from paper
        :param eta: Probability of failure
        :param T: Number of trading days
        '''

        self.n = n_stocks
        self.epsilon = epsilon
        self.eta = eta
        self.t = 0  # Current day
        self.T = T
        self.A = 100

        self.last_portfolio = np.ones(n_stocks) / n_stocks  # Uniform distribution
        self.price_relatives_history = []

        self.__calculate_params()

    def __calculate_params(self):
        """
        Calculate parameters for the algorithm
        The constant A relates to the mixing time of the random walk
        """
        n = self.n
        T = self.T
        epsilon = self.epsilon
        eta = self.eta
        A = self.A

        # delta_0 <= epsilon / (8*n*T*(n+T)^2)
        self.delta_0 = epsilon / (8 * n * T * (n + T) ** 2)

        # Solve: delta * log(1/delta) = epsilon*delta_0 / (A*(n+T)^2)
        target_value = (epsilon * self.delta_0) / (A * (n + T) ** 2)
        self.delta = 0 # TODO find the value for delta 0

        # m >= 64*T^2*(n+T)*ln(nT/eta) / epsilon^2
        self.m = int(np.ceil(64 * T ** 2 * (n + T) * np.log(n * T / eta) / epsilon ** 2))

        # S >= A*n/delta^2 * log((n+T)/(epsilon*delta))
        self.S = int(np.ceil(A * n / self.delta ** 2 * np.log((n + T) / (epsilon * self.delta))))

    def Q_t(self, portfolio):
        """
        Calculate Q_t(b) - damped wealth function
        Q_t(b) = P_t(b) * min(exp((b_n - 2*delta_0)/(n*delta)), 1)
        """
        P_t = 1.0

        if len(self.price_relatives_history) == 0:
            return P_t

        # Calculating P_t(b) - cumulative wealth
        for price_relatives in self.price_relatives_history:
            daily_return = np.dot(portfolio, price_relatives)
            P_t *= daily_return

        # Calculate damping factor
        b_n = portfolio[-1]  # Last component
        damping_exp = (b_n - 2 * self.delta_0) / (self.n * self.delta)
        damping = min(np.exp(damping_exp), 1.0)

        return P_t * damping

    def _is_valid_portfolio(self, portfolio: np.ndarray) -> bool:
        """Check if portfolio satisfies constraints"""
        # All coordinates >= delta_0
        if np.any(portfolio < self.delta_0):
            return False
        # Sum to 1 (not exactly, with some tolerance)
        if not np.isclose(np.sum(portfolio), 1.0):
            return False
        return True

    def _random_walk_step(self, r: np.ndarray) -> np.ndarray:
        """
        Performing one step of the random walk
        """
        # Choose random coordinate j (from 0 to n-2)
        j = np.random.randint(0, self.n - 1)

        # Choose direction X in {-1, +1}
        X = np.random.choice([-1, 1])

        # Check if move is valid
        new_r = r.copy()
        new_r[j] += X * self.delta
        new_r[-1] -= X * self.delta  # Maintain sum = 1

        # Check constraints
        if new_r[j] >= self.delta_0 and new_r[-1] >= self.delta_0:
            # Calculate acceptance probability
            x = self.Q_t(r)
            y = self.Q_t(new_r)

            if y > 0:
                acceptance_prob = min(1.0, x / y)
            else:
                acceptance_prob = 0.0

            # Accept or reject
            if np.random.random() < acceptance_prob:
                return new_r

        return r