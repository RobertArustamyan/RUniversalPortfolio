import numpy as np
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

        self.last_portfolio = np.ones(n_stocks) / n_stocks  # Uniform distribution
        self.price_relatives_history = []

        self.__calculate_params()


    @staticmethod
    def __find_delta_binary_search(value):
        """Binary search fallback for finding delta"""
        low, high = 1e-10, 0.368  # max of delta*log(1/delta) is at 1/e

        for _ in range(100):
            mid = (low + high) / 2
            if mid <= 0:
                low = mid
                continue

            f_mid = mid * np.log(1.0 / mid)

            if f_mid < value:
                low = mid
            else:
                high = mid

        return (low + high) / 2

    def __calculate_params(self):
        # From theorem 2 (Analysis Section)
        n = self.n
        T = self.T
        epsilon = self.epsilon
        eta = self.eta

        # delta_0 <= epsilon / (8*n*T*(n+T)^2)
        self.delta_0 = epsilon / (8 * n * T * (n + T) ** 2)

        A = 100  # Constant from the paper (they say "there is a constant A")

        # delta * log(1/delta) = epsilon*delta_0 / (A*(n+T)^2)
        target_value = (epsilon * self.delta_0) / (A * (n + T) ** 2)


        self.delta = self.__find_delta_binary_search(target_value)

        # m >= 64*T^2*(n+T)*ln(nT/eta) / epsilon^2
        self.m = int(np.ceil(64 * T ** 2 * (n + T) * np.log(n * T / eta) / epsilon ** 2))

        # S >= A*n/delta^2 * log((n+T)/(epsilon*delta))
        self.S = int(np.ceil(A * n / self.delta ** 2 *
                             np.log((n + T) / (epsilon * self.delta))))

        print(f"\nParameters calculated:")
        print(f"  delta_0 = {self.delta_0:.6e}")
        print(f"  delta   = {self.delta:.6e}")
        print(f"  m       = {self.m:,}")
        print(f"  S       = {self.S:,}")


if __name__ == "__main__":
    print("\nExample 1: (3 stocks, 10 days)")
    portfolio1 = RUniversalPortfolio(n_stocks=3, epsilon=0.2, eta=0.1, T=10)
    print('\n' + '-' * 40)

    print("\nExample 2: (10 stocks, 50 days)")
    portfolio2 = RUniversalPortfolio(n_stocks=10, epsilon=0.1, eta=0.05, T=50)
    print('\n' + '-' * 40)

    print("\nExample 3: (100 stocks, 100 days)")
    portfolio3 = RUniversalPortfolio(n_stocks=100, epsilon=0.01, eta=0.01, T=100)
    print('\n' + '-' * 40)