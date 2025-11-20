import numpy as np


class Barrons:
    def __init__(self, n_stocks, beta, eta, T):
        self.n_stocks = n_stocks
        self.beta = beta
        self.eta = eta
        self.T = T

        self.x_history = []
        self.loss_history = []

        self._initialize()
        self._check_config()
        self._print_config()

    def _initialize(self):
        self.x_t = np.ones(self.n_stocks) / self.n_stocks

        # Define restricted simplex: x_i >= 1/(N*T)
        self.x_min = 1 / (self.n_stocks * self.T)

        self.A_t = self.n_stocks * np.eye(self.n_stocks)

        self.x_history.append(self.x_t)

    def _check_config(self):
        if self.beta <= 0 or self.beta > 0.5:
            raise ValueError("Beta must be between 0 and 1/2")

        if self.eta <= 0 or self.eta > 1:
            raise ValueError("Eta must be between 0 and 1")

    def _print_config(self):
        print(f"BARRONS Configuration")
        print(f"Beta: {self.beta}")
        print(f"Eta: {self.eta}")
        print(f"N stocks: {self.n_stocks}")
        print(f"Rounds (T): {self.T}")

    def _update_eta_t(self):
        past_x = np.array(self.x_history)

        max_term = np.max(np.log(1 / (self.n_stocks * past_x)) / np.log(self.T), axis=0)
        self.eta_t = self.eta * np.exp(max_term)

    def _compute_gradient(self, r_t):
        return -r_t / np.dot(self.x_t, r_t)

    def _update_matrix_a(self, gradient):
        self.A_t += np.outer(gradient, gradient)

    def _phi_t_regularized(self, x):
        x = np.asarray(x, dtype=float)
        x = np.maximum(x, self.x_min)

        first_term = 0.5 * self.beta * float(x.T @ (self.A_t @ x))
        second_term = np.sum((1 / self.eta_t) * np.log(1.0 / x))

        return first_term + second_term

    def _gradient_phi_t_regularized(self, x):
        x = np.asarray(x, dtype=float)

        first_term = self.beta * (self.A_t @ x)
        second_term = -(1 / self.eta_t) * (1 / x)

        return first_term + second_term

    def _mirror_descent_update(self, gradient):
        pass
