import numpy as np
from scipy.optimize import minimize


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

        self.x_history.append(self.x_t.copy())

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

        first_term = 0.5 * self.beta * float(x.T @ (self.A_t @ x))
        second_term = np.sum((1 / self.eta_t) * np.log(1.0 / x))

        return first_term + second_term

    def _gradient_phi_t_regularized(self, x):
        x = np.asarray(x, dtype=float)

        first_term = self.beta * (self.A_t @ x)
        second_term = -(1 / self.eta_t) * (1 / x)

        return first_term + second_term

    def _bregman_divergence(self, x, y):
        phi_x = self._phi_t_regularized(x)
        phi_y = self._phi_t_regularized(y)
        grad_phi_y = self._gradient_phi_t_regularized(y)

        return phi_x - phi_y - np.dot(grad_phi_y, (x - y))

    def _x_update(self, gradient):
        def objective(x):
            return np.dot(x,gradient) + self._bregman_divergence(x, self.x_t)

        bounds = [(self.x_min, 1.0) for _ in range(self.n_stocks)] # All values should be at least x_min

        constraints = {'type': 'eq', 'fun': lambda x: sum(x) - 1} # Constraint for sum of values (to be equal to 1)

        x0 = self.x_t.copy()

        result = minimize(objective, x0, bounds=bounds, constraints=constraints)

        if not result.success:
            print("Solver was not able to converge")
            # TODO
            # Implement some fallback logic
        return result.x

    def update(self, r_t):
        # Step 1: compute gradient of the loss
        current_loss = -np.log(np.dot(self.x_t, r_t))
        loss_gradient = self._compute_gradient(r_t)
        # Step 2: update matrix A
        self._update_matrix_a(loss_gradient)
        # Step 3: update eta's
        self._update_eta_t()

        # Step 4: calculate x_{t+1}
        x_next = self._x_update(loss_gradient)

        self.x_t = x_next
        self.x_history.append(self.x_t.copy())
        self.loss_history.append(current_loss)

        return self.x_t.copy()

    def simulate_trading(self, price_relatives_sequence, verbose=True):
        wealth = 1.0
        daily_wealth = [1.0]
        portfolios_used = []

        for day, r_t in enumerate(price_relatives_sequence):
            portfolio = self.x_t.copy()
            portfolios_used.append(portfolio.copy())

            # wealth update based on portfolio chosen before seeing r_t
            daily_return = np.dot(portfolio, r_t)
            wealth *= daily_return
            daily_wealth.append(wealth)

            # update portfolio for next round
            self.update(r_t)

            if verbose and ((day + 1) % 10 == 0 or day == 0):
                print(f"Day {day + 1}: Wealth = {wealth:.4f}")


        return {
            'final_wealth': wealth,
            'daily_wealth': np.array(daily_wealth),
            'portfolios_used': np.array(portfolios_used),
            'num_days': len(price_relatives_sequence)
        }