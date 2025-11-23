import numpy as np
import cvxpy as cp


class Barrons:
    def __init__(self, n_stocks, T, beta, eta):
        self.n_stocks = n_stocks
        self.T = T
        self.eta = eta
        self.beta = beta
        self._check_parameters()

        self.x_min = 1 / (n_stocks * T)
        self.x_0 = np.ones(n_stocks) / n_stocks
        # Current portfolio state
        self.x_t = self.x_0.copy()
        self.A = np.eye(n_stocks) * n_stocks

        # History
        self.observed_loss = []
        self.portfolios_used = []

    def _check_parameters(self):
        if self.beta <= 0 or self.beta > 0.5:
            raise ValueError(f"Invalid beta value: {self.beta}")
        if self.eta <= 0 or self.eta > 1:
            raise ValueError(f"Invalid eta value: {self.eta}")

    def get_portfolio(self):
        """Return portfolio for trading"""
        return self.x_t

    def _calculate_loss(self, r_t):
        return -np.log(np.dot(self.x_t, r_t))

    def _compute_eta_t(self):
        portfolio_hist = np.array(self.portfolios_used)
        values = 1.0 / (self.n_stocks * portfolio_hist)

        log_T_values = np.log(values) / np.log(self.T)
        max_values = np.max(log_T_values, axis=0)

        return self.eta * np.exp(max_values)

    def _psi(self, x):
        x = np.asarray(x)

        first_term = (0.5 * self.beta) * (x @ self.A @ x)
        second_term = np.sum((1.0 / self.eta_t) * np.log(1.0 / x))

        return first_term + second_term

    def _grad_psi(self, x):
        x = np.asarray(x)

        first_term = self.beta * (self.A @ x)
        second_term = -1.0 / (self.eta_t * x)

        return first_term + second_term

    def _bregman_divergence(self, x, y):
        grad_y = self._grad_psi(y)
        psi_x = self._psi(x)
        psi_y = self._psi(y)

        return psi_x - psi_y - np.dot(grad_y, x - y)

    # def _solve_ipm(self, grad_t):
    #     n = self.n_stocks
    #     x = cp.Variable(n)
    #
    #     # gradient of psi at x_t (numpy vector!)
    #     grad_psi_xt = self._grad_psi(self.x_t)
    #
    #     # Build ψ(x) in CVXPY form:
    #     quad_term = 0.5 * self.beta * cp.quad_form(x, self.A)
    #     ent_term = cp.sum(cp.multiply((1.0 / self.eta_t), cp.log(1.0 / x)))
    #     psi_expr = quad_term + ent_term
    #
    #     # Linearization term:
    #     linear_term = grad_t - grad_psi_xt  # constant vector
    #     objective = cp.Minimize(linear_term @ x + psi_expr)
    #
    #     constraints = [cp.sum(x) == 1, x >= self.x_min]
    #
    #     prob = cp.Problem(objective, constraints)
    #     prob.solve(solver=cp.ECOS,
    #                warm_start=True,
    #                abstol=1e-8, reltol=1e-6, feastol=1e-8)
    #
    #     if x.value is None:
    #         raise RuntimeError("IPM solver failed")
    #
    #     # Projection safety
    #     x_val = np.maximum(x.value, self.x_min)
    #     x_val /= x_val.sum()
    #     return x_val

    def _solve_ipm(self, grad_t):
        n = self.n_stocks
        x = cp.Variable(n)

        grad_psi_xt = self._grad_psi(self.x_t)

        quad_term = 0.5 * self.beta * cp.quad_form(x, self.A)
        ent_term_lin = cp.sum(cp.multiply(-1.0 / (self.eta_t * self.x_t), x - self.x_t))
        psi_expr = quad_term + ent_term_lin

        linear_term = grad_t - grad_psi_xt
        objective = cp.Minimize(linear_term @ x + psi_expr)

        constraints = [cp.sum(x) == 1, x >= self.x_min]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CVXOPT, verbose=False)

        if x.value is None:
            raise RuntimeError("Solver failed")

        x_val = np.maximum(x.value, self.x_min)
        x_val /= x_val.sum()
        return x_val

    def update(self, r_t):
        x_used = self.portfolios_used[-1]
        grad_t = -r_t / np.dot(x_used, r_t)

        self.A += np.outer(grad_t, grad_t)
        self.eta_t = self._compute_eta_t()

        self.x_t = self._solve_ipm(grad_t)


    def simulate_trading(self, price_relatives_sequence, verbose=True, verbose_days=100):
        wealth = 1.0
        daily_wealth = [1.0]

        for day, r_t in enumerate(price_relatives_sequence):
            # Get portfolio for current day
            portfolio = self.get_portfolio()
            self.portfolios_used.append(portfolio.copy())

            loss = self._calculate_loss(r_t)
            self.observed_loss.append(loss.copy())

            daily_return = np.dot(portfolio, r_t)
            wealth *= daily_return
            daily_wealth.append(wealth)

            # Update for next day
            self.update(r_t)
            if verbose and ((day + 1) % verbose_days == 0 or day == 0):
                print(f"Day {day + 1}: Wealth = {wealth:.4f}, "
                      f"Portfolio = [{', '.join([f'{x:.3f}' for x in portfolio])}]")

        return {
            'final_wealth': wealth,
            'daily_wealth': np.array(daily_wealth),
            'portfolios_used': np.array(self.portfolios_used),
            'num_days': len(price_relatives_sequence)
        }