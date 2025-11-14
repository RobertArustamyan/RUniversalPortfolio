import numpy as np

def hindsight_optimal_wealth_onsale_updated(stock_price):
    """Calculate hindsight optimal wealth"""
    wealth = 1.0
    daily_wealth = [1.0]
    i = 1
    holding = False
    entry_price = 0

    while i < len(stock_price) - 1:
        # Buying when the price is minimal and sell when it is maximal
        if stock_price[i] < stock_price[i - 1] and stock_price[i] < stock_price[i + 1] and not holding:
            entry_price = stock_price[i]
            holding = True
        elif stock_price[i] > stock_price[i - 1] and stock_price[i] > stock_price[i + 1] and holding:
            exit_price = stock_price[i]
            profit_ratio = exit_price / entry_price
            wealth *= profit_ratio
            holding = False

        daily_wealth.append(wealth)
        i += 1

    daily_wealth.append(wealth)
    if holding:
        profit_ratio = stock_price[-1] / entry_price
        wealth *= profit_ratio
        daily_wealth[-1] = wealth

    return wealth, np.array(daily_wealth)


def hindsight_optimal_wealth_daily_updated(stock_price):
    """Calculate hindsight optimal wealth with daily mark-to-market"""
    wealth = 1.0
    daily_wealth = [1.0]
    i = 1
    holding = False
    entry_price = 0
    shares = 0

    while i < len(stock_price) - 1:
        # Buy at local minimum
        if stock_price[i] < stock_price[i - 1] and stock_price[i] < stock_price[i + 1] and not holding:
            entry_price = stock_price[i]
            shares = wealth / entry_price
            holding = True
        # Sell at local maximum
        elif stock_price[i] > stock_price[i - 1] and stock_price[i] > stock_price[i + 1] and holding:
            wealth = shares * stock_price[i]
            shares = 0
            holding = False

        if holding:
            daily_wealth.append(shares * stock_price[i])
        else:
            daily_wealth.append(wealth)

        i += 1

    # Final day
    if holding:
        wealth = shares * stock_price[-1]
    daily_wealth.append(wealth)

    return wealth, np.array(daily_wealth)
