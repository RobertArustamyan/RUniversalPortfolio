import numpy as np
import matplotlib.pyplot as plt


def generate_stock_data(initial_value, total_steps,
                        min_oscillation_steps=1, max_oscillation_steps=10,
                        min_flat_steps=0, max_flat_steps=10,
                        step_size=1):
    """
    Simple data generator with alterning up/down movements

    - initial_value: Starting stock price
    - total_steps: Total number of time steps
    - min_oscillation_steps: Minimum steps to move away from initial value
    - max_oscillation_steps: Maximum steps to move away from initial value
    - min_flat_steps: Minimum steps to stay flat at initial value
    - max_flat_steps: Maximum steps to stay flat at initial value
    - step_size: Amount to change per step (+/- this value)
    """
    prices = [initial_value]
    current_value = initial_value
    t = 0

    # Random wait time
    initial_wait = np.random.randint(min_flat_steps, max_flat_steps + 1)

    # Stay at initial value for first wait time
    for _ in range(min(initial_wait, total_steps - 1)):
        prices.append(initial_value)
        t += 1


    direction = np.random.choice([1, -1])

    while t < total_steps - 1:
        # Duration of oscillation
        oscillation_steps = np.random.randint(min_oscillation_steps, max_oscillation_steps + 1)

        # Move in selected direction
        for _ in range(min(oscillation_steps, total_steps - 1 - t)):
            current_value += direction * step_size
            prices.append(current_value)
            t += 1
            if t >= total_steps - 1:
                break

        # Return to initial value
        while not np.isclose(current_value, initial_value) and t < total_steps - 1:
            if current_value > initial_value:
                current_value -= step_size
            else:
                current_value += step_size
            prices.append(current_value)
            t += 1

        current_value = initial_value

        # Random flat period at initial value
        flat_steps = np.random.randint(min_flat_steps, max_flat_steps + 1)
        for _ in range(min(flat_steps, total_steps - 1 - t)):
            prices.append(initial_value)
            t += 1
            if t >= total_steps - 1:
                break

        # Reverse direction for next movement
        direction *= -1

    return np.array(prices)

def prices_to_price_relatives(prices):
    """Convert price series to price relatives"""
    price_relatives = prices[1:] / prices[:-1]

    price_relatives = np.nan_to_num(price_relatives, nan=1.0, posinf=1.0, neginf=1.0)
    return price_relatives

