import numpy as np
import matplotlib.pyplot as plt

# Import your classes
from r_universal import RUniversalPortfolio
from data_generators.simple_data_generator import generate_stock_data, prices_to_price_relatives
from benchmarks.compare_strategies import compare_strategies


def simulate_threshold_strategy(risky_prices, buy_threshold, sell_threshold):
    """Simulate threshold strategy, return wealth history"""
    wealth = 1.0
    daily_wealth = [1.0]
    holding_risky = False
    entry_price = 0

    for price in risky_prices[1:]:
        if not holding_risky and price <= buy_threshold:
            holding_risky = True
            entry_price = price
        elif holding_risky and price >= sell_threshold:
            profit_ratio = price / entry_price
            wealth *= profit_ratio
            holding_risky = False

        daily_wealth.append(wealth)

    if holding_risky:
        profit_ratio = risky_prices[-1] / entry_price
        wealth *= profit_ratio
        daily_wealth[-1] = wealth

    return wealth, np.array(daily_wealth)


def find_best_threshold(train_risky_prices, threshold_range):
    """Find best buy/sell thresholds on training data"""
    best_buy, best_sell, best_wealth = 0, 0, 0

    for buy_th in threshold_range:
        for sell_th in threshold_range:
            if sell_th > buy_th:
                wealth, _ = simulate_threshold_strategy(train_risky_prices, buy_th, sell_th)
                if wealth > best_wealth:
                    best_wealth = wealth
                    best_buy = buy_th
                    best_sell = sell_th

    return best_buy, best_sell, best_wealth


def hindsight_optimal_wealth(risky_prices):
    """Calculate hindsight optimal wealth"""
    wealth = 1.0
    daily_wealth = [1.0]
    i = 1
    holding = False
    entry_price = 0

    while i < len(risky_prices) - 1:
        if risky_prices[i] < risky_prices[i - 1] and risky_prices[i] < risky_prices[i + 1] and not holding:
            entry_price = risky_prices[i]
            holding = True
        elif risky_prices[i] > risky_prices[i - 1] and risky_prices[i] > risky_prices[i + 1] and holding:
            profit_ratio = risky_prices[i] / entry_price
            wealth *= profit_ratio
            holding = False

        daily_wealth.append(wealth)
        i += 1

    daily_wealth.append(wealth)

    if holding:
        profit_ratio = risky_prices[-1] / entry_price
        wealth *= profit_ratio
        daily_wealth[-1] = wealth

    return wealth, np.array(daily_wealth)


# Main simulation
if __name__ == "__main__":
    np.random.seed(42)

    # Parameters
    train_days = 365
    test_days = 30
    total_days = train_days + test_days

    print("=" * 60)
    print("GENERATING STOCK DATA")
    print("=" * 60)

    # Generate ONE continuous stock for entire period
    risky_stock = generate_stock_data(
        initial_value=100,
        total_steps=total_days,
        min_oscillation_steps=2,
        max_oscillation_steps=8,
        min_flat_steps=1,
        max_flat_steps=5,
        step_size=0.5
    )

    # Safe stock (constant = 100)
    safe_stock = np.ones(total_days) * 100

    # Split into train and test
    train_risky = risky_stock[:train_days]
    test_risky = risky_stock[train_days:]

    # Combine stocks into price matrix
    all_prices = np.column_stack([risky_stock, safe_stock])

    # Convert to price relatives
    all_price_relatives = prices_to_price_relatives(all_prices)
    train_price_relatives = all_price_relatives[:train_days - 1]
    test_price_relatives = all_price_relatives[train_days - 1:]

    print(f"\nTotal days: {total_days} (Train: {train_days}, Test: {test_days})")
    print(f"Risky stock range: [{risky_stock.min():.2f}, {risky_stock.max():.2f}]")

    # ==================================================================
    # 1. R-UNIVERSAL PORTFOLIO
    # ==================================================================
    print("\n" + "=" * 60)
    print("R-UNIVERSAL PORTFOLIO")
    print("=" * 60)

    # Initialize R-Universal with manual parameters
    runiversal = RUniversalPortfolio(
        n_stocks=2,
        m=50,  # Number of samples
        S=30,  # Random walk steps
        delta=0.01,  # Grid spacing
        delta_0=0.001,  # Minimum coordinate
        use_damping=True,
        use_parallel=False
    )

    # Run on ALL data (train + test)
    print("\nRunning R-Universal on ALL data...")
    runiversal_results = runiversal.simulate_trading(all_price_relatives, verbose=True)

    print(f"\nR-Universal Final Wealth: ${runiversal_results['final_wealth']:.4f}")

    # ==================================================================
    # 2. THRESHOLD STRATEGY
    # ==================================================================
    print("\n" + "=" * 60)
    print("THRESHOLD STRATEGY")
    print("=" * 60)

    # Find best thresholds on training data
    print("Finding best thresholds on TRAIN data...")
    threshold_range = np.arange(95, 105, 0.5)
    best_buy, best_sell, best_train_wealth = find_best_threshold(train_risky, threshold_range)

    print(f"Best buy threshold: {best_buy:.2f}")
    print(f"Best sell threshold: {best_sell:.2f}")
    print(f"Training wealth: {best_train_wealth:.4f}")

    # Apply on ALL data (train + test)
    print("\nApplying on ALL data...")
    threshold_wealth, threshold_history = simulate_threshold_strategy(
        risky_stock, best_buy, best_sell
    )

    threshold_results = {
        'strategy': 'Threshold Strategy',
        'daily_wealth': threshold_history,
        'final_wealth': threshold_wealth
    }

    print(f"Threshold Final Wealth: ${threshold_wealth:.4f}")

    # ==================================================================
    # 3. HINDSIGHT OPTIMAL
    # ==================================================================
    print("\n" + "=" * 60)
    print("HINDSIGHT OPTIMAL")
    print("=" * 60)

    # Calculate on all data
    hindsight_wealth, hindsight_history = hindsight_optimal_wealth(risky_stock)

    hindsight_results = {
        'strategy': 'Hindsight Optimal',
        'daily_wealth': hindsight_history,
        'final_wealth': hindsight_wealth
    }

    print(f"Hindsight Optimal Wealth: ${hindsight_wealth:.4f}")

    # ==================================================================
    # 4. COMPARE ALL STRATEGIES USING YOUR FUNCTION
    # ==================================================================
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    print(f"\nR-Universal:  ${runiversal_results['final_wealth']:.4f}  "
          f"({runiversal_results['final_wealth'] / hindsight_wealth * 100:.1f}% of optimal)")
    print(f"Threshold:    ${threshold_wealth:.4f}  "
          f"({threshold_wealth / hindsight_wealth * 100:.1f}% of optimal)")
    print(f"Hindsight:    ${hindsight_wealth:.4f}  (100% - theoretical best)")

    # Use your compare_strategies function
    all_results = compare_strategies(
        all_price_relatives,
        stock_names=['Risky Stock', 'Safe Stock'],
        show_plot=True,
        save_path='portfolio_comparison.png',
        additional_results=[
            runiversal_results,
            threshold_results,
            hindsight_results
        ]
    )

    # Additional visualization: Stock prices with thresholds
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Stock prices
    axes[0].plot(risky_stock, linewidth=1.5, label='Risky Stock', color='blue', alpha=0.7)
    axes[0].axhline(y=best_buy, color='g', linestyle='--', label=f'Buy: {best_buy:.1f}', alpha=0.7)
    axes[0].axhline(y=best_sell, color='r', linestyle='--', label=f'Sell: {best_sell:.1f}', alpha=0.7)
    axes[0].axvline(x=train_days, color='black', linestyle=':', label='Train/Test Split', alpha=0.5)
    axes[0].set_xlabel('Day')
    axes[0].set_ylabel('Price')
    axes[0].set_title('Risky Stock Prices with Threshold Levels')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Wealth comparison (focused view)
    axes[1].plot(runiversal_results['daily_wealth'], linewidth=2,
                 label='R-Universal', color='purple')
    axes[1].plot(threshold_history, linewidth=2,
                 label='Threshold', color='orange')
    axes[1].plot(hindsight_history, linewidth=2,
                 label='Hindsight Optimal', color='green', linestyle='--', alpha=0.7)
    axes[1].axvline(x=train_days, color='black', linestyle=':',
                    label='Train/Test Split', alpha=0.5)
    axes[1].axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    axes[1].set_xlabel('Day')
    axes[1].set_ylabel('Wealth')
    axes[1].set_title('Wealth Accumulation: All Strategies')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('detailed_comparison.png', dpi=300, bbox_inches='tight')
    print("\nDetailed plot saved to: detailed_comparison.png")
    plt.show()