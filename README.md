# Online Trading

## Project Structure

```
.
├── algorithms/
│   ├── portfolio/
│   │   └── r_universal.py                    # R-Universal Portfolio algorithm
│   └── threshold/
│       ├── adaptive_threshold.py             # Adaptive threshold trader using percentiles
│       └── local_extrema_volatility.py       # Extrema-based volatility-normalized trader
│
├── benchmarks/
│   ├── portfolio/
│   │   ├── buy_and_hold.py                   # Buy and hold strategy
│   │   ├── constant_rebalanced_portfolio.py  # Constant rebalanced portfolio (CRP)
│   │   ├── best_constant_rebalanced_portfolio.py  # Best CRP in hindsight
│   │   └── compare_strategies.py             # Strategy comparison utilities
│   └── threshold/
│       └── oracle_trader.py                  # Hindsight optimal trading (perfect foresight)
│
├── data_generators/
│   └── simple_data_generator.py              # Synthetic oscillating stock data generator
│
├── experiments/
│   ├── portfolio/
│   │   └── universal_portfolio_vs_benchmarks.py  # Compare universal portfolio algorithms
│   └── threshold/
│       └── adaptive_vs_extrema_traders.py    # Compare threshold trading strategies
│
└── plots/
    ├── portfolio/                            # Portfolio experiment visualizations
    └── threshold/                            # Threshold trading visualizations
```

## Categories

The project is divided into two main problems:

### Portfolio Management
Universal Portfolio based solution(s)

**Algorithms:**
- R-Universal Portfolio: uses Cover's universal portfolio framework

**Benchmarks:**
- Buy and Hold: Invest once at the beginning and hold
- Constant Rebalanced Portfolio (CRP): Maintain fixed asset allocation through rebalancing
- Best CRP: Optimal fixed allocation found in hindsight

### Threshold-Based Trading
For simplicity now only single asset cases are considered. Algorithms should find threshold depending on which the decision of buy/sell will be done


**Algorithms:**
- Adaptive Threshold Trader: Learns buy/sell thresholds from percentiles of historical price changes, adapts using exponential smoothing
- Local Extrema Volatility Trader: Learns thresholds from price movements at local minima/maxima normalizes by volatility

**Benchmarks:**
- Oracle Trader: Buys at every local minimum and sells at every local maximum

## Usage

Each experiment script in `experiments/` can be run independently to compare strategies and generate visualizations saved to `plots/`.

## Dependencies

Project Python v. 3.13.5.

To install required packages:

```bash
pip install -r requirements.txt
```