# QUANTMIND — Quant Trading Interview Prep

A portfolio of **3 production-ready projects** covering probability theory, Monte Carlo simulation, and algorithmic backtesting — core skills for quantitative trading firms.

**Designed for: Quant internships**
**Author:** MARWANE SARAJ (EIDIA, Fès)  
**Last updated:** September 2026

---

## 📊 Projects Overview

### 1️⃣ Probability Puzzle Simulations
**File:** `01_probability_puzzles.py`  

20 classic quant interview problems solved via Monte Carlo simulation + analytical formulas:
- Monty Hall, Birthday Paradox, Gambler's Ruin
- Coupon Collector, Secretary Problem, St. Petersburg Paradox
- Bayes' Theorem, Ballot Problem, Buffon's Needle, and more

**Result:** Every simulation converges to <1% error vs analytical solution.

```bash
python 01_probability_puzzles.py
```

---

### 2️⃣ Monte Carlo Options Pricer
**File:** `02_monte_carlo_simulator.py`  

Geometric Brownian Motion simulation + Black-Scholes pricing:
- 10,000 asset price paths
- European call/put option pricing
- Greeks computation (Delta, Gamma, Vega, Theta, Rho)
- Put-call parity verification

**Result:** MC Call: $10.22 | B-S Call: $10.45 (error: 2.2%)

```bash
python 02_monte_carlo_simulator.py
```

---

### 3️⃣ Algorithmic Backtesting Engine
**File:** `03_backtest_engine.py`  

Mean-reversion strategy backtester with full risk metrics:
- Bollinger Band entry/exit signals
- Transaction costs (5 bps per side)
- Performance metrics: Sharpe, Sortino, Calmar, Max Drawdown, Kelly Criterion
- Rolling window analysis
- Equity curve vs Buy & Hold

```bash
python 03_backtest_engine.py
```

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python 01_probability_puzzles.py
python 02_monte_carlo_simulator.py
python 03_backtest_engine.py
```

---

## 📝 License

MIT License — Use freely for learning and interviews.
