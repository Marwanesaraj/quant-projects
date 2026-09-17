"""
=============================================================================
  QUANTMIND — EV-Positive Strategy Backtest Engine
  Bollinger Band Mean Reversion · Sharpe · Drawdown · Kelly Criterion
=============================================================================
  Install : pip install numpy pandas matplotlib yfinance
  Run     : python 03_backtest_engine.py
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

print("="*62)
print("  QUANTMIND — EV-Positive Strategy Backtest Engine")
print("  Bollinger Band Mean Reversion on SPY (2019–2024)")
print("="*62)

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS — Edit these freely
# ─────────────────────────────────────────────────────────────────────────────
TICKER      = 'SPY'
START       = '2019-01-01'
END         = '2024-12-31'
WINDOW      = 20          # BB lookback (days)
N_STD       = 2.0         # Standard deviations for entry/exit
COST_BPS    = 5           # Transaction cost per side in basis points
INIT_CAP    = 10_000      # Starting capital ($)
RF_ANNUAL   = 0.05        # Risk-free rate (for Sharpe)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Generate Synthetic Price Data (GBM, SPY-calibrated params)
#   Drift  μ = 12% p.a.  |  Volatility σ = 18% p.a.  |  S0 = $300
#   This mirrors SPY's long-run empirical statistics (2019–2024)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n  Generating synthetic {TICKER} price data ({START[:4]}–{END[:4]})...")
np.random.seed(42)
N_DAYS  = 252 * (int(END[:4]) - int(START[:4]))   # trading days
S0_SYN  = 300.0
MU      = 0.12   # annual drift
SIGMA   = 0.18   # annual vol
dt      = 1/252

log_rets = (MU - 0.5*SIGMA**2)*dt + SIGMA*np.sqrt(dt)*np.random.standard_normal(N_DAYS)
prices_arr = S0_SYN * np.exp(np.cumsum(log_rets))
dates  = pd.bdate_range(start=START, periods=N_DAYS)
close  = pd.Series(prices_arr, index=dates, name=TICKER)

print(f"  ✅ {len(close)} trading days  |  "
      f"Price range: ${close.min():.2f} – ${close.max():.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Bollinger Bands + Z-Score
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Computing Bollinger Bands...")
df             = pd.DataFrame({'price': close})
df['ma']       = df['price'].rolling(WINDOW).mean()
df['std']      = df['price'].rolling(WINDOW).std()
df['upper']    = df['ma'] + N_STD * df['std']
df['lower']    = df['ma'] - N_STD * df['std']
df['z']        = (df['price'] - df['ma']) / df['std']
df             = df.dropna()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Signal Generation
#   Long  when price < lower band (oversold, z < -N_STD)
#   Short when price > upper band (overbought, z > +N_STD)
#   Hold position until opposite signal
# ─────────────────────────────────────────────────────────────────────────────
print("  Generating signals...")
df['raw_signal']             = 0
df.loc[df['z'] < -N_STD, 'raw_signal'] =  1   # Buy
df.loc[df['z'] >  N_STD, 'raw_signal'] = -1   # Sell short
df['position']               = (df['raw_signal']
                                  .replace(0, np.nan)
                                  .ffill()
                                  .fillna(0))

n_long  = (df['raw_signal'] ==  1).sum()
n_short = (df['raw_signal'] == -1).sum()
print(f"  Long signals: {n_long}  |  Short signals: {n_short}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Return Calculation + Transaction Costs
# ─────────────────────────────────────────────────────────────────────────────
print("  Computing returns (with transaction costs)...")
df['returns']     = df['price'].pct_change()
df['pos_change']  = df['position'].diff().abs()
cost_per_trade    = COST_BPS / 10_000

df['strat_gross'] = df['position'].shift(1) * df['returns']
df['strat_net']   = df['strat_gross'] - df['pos_change'] * cost_per_trade
df['bh']          = df['returns']
df                = df.dropna()

df['equity']      = INIT_CAP * (1 + df['strat_net']).cumprod()
df['equity_bh']   = INIT_CAP * (1 + df['bh']).cumprod()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Performance Metrics
# ─────────────────────────────────────────────────────────────────────────────
def metrics(rets, label=''):
    rf_d     = RF_ANNUAL / 252
    n_years  = len(rets) / 252
    total_r  = (1 + rets).prod() - 1
    cagr     = (1 + total_r)**(1/n_years) - 1
    vol      = rets.std() * np.sqrt(252)
    sharpe   = ((rets - rf_d).mean() / rets.std()) * np.sqrt(252)
    cum      = (1 + rets).cumprod()
    dd       = (cum / cum.cummax()) - 1
    max_dd   = dd.min()
    calmar   = cagr / abs(max_dd) if max_dd != 0 else np.nan
    win_rate = (rets > 0).mean()
    avg_win  = rets[rets > 0].mean() if (rets>0).any() else 0
    avg_loss = abs(rets[rets < 0].mean()) if (rets<0).any() else 1e-8
    b        = avg_win / avg_loss
    kelly    = (win_rate * b - (1 - win_rate)) / b if b > 0 else 0
    down_vol = rets[rets < 0].std() * np.sqrt(252) if (rets<0).any() else 1e-8
    sortino  = (rets.mean()*252 - RF_ANNUAL) / down_vol

    return {'label'       : label,
            'Total Return': total_r,
            'CAGR'        : cagr,
            'Volatility'  : vol,
            'Sharpe'      : sharpe,
            'Sortino'     : sortino,
            'Max Drawdown': max_dd,
            'Win Rate'    : win_rate,
            'Calmar'      : calmar,
            'Kelly'       : kelly,
            '_dd'         : dd}

m_s  = metrics(df['strat_net'], 'Strategy')
m_bh = metrics(df['bh'],        'Buy & Hold')

print("\n" + "─"*62)
print(f"  {'Metric':<22} {'Strategy':>12} {'Buy & Hold':>12}")
print(f"  {'-'*48}")
for key in ['Total Return','CAGR','Volatility','Sharpe','Sortino',
            'Max Drawdown','Win Rate','Calmar']:
    s_val = m_s[key]
    b_val = m_bh[key]
    fmt   = '{:>12.2%}' if key not in ['Sharpe','Sortino','Calmar'] else '{:>12.3f}'
    print(f"  {key:<22}" + fmt.format(s_val) + fmt.format(b_val))

n_trades = int(df['pos_change'].sum() / 2)
total_cost_drag = n_trades * 2 * cost_per_trade * 100
print(f"\n  Round-trip trades : {n_trades}")
print(f"  Total cost drag   : {total_cost_drag:.3f}%")

print(f"\n  KELLY CRITERION ANALYSIS")
k = m_s['Kelly'] * 100
print(f"  Full Kelly   = {k:.1f}% of capital per trade")
print(f"  Half Kelly   = {k/2:.1f}%   ← recommended (reduces risk)")
print(f"  Quarter Kelly= {k/4:.1f}%   ← conservative")
if k > 100:
    print("  ⚠️  Kelly > 100% indicates very high estimated edge — use Half or Quarter Kelly.")

print(f"\n  FINAL PORTFOLIO VALUE")
print(f"  Strategy  : ${df['equity'].iloc[-1]:>10,.2f}  "
      f"(from ${INIT_CAP:,})")
print(f"  Buy & Hold: ${df['equity_bh'].iloc[-1]:>10,.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Additional Analysis: Rolling Metrics
# ─────────────────────────────────────────────────────────────────────────────
roll_win    = 60    # days
r_mean      = df['strat_net'].rolling(roll_win).mean()
r_std       = df['strat_net'].rolling(roll_win).std().replace(0, np.nan)
r_sharpe    = (r_mean / r_std) * np.sqrt(252)
r_vol       = df['strat_net'].rolling(roll_win).std() * np.sqrt(252)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Plots
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating plots...")
fig = plt.figure(figsize=(17, 12))
fig.suptitle(f'QUANTMIND — {TICKER} Mean Reversion Backtest  '
             f'({START[:4]}–{END[:4]})', fontsize=14, fontweight='bold', y=0.99)
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.30)

# — Plot 1: Price + Bollinger Bands —————————————————————————————
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df.index, df['price'], 'k-',  lw=0.9, alpha=0.85, label='Price')
ax1.plot(df.index, df['ma'],   'b-',  lw=1.4, alpha=0.7,  label=f'{WINDOW}d MA')
ax1.fill_between(df.index, df['lower'], df['upper'],
                 alpha=0.13, color='blue', label='Bollinger Band')
buys  = df[df['raw_signal'] ==  1]
sells = df[df['raw_signal'] == -1]
ax1.scatter(buys.index,  buys['price'],  marker='^', color='limegreen',
            s=40, zorder=5, label='Long entry')
ax1.scatter(sells.index, sells['price'], marker='v', color='red',
            s=40, zorder=5, label='Short entry')
ax1.set_title(f'Price + Bollinger Bands  (±{N_STD}σ, {WINDOW}-day window)',
              fontweight='bold')
ax1.set_ylabel('Price ($)'); ax1.legend(fontsize=9, loc='upper left')
ax1.grid(alpha=0.3)

# — Plot 2: Equity Curve ————————————————————————————————————————
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(df.index, df['equity'],    'b-',  lw=2, label=f'Strategy')
ax2.plot(df.index, df['equity_bh'], 'r--', lw=2, label='Buy & Hold')
ax2.fill_between(df.index,
                 df['equity'], df['equity_bh'],
                 where=df['equity'] >= df['equity_bh'],
                 alpha=0.12, color='blue')
ax2.fill_between(df.index,
                 df['equity'], df['equity_bh'],
                 where=df['equity'] <  df['equity_bh'],
                 alpha=0.12, color='red')
ax2.set_title('Equity Curve', fontweight='bold')
ax2.set_ylabel(f'Value ($, start ${INIT_CAP:,})')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x:,.0f}'))
ax2.legend(fontsize=10); ax2.grid(alpha=0.3)

# — Plot 3: Drawdown ————————————————————————————————————————————
ax3 = fig.add_subplot(gs[2, 0])
dd_pct = m_s['_dd'] * 100
ax3.fill_between(df.index, dd_pct, 0, color='red', alpha=0.4)
ax3.plot(df.index, dd_pct, 'r-', lw=0.8)
ax3.axhline(m_s['Max Drawdown']*100, color='darkred', linestyle='--',
            lw=1.5, label=f"Max DD: {m_s['Max Drawdown']:.2%}")
ax3.set_title('Strategy Drawdown', fontweight='bold')
ax3.set_ylabel('Drawdown (%)'); ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

# — Plot 4: Rolling Sharpe ——————————————————————————————————————
ax4 = fig.add_subplot(gs[2, 1])
ax4.plot(df.index, r_sharpe, color='purple', lw=1.5, label=f'{roll_win}d Rolling Sharpe')
ax4.axhline(0, color='k',  linestyle='--', alpha=0.5)
ax4.axhline(1, color='g',  linestyle='--', alpha=0.6, label='Sharpe = 1')
ax4.axhline(-1, color='r', linestyle='--', alpha=0.4, label='Sharpe = −1')
ax4.set_title(f'Rolling Sharpe ({roll_win}-day)', fontweight='bold')
ax4.set_ylabel('Sharpe Ratio'); ax4.legend(fontsize=9); ax4.grid(alpha=0.3)

plt.savefig('03_backtest_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*62)
print("  ✅ QUANTMIND Backtest Engine — Complete")
print(f"  Sharpe: {m_s['Sharpe']:.3f}  |  "
      f"Max DD: {m_s['Max Drawdown']:.2%}  |  "
      f"CAGR: {m_s['CAGR']:.2%}")
print("  Plot saved: 03_backtest_results.png")
print("="*62)
