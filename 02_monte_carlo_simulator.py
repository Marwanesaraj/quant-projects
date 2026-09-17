"""
=============================================================================
  QUANTMIND — Monte Carlo Market Simulator
  Geometric Brownian Motion · Black-Scholes · Options Pricing · Greeks
=============================================================================
  Install : pip install numpy matplotlib scipy
  Run     : python 02_monte_carlo_simulator.py
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm, lognorm
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*62)
print("  QUANTMIND — Monte Carlo Market Simulator")
print("  GBM Simulation + Black-Scholes Options Pricing")
print("="*62)

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS — Edit these freely
# ─────────────────────────────────────────────────────────────────────────────
S0    = 100.0    # Current stock price ($)
K     = 100.0    # Strike price ($)
T     = 1.0      # Time to expiry (years)
r     = 0.05     # Risk-free rate (annualized)
sigma = 0.20     # Volatility (annualized)
N     = 252      # Time steps per path (trading days)
M     = 10_000   # Monte Carlo paths

dt = T / N

print(f"\n  Parameters: S₀={S0} | K={K} | T={T}yr | r={r:.0%} | σ={sigma:.0%}")
print(f"  {M:,} paths × {N} steps\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — GBM Simulation
#   Exact solution: S(t) = S₀ · exp((r - σ²/2)t + σ·W(t))
# ─────────────────────────────────────────────────────────────────────────────
print("─"*62)
print("  SECTION 1: Geometric Brownian Motion Simulation")

Z            = np.random.standard_normal((M, N))
increments   = (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
S            = S0 * np.exp(np.cumsum(increments, axis=1))   # shape: (M, N)
S_T          = S[:, -1]                                      # terminal prices

E_ST_ana  = S0 * np.exp(r*T)
Std_ST_ana = S0 * np.exp(r*T) * np.sqrt(np.exp(sigma**2*T) - 1)

print(f"  E[S(T)] : analytical = {E_ST_ana:.4f}  |  simulated = {np.mean(S_T):.4f}")
print(f"  Std[S(T)]: analytical = {Std_ST_ana:.4f}  |  simulated = {np.std(S_T):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Black-Scholes Analytical Solution
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*62)
print("  SECTION 2: Black-Scholes Analytical Pricing")

def black_scholes(S0, K, T, r, sigma, opt='call'):
    """Return price and full Greeks for a European option."""
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    pv_K = K * np.exp(-r*T)

    if opt == 'call':
        price = S0*norm.cdf(d1) - pv_K*norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-(S0*norm.pdf(d1)*sigma)/(2*np.sqrt(T))
                 - r*pv_K*norm.cdf(d2)) / 365
        rho   = pv_K * T * norm.cdf(d2) / 100
    else:
        price = pv_K*norm.cdf(-d2) - S0*norm.cdf(-d1)
        delta = -norm.cdf(-d1)
        theta = (-(S0*norm.pdf(d1)*sigma)/(2*np.sqrt(T))
                 + r*pv_K*norm.cdf(-d2)) / 365
        rho   = -pv_K * T * norm.cdf(-d2) / 100

    gamma = norm.pdf(d1) / (S0*sigma*np.sqrt(T))
    vega  = S0 * norm.pdf(d1) * np.sqrt(T) / 100   # per 1% vol change

    return dict(price=price, delta=delta, gamma=gamma,
                vega=vega, theta=theta, rho=rho, d1=d1, d2=d2)

call_bs = black_scholes(S0, K, T, r, sigma, 'call')
put_bs  = black_scholes(S0, K, T, r, sigma, 'put')

print(f"\n  BLACK-SCHOLES PRICES")
print(f"  Call = ${call_bs['price']:.4f}   Put = ${put_bs['price']:.4f}")

pcp_lhs = call_bs['price'] - put_bs['price']
pcp_rhs = S0 - K*np.exp(-r*T)
print(f"\n  PUT-CALL PARITY CHECK: C − P = {pcp_lhs:.4f}  |  S₀ − Ke^(−rT) = {pcp_rhs:.4f}")
print(f"  {'✅ HOLDS' if abs(pcp_lhs - pcp_rhs) < 0.001 else '❌ FAILS'}")

print(f"\n  GREEKS (Call option, ATM)")
for g, v in [('Δ Delta ', call_bs['delta']),
             ('Γ Gamma ', call_bs['gamma']),
             ('ν Vega  ', call_bs['vega']),
             ('Θ Theta ', call_bs['theta']),
             ('ρ Rho   ', call_bs['rho'])]:
    desc = {'Δ Delta ': '[ΔPrice / ΔS]',
            'Γ Gamma ': '[ΔDelta / ΔS]',
            'ν Vega  ': '[per 1% vol move]',
            'Θ Theta ': '[daily decay $]',
            'ρ Rho   ': '[per 1% rate move]'}[g]
    print(f"    {g} = {v:>8.4f}   {desc}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Monte Carlo Option Pricing
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*62)
print("  SECTION 3: Monte Carlo Option Pricing")

disc    = np.exp(-r*T)
call_mc = disc * np.mean(np.maximum(S_T - K, 0))
put_mc  = disc * np.mean(np.maximum(K - S_T, 0))
call_se = disc * np.std(np.maximum(S_T-K,0)) / np.sqrt(M)
put_se  = disc * np.std(np.maximum(K-S_T,0)) / np.sqrt(M)

print(f"\n  {'':6} {'MC Price':>10} {'95% CI':>24} {'B-S':>8} {'Error':>8}")
print(f"  {'-'*58}")
for name, mc, se, bs in [('Call', call_mc, call_se, call_bs['price']),
                          ('Put ', put_mc, put_se,  put_bs['price'])]:
    err = (mc - bs)/bs * 100
    ci  = f"[{mc-1.96*se:.4f}, {mc+1.96*se:.4f}]"
    print(f"  {name}   ${mc:>9.4f}  {ci:>24}  ${bs:>7.4f} {err:>+7.2f}%")

print(f"\n  ✅ Monte Carlo converges to Black-Scholes as M → ∞")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Sensitivity Across Strikes
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*62)
print("  SECTION 4: Option Sensitivity — Strike vs Price / Greeks")

strikes      = np.linspace(60, 140, 81)
call_prices  = [black_scholes(S0, k, T, r, sigma, 'call')['price'] for k in strikes]
put_prices   = [black_scholes(S0, k, T, r, sigma, 'put')['price']  for k in strikes]
call_deltas  = [black_scholes(S0, k, T, r, sigma, 'call')['delta'] for k in strikes]
call_gammas  = [black_scholes(S0, k, T, r, sigma, 'call')['gamma'] for k in strikes]
call_vegas   = [black_scholes(S0, k, T, r, sigma, 'call')['vega']  for k in strikes]

# Volatility smile hint (skew): vary sigma across strikes
sigma_skew = sigma + 0.002 * ((strikes - S0)/S0)**2 * 30
call_smile = [black_scholes(S0, k, T, r, s, 'call')['price']
              for k, s in zip(strikes, sigma_skew)]

# MC convergence by path count
path_counts = [100, 500, 1000, 2000, 5000, 10000]
mc_conv     = [np.exp(-r*T)*np.mean(np.maximum(S[:n,-1]-K,0)) for n in path_counts]

print("  Sensitivity table computed. Generating plots...")

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 11))
fig.suptitle('QUANTMIND — Monte Carlo Market Simulator', fontsize=15, fontweight='bold', y=0.99)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

# — Plot 1: GBM Price Paths ——————————————————————————————————————
ax1 = fig.add_subplot(gs[0, :2])
t_axis = np.linspace(0, T, N)
for i in range(min(300, M)):
    ax1.plot(t_axis, S[i], alpha=0.03, color='steelblue', linewidth=0.6)
ax1.plot(t_axis, np.percentile(S,  5, axis=0), 'r--', lw=2, label='5th pct')
ax1.plot(t_axis, np.percentile(S, 50, axis=0), 'k-',  lw=2, label='Median')
ax1.plot(t_axis, np.percentile(S, 95, axis=0), 'g--', lw=2, label='95th pct')
ax1.axhline(K, color='orange', linestyle=':', lw=2, label=f'Strike K={K}')
ax1.set_title(f'GBM Price Paths  (M={M:,} paths, σ={sigma:.0%})', fontweight='bold')
ax1.set_xlabel('Time (years)'); ax1.set_ylabel('Stock Price ($)')
ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

# — Plot 2: Terminal Price Distribution ——————————————————————————
ax2 = fig.add_subplot(gs[0, 2])
ax2.hist(S_T, bins=80, density=True, color='steelblue', alpha=0.65, label='MC')
xs = np.linspace(S_T.min(), S_T.max(), 300)
mu_ln  = np.log(S0) + (r - 0.5*sigma**2)*T
std_ln = sigma*np.sqrt(T)
ax2.plot(xs, lognorm.pdf(xs, s=std_ln, scale=np.exp(mu_ln)), 'r-', lw=2, label='Lognormal')
ax2.axvline(K, color='orange', linestyle='--', lw=2, label=f'K={K}')
ax2.set_title('Terminal Price Distribution', fontweight='bold')
ax2.set_xlabel('S(T)'); ax2.set_ylabel('Density')
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

# — Plot 3: MC Convergence to B-S ————————————————————————————————
ax3 = fig.add_subplot(gs[1, 0])
ax3.semilogx(path_counts, mc_conv, 'bo-', lw=2, markersize=6, label='MC Call')
ax3.axhline(call_bs['price'], color='r', linestyle='--', lw=2,
            label=f"B-S = ${call_bs['price']:.2f}")
ax3.set_title('MC Convergence to B-S', fontweight='bold')
ax3.set_xlabel('Simulations'); ax3.set_ylabel('Call Price ($)')
ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

# — Plot 4: Option Price vs Strike ———————————————————————————————
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(strikes, call_prices,  'b-', lw=2, label='Call (B-S)')
ax4.plot(strikes, put_prices,   'r-', lw=2, label='Put (B-S)')
ax4.plot(strikes, call_smile,   'b--', lw=1.5, alpha=0.6, label='Call (vol skew)')
ax4.axvline(S0, color='k', linestyle=':', alpha=0.5, label=f'S₀={S0}')
ax4.set_title('Option Price vs Strike', fontweight='bold')
ax4.set_xlabel('Strike K'); ax4.set_ylabel('Option Price ($)')
ax4.legend(fontsize=9); ax4.grid(alpha=0.3)

# — Plot 5: Greeks vs Strike —————————————————————————————————————
ax5  = fig.add_subplot(gs[1, 2])
ax5r = ax5.twinx()
ax5.plot(strikes, call_deltas, 'b-',  lw=2, label='Delta')
ax5.plot(strikes, call_vegas,  'g-',  lw=2, label='Vega (×100)')
ax5r.plot(strikes, call_gammas, 'r-', lw=2, label='Gamma')
ax5.set_title('Call Greeks vs Strike', fontweight='bold')
ax5.set_xlabel('Strike K')
ax5.set_ylabel('Delta / Vega', color='k')
ax5r.set_ylabel('Gamma', color='r')
ax5.axvline(S0, color='k', linestyle=':', alpha=0.4)
lines = ax5.get_legend_handles_labels()[0] + ax5r.get_legend_handles_labels()[0]
labs  = ax5.get_legend_handles_labels()[1] + ax5r.get_legend_handles_labels()[1]
ax5.legend(lines, labs, fontsize=9); ax5.grid(alpha=0.3)

plt.savefig('02_monte_carlo_simulator.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*62)
print("  ✅ QUANTMIND Monte Carlo Simulator — Complete")
print(f"  MC Call = ${call_mc:.4f}  |  B-S Call = ${call_bs['price']:.4f}")
print(f"  MC Put  = ${put_mc:.4f}  |  B-S Put  = ${put_bs['price']:.4f}")
print(f"  Put-Call Parity: ✅  |  Greeks computed for all strikes")
print("  Plot saved: 02_monte_carlo_simulator.png")
print("="*62)
