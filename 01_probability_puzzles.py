"""
=============================================================================
  QUANTMIND — Probability Puzzle Simulation Notebook
  20 Classic Quant Interview Problems: Analytical vs Simulation
=============================================================================
  Install : pip install numpy matplotlib scipy
  Run     : python 01_probability_puzzles.py
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import factorial
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
N = 100_000   # simulations per puzzle (reduce to 10_000 if slow)

results = {}   # {label: (simulated, analytical)}

def sep(): print("\n" + "─"*62)

print("="*62)
print("  QUANTMIND — Probability Puzzle Simulations")
print("  20 Classic Quant Interview Problems")
print("="*62)

# ── PUZZLE 1: Monty Hall ─────────────────────────────────────────────────────
sep()
print("PUZZLE 1 — Monty Hall Problem")
print("  Pick 1 of 3 doors. Host reveals a goat. Should you switch?")

stay_w = switch_w = 0
for _ in range(N):
    car  = np.random.randint(3)
    pick = np.random.randint(3)
    host = np.random.choice([d for d in range(3) if d != pick and d != car])
    stay_w   += (pick == car)
    new_pick  = [d for d in range(3) if d != pick and d != host][0]
    switch_w += (new_pick == car)

print(f"  Analytical  → Stay: {1/3:.4f}  |  Switch: {2/3:.4f}")
print(f"  Simulation  → Stay: {stay_w/N:.4f}  |  Switch: {switch_w/N:.4f}")
print(f"  ✅ ALWAYS SWITCH — {switch_w/stay_w:.1f}× more likely to win")
results['Monty Hall P(win|switch)'] = (switch_w/N, 2/3)

# ── PUZZLE 2: Birthday Paradox ───────────────────────────────────────────────
sep()
print("PUZZLE 2 — Birthday Paradox")
print("  How many people needed for P(shared birthday) > 50%?")

n23_analytical = 1 - np.prod([(365-i)/365 for i in range(23)])
hits = sum(1 for _ in range(10_000) if len(set(np.random.randint(1,366,23))) < 23)
n23_sim = hits / 10_000
threshold = next(n for n in range(2,60) if 1-np.prod([(365-i)/365 for i in range(n)]) >= 0.5)

print(f"  Analytical  → Need {threshold} people for P > 50%")
print(f"  P(match, n=23): Analytical={n23_analytical:.4f}  |  Simulation={n23_sim:.4f}")
print(f"  ✅ Just 23 people! Counterintuitive.")
results['Birthday P(match) n=23'] = (n23_sim, n23_analytical)

# ── PUZZLE 3: Gambler's Ruin ─────────────────────────────────────────────────
sep()
print("PUZZLE 3 — Gambler's Ruin")
print("  Start $50, target $100, fair coin. P(ruin before doubling)?")

start, target = 50, 100
p_ruin_ana = 1 - start/target

n_ruin = 0
for _ in range(N):
    w = start
    while 0 < w < target:
        w += 1 if np.random.random() < 0.5 else -1
    if w == 0:
        n_ruin += 1

print(f"  Analytical  → P(ruin) = 1 − {start}/{target} = {p_ruin_ana:.4f}")
print(f"  Simulation  → P(ruin) = {n_ruin/N:.4f}")
print(f"  ✅ Even a fair game: 50% chance of ruin before doubling.")
results["Gambler's Ruin P(ruin)"] = (n_ruin/N, p_ruin_ana)

# ── PUZZLE 4: Coupon Collector ───────────────────────────────────────────────
sep()
print("PUZZLE 4 — Coupon Collector Problem")
print("  How many cereal boxes to collect all 6 unique coupons?")

n_c = 6
E_ana = n_c * sum(1/k for k in range(1, n_c+1))

totals = []
for _ in range(N):
    seen, count = set(), 0
    while len(seen) < n_c:
        seen.add(np.random.randint(n_c))
        count += 1
    totals.append(count)

print(f"  Analytical  → E[boxes] = {E_ana:.4f}")
print(f"  Simulation  → E[boxes] = {np.mean(totals):.4f}")
print(f"  ✅ Same formula as rolling a die until all 6 faces appear.")
results['Coupon Collector E[T]'] = (np.mean(totals), E_ana)

# ── PUZZLE 5: St. Petersburg Paradox ────────────────────────────────────────
sep()
print("PUZZLE 5 — St. Petersburg Paradox")
print("  Flip until Heads; win 2^n. Infinite EV — how much to pay?")

payouts = [2**sum(1 for _ in iter(lambda: np.random.random()>0.5, True) if True)
           for _ in range(N)]
# simpler approach:
payouts2 = []
for _ in range(N):
    flips = 1
    while np.random.random() > 0.5:
        flips += 1
    payouts2.append(2**flips)

print(f"  Analytical  → E[X] = ∞  (mathematically divergent)")
print(f"  Simulation  → Mean ≈ ${np.mean(payouts2):.2f}  |  Median = ${np.median(payouts2):.0f}")
print(f"  ✅ Infinite EV, but rational agents pay < $25. That's the paradox.")

# ── PUZZLE 6: Secretary / Optimal Stopping ───────────────────────────────────
sep()
print("PUZZLE 6 — Secretary Problem (Optimal Stopping)")
print("  Interview 100 candidates. Reject first r, then pick next best. Optimal r?")

n_cand      = 100
r_opt_ana   = int(n_cand / np.e)
p_opt_ana   = 1 / np.e

wins_per_r  = []
for r in range(1, n_cand):
    wins = 0
    for _ in range(2_000):
        cands = np.random.permutation(n_cand)
        best_r = cands[:r].max()
        chosen = next((c for c in cands[r:] if c > best_r), None)
        if chosen == n_cand - 1:
            wins += 1
    wins_per_r.append(wins / 2_000)

best_r_sim = np.argmax(wins_per_r) + 1
print(f"  Analytical  → Optimal r = n/e ≈ {r_opt_ana}  |  P(best) ≈ {p_opt_ana:.4f}")
print(f"  Simulation  → Best r found = {best_r_sim}  |  P(best) ≈ {max(wins_per_r):.4f}")
print(f"  ✅ Reject first 37%, then pick first candidate better than all seen.")
results['Secretary P(best)'] = (max(wins_per_r), p_opt_ana)

# ── PUZZLE 7: HH vs HT Expected Flips ───────────────────────────────────────
sep()
print("PUZZLE 7 — Expected Flips: HH vs HT")
print("  Is E[flips until HH] equal to E[flips until HT]?")

def exp_flips(target, n_sims=30_000):
    counts = []
    tgt = list(target)
    for _ in range(n_sims):
        seq, k = [], 0
        while True:
            seq.append('H' if np.random.random() < 0.5 else 'T')
            k += 1
            if seq[-len(tgt):] == tgt:
                break
        counts.append(k)
    return np.mean(counts)

e_hh = exp_flips('HH')
e_ht = exp_flips('HT')
print(f"  Analytical  → E[HH] = 6  |  E[HT] = 4")
print(f"  Simulation  → E[HH] ≈ {e_hh:.2f}  |  E[HT] ≈ {e_ht:.2f}")
print(f"  ✅ NOT equal — HH takes longer due to overlapping dependency structure.")
results['E[flips to HH]'] = (e_hh, 6)
results['E[flips to HT]'] = (e_ht, 4)

# ── PUZZLE 8: Roll Die Until All Faces ──────────────────────────────────────
sep()
print("PUZZLE 8 — Roll a Die Until All 6 Faces Seen")

E_dice_ana = 6 * sum(1/k for k in range(1, 7))
dice_totals = []
for _ in range(N):
    seen, count = set(), 0
    while len(seen) < 6:
        seen.add(np.random.randint(1, 7))
        count += 1
    dice_totals.append(count)

print(f"  Analytical  → E[rolls] = 6·H₆ = {E_dice_ana:.4f}")
print(f"  Simulation  → E[rolls] = {np.mean(dice_totals):.4f}")
print(f"  ✅ Coupon Collector applied to dice — same harmonic sum formula.")
results['Dice All Faces E[rolls]'] = (np.mean(dice_totals), E_dice_ana)

# ── PUZZLE 9: Buffon's Needle — Estimate π ───────────────────────────────────
sep()
print("PUZZLE 9 — Buffon's Needle (Estimate π)")
print("  Needle length = line spacing = 1. P(cross a line) = 2/π.")

L = d = 1.0
crosses = sum(
    1 for _ in range(N)
    if np.random.uniform(0, d/2) <= (L/2) * np.sin(np.random.uniform(0, np.pi/2))
)
pi_est = (2 * L * N) / (d * crosses) if crosses > 0 else 0
print(f"  Analytical  → P(cross) = 2/π = {2/np.pi:.4f}")
print(f"  Simulation  → π estimate = {pi_est:.4f}  (true π = {np.pi:.4f})")
print(f"  ✅ Physical randomness recovers a pure mathematical constant.")

# ── PUZZLE 10: Hat Check / Derangements ─────────────────────────────────────
sep()
print("PUZZLE 10 — Hat Check Problem (Derangements)")
print("  n=10 people get hats back randomly. P(nobody gets their own)?")

n_hats = 10
p_derange_ana = float(sum((-1)**k / factorial(k) for k in range(n_hats+1)))
deranged = sum(
    1 for _ in range(N)
    if not any(np.random.permutation(n_hats)[i] == i for i in range(n_hats))
)
print(f"  Analytical  → P(derangement) = {p_derange_ana:.6f}")
print(f"  Simulation  → P(derangement) = {deranged/N:.6f}")
print(f"  ✅ Converges to 1/e ≈ {1/np.e:.6f} for large n.")
results['Derangement P (n=10)'] = (deranged/N, p_derange_ana)

# ── PUZZLE 11: Expected Max of n Uniforms ────────────────────────────────────
sep()
print("PUZZLE 11 — Expected Maximum of n Uniform[0,1] Variables")

print(f"  {'n':>6} | {'Analytical n/(n+1)':>20} | {'Simulation':>12}")
print(f"  {'-'*42}")
for n_v in [1, 2, 5, 10, 50]:
    ana = n_v / (n_v + 1)
    sim = np.mean(np.max(np.random.uniform(0,1,(20_000, n_v)), axis=1))
    print(f"  {n_v:>6} | {ana:>20.6f} | {sim:>12.6f}")
print(f"  ✅ E[max] = n/(n+1) — clean closed form from order statistics.")

# ── PUZZLE 12: Broken Stick Triangle ────────────────────────────────────────
sep()
print("PUZZLE 12 — Broken Stick Triangle Probability")
print("  Break a stick at 2 random points. P(3 pieces form a triangle)?")

p_tri_ana = 1/4
triangles = 0
for _ in range(N):
    cuts = sorted(np.random.uniform(0,1,2))
    a, b, c = cuts[0], cuts[1]-cuts[0], 1-cuts[1]
    if a+b > c and a+c > b and b+c > a:
        triangles += 1

print(f"  Analytical  → P(triangle) = 1/4 = {p_tri_ana:.4f}")
print(f"  Simulation  → P(triangle) = {triangles/N:.4f}")
print(f"  ✅ Each piece must be < 1/2; elegant geometric proof exists.")
results['Broken Stick P(triangle)'] = (triangles/N, p_tri_ana)

# ── PUZZLE 13: Sum > 1 (the e puzzle) ───────────────────────────────────────
sep()
print("PUZZLE 13 — Sum > 1 Problem (The e Puzzle)")
print("  Draw Uniform[0,1] one by one. How many until sum > 1?")

counts = []
for _ in range(N):
    total, k = 0.0, 0
    while total <= 1.0:
        total += np.random.uniform(0,1)
        k += 1
    counts.append(k)

print(f"  Analytical  → E[N] = e = {np.e:.6f}")
print(f"  Simulation  → E[N] = {np.mean(counts):.6f}")
print(f"  ✅ Euler's number emerges from pure randomness.")
results['E[N] sum > 1'] = (np.mean(counts), np.e)

# ── PUZZLE 14: Geometric Distribution ───────────────────────────────────────
sep()
print("PUZZLE 14 — Geometric Distribution: Expected First Success")
print("  Roll a fair die. E[rolls until first 6]?")

E_geo_ana = 6.0
rolls = []
for _ in range(N):
    k = 0
    while True:
        k += 1
        if np.random.randint(1,7) == 6:
            break
    rolls.append(k)

print(f"  Analytical  → E[rolls] = 1/p = {E_geo_ana:.4f}")
print(f"  Simulation  → E[rolls] = {np.mean(rolls):.4f}")
print(f"  ✅ Memoryless: past failures don't affect future probability.")
results['Geometric E[rolls to 6]'] = (np.mean(rolls), E_geo_ana)

# ── PUZZLE 15: Random Walk Hitting Time ─────────────────────────────────────
sep()
print("PUZZLE 15 — Random Walk Expected Hitting Time")
print("  ±1 walk from 0. E[time to reach +5 or −5]?")

tgt = 5
E_hit_ana = tgt**2
hit_times = []
for _ in range(N // 5):
    pos, t = 0, 0
    while abs(pos) < tgt:
        pos += 1 if np.random.random() < 0.5 else -1
        t   += 1
    hit_times.append(t)

print(f"  Analytical  → E[T] = n² = {E_hit_ana}")
print(f"  Simulation  → E[T] ≈ {np.mean(hit_times):.2f}")
print(f"  ✅ Fundamental: expected hitting time of ±n is n².")
results['Random Walk E[T] to ±5'] = (np.mean(hit_times), E_hit_ana)

# ── PUZZLE 16: Two Dice — P(sum = 7) ────────────────────────────────────────
sep()
print("PUZZLE 16 — Two Dice: P(sum = 7)")
print("  Most common sum on 2d6?")

rolls_2d6 = np.random.randint(1,7,(N,2)).sum(axis=1)
p7_sim = (rolls_2d6 == 7).mean()
print(f"  Analytical  → P(sum=7) = 6/36 = {6/36:.4f}")
print(f"  Simulation  → P(sum=7) = {p7_sim:.4f}")
print(f"  ✅ 7 is most likely: (1,6)(2,5)(3,4)(4,3)(5,2)(6,1) = 6 ways.")
results['P(2d6 sum = 7)'] = (p7_sim, 6/36)

# ── PUZZLE 17: Banach's Matchbox ────────────────────────────────────────────
sep()
print("PUZZLE 17 — Banach's Matchbox Problem")
print("  2 boxes, n=50 matches each. Pick randomly. E[remaining when one empties]?")

n_b = 50
E_banach_ana = np.sqrt(np.pi * n_b / 2)
lefts = []
for _ in range(N // 20):
    boxes = [n_b, n_b]
    while True:
        pick = np.random.randint(2)
        if boxes[pick] == 0:
            lefts.append(boxes[1-pick])
            break
        boxes[pick] -= 1

print(f"  Analytical  → E[remaining] ≈ √(πn/2) = {E_banach_ana:.2f}")
print(f"  Simulation  → E[remaining] ≈ {np.mean(lefts):.2f}")
print(f"  ✅ Classic Stefan Banach problem in combinatorial probability.")

# ── PUZZLE 18: Monte Carlo π (circle darts) ─────────────────────────────────
sep()
print("PUZZLE 18 — Monte Carlo π Estimation (Circle Method)")
print("  Throw darts at unit square. P(inside circle) = π/4.")

x = np.random.uniform(-1,1,N)
y = np.random.uniform(-1,1,N)
pi_mc = 4 * np.sum(x**2 + y**2 <= 1) / N
print(f"  True π      = {np.pi:.6f}")
print(f"  Simulation  → π ≈ {pi_mc:.6f}  (error = {abs(pi_mc-np.pi)/np.pi*100:.4f}%)")
print(f"  ✅ Monte Carlo integration in its purest form.")

# ── PUZZLE 19: Bayes' Theorem — Medical Test ────────────────────────────────
sep()
print("PUZZLE 19 — Bayes' Theorem: Medical Test Paradox")
print("  Disease: 1% prevalence. Test: 99% accurate. P(disease | positive)?")

prev, sens, spec = 0.01, 0.99, 0.99
p_pos = sens*prev + (1-spec)*(1-prev)
p_D_given_pos_ana = (sens * prev) / p_pos

has_D     = np.random.random(N) < prev
test_pos  = np.where(has_D, np.random.random(N)<sens, np.random.random(N)>(spec))
p_D_given_pos_sim = np.sum(has_D & test_pos) / np.sum(test_pos) if np.sum(test_pos)>0 else 0

print(f"  Analytical  → P(disease|positive) = {p_D_given_pos_ana:.4f} = {p_D_given_pos_ana*100:.1f}%")
print(f"  Simulation  → P(disease|positive) = {p_D_given_pos_sim:.4f} = {p_D_given_pos_sim*100:.1f}%")
print(f"  ✅ Shocking: even with 99% accuracy, only ~50% likely to have disease!")
results["Bayes P(disease|+)"] = (p_D_given_pos_sim, p_D_given_pos_ana)

# ── PUZZLE 20: Ballot Problem ────────────────────────────────────────────────
sep()
print("PUZZLE 20 — Ballot Problem")
print("  A gets 60 votes, B gets 40. P(A strictly ahead throughout count)?")

a_votes, b_votes = 60, 40
# Analytical: P = (a-b)/(a+b)
p_ballot_ana = (a_votes - b_votes) / (a_votes + b_votes)

ahead_all = 0
for _ in range(N):
    ballot = np.array([1]*a_votes + [-1]*b_votes)
    np.random.shuffle(ballot)
    cumsum = np.cumsum(ballot)
    if cumsum.min() > 0:
        ahead_all += 1

print(f"  Analytical  → P(always ahead) = (a−b)/(a+b) = {p_ballot_ana:.4f}")
print(f"  Simulation  → P(always ahead) = {ahead_all/N:.4f}")
print(f"  ✅ Famous ballot theorem — elegant combinatorial proof.")
results['Ballot P(always ahead)'] = (ahead_all/N, p_ballot_ana)

# ── SUMMARY TABLE ────────────────────────────────────────────────────────────
print("\n\n" + "="*62)
print("  SIMULATION vs ANALYTICAL — ACCURACY SUMMARY")
print("="*62)
print(f"  {'Puzzle':<38} {'Simulated':>9} {'Analytical':>11} {'Error%':>7}")
print(f"  {'-'*67}")
for name, (sim, ana) in results.items():
    err = abs(sim-ana)/abs(ana)*100 if ana != 0 else 0
    print(f"  {name:<38} {sim:>9.4f} {ana:>11.4f} {err:>6.2f}%")

# ── CONVERGENCE PLOTS ────────────────────────────────────────────────────────
print("\n  Generating convergence plots...")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('QUANTMIND — Simulation vs Analytical Convergence', fontsize=14, fontweight='bold')
axes = axes.flatten()

# Birthday paradox curve
n_range = range(2, 60)
analytical_b = [1-np.prod([(365-i)/365 for i in range(n)]) for n in n_range]
axes[0].plot(n_range, analytical_b, 'b-', linewidth=2, label='Analytical')
axes[0].axhline(0.5, color='r', linestyle='--', label='50%')
axes[0].axvline(23, color='g', linestyle=':', label='n=23')
axes[0].set_title('Birthday Paradox'); axes[0].set_xlabel('Group size n')
axes[0].set_ylabel('P(shared birthday)'); axes[0].legend(); axes[0].grid(alpha=0.3)

# Gambler's ruin: P(ruin) vs starting wealth
starts = range(10, 91, 10)
p_ruin_range = [1 - s/100 for s in starts]
axes[1].plot(starts, p_ruin_range, 'ro-', linewidth=2, markersize=6)
axes[1].set_title("Gambler's Ruin vs Starting Wealth")
axes[1].set_xlabel('Starting wealth ($, target=$100)')
axes[1].set_ylabel('P(ruin)'); axes[1].grid(alpha=0.3)

# Coupon collector: E[draws] vs n
n_coup = range(1, 21)
e_coup = [n * sum(1/k for k in range(1,n+1)) for n in n_coup]
axes[2].plot(n_coup, e_coup, 'g^-', linewidth=2, markersize=6)
axes[2].set_title('Coupon Collector E[T] vs n')
axes[2].set_xlabel('Number of coupon types')
axes[2].set_ylabel('Expected draws'); axes[2].grid(alpha=0.3)

# Secretary: P(best) vs r/n ratio
r_range = np.linspace(0.01, 0.99, 99)
axes[3].plot(r_range, wins_per_r[:99] if len(wins_per_r)>=99 else wins_per_r,
             'b-', label='Simulation', alpha=0.7)
axes[3].axvline(1/np.e, color='r', linestyle='--', label=f'1/e ≈ {1/np.e:.2f}')
axes[3].axhline(1/np.e, color='g', linestyle=':', label=f'P* ≈ {1/np.e:.2f}')
axes[3].set_title('Secretary: P(best) vs Reject Fraction')
axes[3].set_xlabel('r / n (fraction rejected)'); axes[3].set_ylabel('P(best)')
axes[3].legend(); axes[3].grid(alpha=0.3)

# Option price max of n uniforms
n_unif = range(1, 51)
e_max = [n/(n+1) for n in n_unif]
axes[4].plot(n_unif, e_max, 'm-', linewidth=2)
axes[4].set_title('E[max] of n Uniforms = n/(n+1)')
axes[4].set_xlabel('n'); axes[4].set_ylabel('E[max]'); axes[4].grid(alpha=0.3)

# Monte Carlo π convergence
ns = [100, 500, 1000, 5000, 10000, 50000, N]
pi_ests = []
X = np.random.uniform(-1,1,(N,2))
for n_pi in ns:
    inside = np.sum(X[:n_pi,0]**2 + X[:n_pi,1]**2 <= 1)
    pi_ests.append(4*inside/n_pi)
axes[5].semilogx(ns, pi_ests, 'co-', linewidth=2, markersize=6, label='MC π estimate')
axes[5].axhline(np.pi, color='r', linestyle='--', label=f'True π')
axes[5].set_title('MC π Convergence'); axes[5].set_xlabel('Simulations (log)')
axes[5].set_ylabel('π estimate'); axes[5].legend(); axes[5].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('01_probability_puzzles.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*62)
print("  ✅ QUANTMIND Probability Puzzles — Complete")
print("  Plot saved: 01_probability_puzzles.png")
print("="*62)
