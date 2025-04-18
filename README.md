# Synthetic Daily Spare‑Parts Demand Generator (Poisson–Negative‑Binomial Framework)

A Python tool for generating synthetic demand data for spare parts inventory modeling. This simulator creates realistic daily demand patterns following different demand categories (smooth, erratic, slow, and lumpy) with seasonality and aging effects.

## Purpose

This generator creates a **statistically credible, fully padded day‑by‑day demand record** for any number of service‑parts SKUs.  
It is designed to:

* Benchmark Croston/SBA, bootstrap or causal forecasting engines
* Stress‑test inventory and network optimisers
* Provide shareable demo data that reveal no proprietary volumes

## Stochastic foundations

| Component | Chosen model | Rationale |
|-----------|--------------|-----------|
|Demand incidence (hits)|**Poisson arrival process**, discretised to daily Bernoulli draws (`p ≈ λ_d`).|Matches the exponential inter‑arrival assumption used in Croston‑family research and observed in field data.|
|Demand size when a hit occurs|**Negative‑Binomial (r,p)**; if variance ≤ mean the model falls back to **Poisson(μ)**.|NB can reproduce any mean/CV pair and the heavy tails typical of pick tickets.|
|Intermittency categories|SKUs randomly assigned to *smooth, erratic, slow,* or *lumpy* (20 ⁄ 20 ⁄ 30 ⁄ 30 %).|Mirrors textbook segmentation (Bartezzaghi & Kalchschmidt) for realistic mix.|
|Seasonality|Weekly factor *(Mon–Fri = 1.0, Sat = 0.6, Sun = 0.4)* plus optional quarterly uplift *(July, Dec × 1.25).*|Captures maintenance scheduling and year‑end shutdown spikes.|
|Ageing (obsolescence)|Daily hit‑rate λ decays linearly after a random peak year (1‑4 yr) at 5‑15 % per year.|Models declining installed base or product substitution lifecycle.|

## Parameter synthesis

1. **Choose λ<sub>d</sub>**  
   *Smooth/Erratic*: 0.7 – 1.0 hits/day  
   *Slow/Lumpy*: 10^{U(‑3.5,‑1.0)} ≈ 1 hit / 30–3000 days  
2. **Set size mean μ ∈ U(1, 10)**; draw **CV** based on category.  
3. **Convert (μ, CV) → (r,p)** via  
   `var = (CV·μ)^2 ; r = μ² / (var–μ) ; p = r/(r+μ)`  
   Clip `p` to (10⁻⁶, 1‑10⁻⁶); if `var ≤ μ` default to Poisson.  
4. **Apply seasonality & ageing multipliers** to λ each day.  
5. **Draw hit** (`Uniform(0,1) < λ_today`). If hit ⇒ sample size from NB/Poisson; else size = 0.

## Algorithmic complexity

*Time*: **O(n_sku × n_days)** – 400 SKUs × 730 days ≈ 0.02 s on a 2020 MacBook (NumPy vectorised except inner Bernoulli loop).  
*Memory*: single pass; holds one day's data per SKU, so negligible RAM (< 50 MB for 10 k SKUs × 5 yr).

## Outputs

CSV columns:  

|Field|Type|Description|
|-----|----|-----------|
|`Demand Date`|ISO‑8601 date|Daily bucket – dense grid (no missing days).|
|`Material`|string|SKU label `SKU‑0001` …|
|`Demand Size`|int (≥ 0)|0 ⇒ no demand; > 0 ⇒ units requested.|

## CLI interface

```
python main.py --sku 1000 \
               --start 2021-01-01 --end 2023-12-31 \
               --seed 123 \
               --outfile syn_1k.csv
```

* `--sku`      Number of SKUs.  
* `--start`/`--end` Date window (daily resolution).  
* `--seed`     Deterministic reproducibility.  
* `--outfile`  CSV path.

Progress is visualised via **tqdm**.

## Dependencies

|Library|Role|Version tested|
|-------|----|-------------|
|NumPy|random draws & math|≥ 1.23|
|Pandas|dataframe assembly & CSV|≥ 1.5|
|SciPy|negative‑binomial & χ² utilities|≥ 1.11|
|tqdm|CLI progress bar|≥ 4.65|

*(all listed in `requirements.txt`)*

## License

MIT License