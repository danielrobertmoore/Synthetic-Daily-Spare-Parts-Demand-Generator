import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import nbinom, poisson


# ─────────────────────────  CONFIG VIA CLI  ──────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sku", type=int, default=400,
                   help="number of synthetic SKUs")
    p.add_argument("--start", type=str, default="2023-01-01",
                   help="window start date YYYY‑MM‑DD")
    p.add_argument("--end", type=str, default="2024-12-31",
                   help="window end date YYYY‑MM‑DD")
    p.add_argument("--seed", type=int, default=42,
                   help="random seed")
    p.add_argument("--outfile", type=str,
                   default="synthetic_spare_parts_daily.csv",
                   help="CSV output path")
    return p.parse_args()


# ─────────────────────────  NB PARAM SOLVER  ─────────────────────────
def nb_params(mean: float, cv: float):
    """
    Return (r, p) for numpy's Negative‑Binomial given mean & CV.
    If variance ≤ mean, returns (None, None) signalling 'use Poisson'.
    """
    var = (cv * mean) ** 2
    if var <= mean:          # NB undefined; revert to Poisson
        return None, None
    r = mean ** 2 / (var - mean)
    p = r / (r + mean)
    # numerical safety clip
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return r, p


# ─────────────────────────  MAIN GENERATOR  ─────────────────────────
def generate_data(n_sku: int,
                  start_date: str,
                  end_date: str,
                  seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    categories = np.random.choice(
        ["smooth", "erratic", "slow", "lumpy"],
        size=n_sku,
        p=[0.20, 0.20, 0.30, 0.30]
    )

    date_idx = pd.date_range(start_date, end_date, freq="D")
    t0 = pd.to_datetime(start_date)

    recs = []

    for idx in tqdm(range(n_sku), desc="Generating SKUs", unit="sku"):
        cat = categories[idx]
        sku = f"SKU-{idx+1:04d}"

        # daily Poisson rate λ_d
        if cat in ("smooth", "erratic"):
            lam_daily = np.random.uniform(0.7, 1.0)    # ~daily
        else:
            lam_daily = 10 ** np.random.uniform(-3.5, -1.0)  # sparse

        # seasonality, ageing parameters
        peak_year = np.random.randint(1, 4)
        decay = np.random.uniform(0.05, 0.15)

        # demand size stats
        mean_size = np.random.uniform(1, 10)
        cv_size = (np.random.uniform(0.3, 0.6) if cat in ("smooth", "slow")
                   else np.random.uniform(0.8, 2.5))

        r_nb, p_nb = nb_params(mean_size, cv_size)
        uses_poisson_size = r_nb is None

        for cal_date in date_idx:
            # ageing decay
            yrs = cal_date.year - t0.year
            lam_adj = lam_daily * max(1 - decay * max(yrs - peak_year, 0), 0.1)

            # seasonality factors
            dow_factor = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0,
                          5: 0.6, 6: 0.4}[cal_date.weekday()]
            q_factor = 1.25 if cal_date.month in (7, 12) else 1.0
            lam_today = lam_adj * dow_factor * q_factor

            # Bernoulli draw (Poisson λ≪1)
            if np.random.rand() < lam_today:
                if uses_poisson_size:
                    qty = max(1, poisson.rvs(mean_size))
                else:
                    qty = max(1, nbinom.rvs(r_nb, p_nb))
            else:
                qty = 0

            recs.append((cal_date, sku, qty))

    df = pd.DataFrame(recs, columns=["Demand Date", "Material", "Demand Size"])
    df.sort_values(["Material", "Demand Date"], inplace=True)
    return df


# ─────────────────────────────  ENTRY  ───────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    df_syn = generate_data(args.sku, args.start, args.end, args.seed)
    df_syn.to_csv(args.outfile, index=False)
    print(f"\nSynthetic data written → {args.outfile}  "
          f"({len(df_syn):,} rows, {args.sku} SKUs)")