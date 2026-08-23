"""
eda.py
Exploratory data analysis for the multi-planet system dataset -- run this
BEFORE any modeling, right after pulling the real archive data. This should
have been Phase 2's first step; several of the bugs we already hit
(missing stellar data, extreme outliers, Kepler dominance) would have been
visible here immediately instead of surfacing as training crashes later.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_pipeline import FEATURE_COLS, STAR_COLS, ALL_COLS


def run_eda(df: pd.DataFrame, save_dir: str = "eda_plots"):
    import os
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("1. BASIC SHAPE")
    print("=" * 60)
    print(f"Total planet rows: {len(df)}")
    print(f"Total unique systems: {df['hostname'].nunique()}")

    # --- Multiplicity distribution ---
    systems = df.drop_duplicates(subset="hostname")
    print("\nMultiplicity distribution:")
    print(systems["sy_pnum"].value_counts().sort_index())

    fig, ax = plt.subplots(figsize=(7, 4))
    systems["sy_pnum"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("System multiplicity distribution")
    ax.set_xlabel("Number of planets in system")
    ax.set_ylabel("Number of systems")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/01_multiplicity_distribution.png", dpi=120)
    plt.close()

    print("\n" + "=" * 60)
    print("2. MISSING DATA")
    print("=" * 60)
    missing_pct = df[ALL_COLS].isna().mean() * 100
    print(missing_pct.round(2))

    fig, ax = plt.subplots(figsize=(8, 4))
    missing_pct.plot(kind="bar", ax=ax, color="firebrick")
    ax.set_title("Missing data by column (%)")
    ax.set_ylabel("% missing")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/02_missing_data.png", dpi=120)
    plt.close()

    print("\n" + "=" * 60)
    print("3. DISCOVERY METHOD / FACILITY BREAKDOWN")
    print("=" * 60)
    # This directly addresses the "why is the data Kepler-dominated" question --
    # quantify it explicitly rather than eyeballing it from row samples.
    if "discoverymethod" in df.columns:
        print(systems["discoverymethod"].value_counts())
    kepler_frac = systems["hostname"].str.contains("Kepler|KOI", case=False, na=False).mean()
    print(f"\nFraction of systems with Kepler/KOI in hostname: {kepler_frac:.1%}")
    print("(This is expected, not a bug -- Kepler's long continuous baseline")
    print(" uniquely suited it to finding many-planet systems with full orbital")
    print(" solutions; TESS's short sector length rarely does. Report this as")
    print(" a limitation on generalization to non-Kepler architectures.)")

    print("\n" + "=" * 60)
    print("4. DISTRIBUTIONS OF KEY FEATURES (log scale)")
    print("=" * 60)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, FEATURE_COLS):
        vals = df[col].dropna()
        vals = vals[vals > 0]  # log requires positive
        ax.hist(np.log10(vals), bins=40, color="steelblue", edgecolor="white")
        ax.set_title(f"log10({col})")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/03_feature_distributions.png", dpi=120)
    plt.close()

    print("Printed summary stats (raw units):")
    print(df[FEATURE_COLS].describe())

    print("\n" + "=" * 60)
    print("5. OUTLIER CHECK -- exactly what caused earlier NaN training crashes")
    print("=" * 60)
    for col in ["pl_orbper", "pl_bmasse", "pl_rade"]:
        vals = df[col].dropna()
        p1, p50, p99 = vals.quantile([0.01, 0.5, 0.99])
        print(f"{col}: p1={p1:.3g}  median={p50:.3g}  p99={p99:.3g}  "
              f"max={vals.max():.3g}  (ratio max/median = {vals.max()/p50:.1f}x)")

    print("\n" + "=" * 60)
    print("6. PEAS-IN-A-POD CHECK -- adjacent period ratio distribution")
    print("=" * 60)
    # This is the core physical signal your whole approach depends on --
    # visualize it directly rather than assuming it's there.
    ratios = []
    for hostname, sys_df in df.groupby("hostname"):
        periods = np.sort(sys_df["pl_orbper"].dropna().values)
        if len(periods) >= 2:
            ratios.extend(periods[1:] / periods[:-1])
    ratios = np.array(ratios)
    print(f"Adjacent period ratio: median={np.median(ratios):.2f}, "
          f"mean={np.mean(ratios):.2f}, std={np.std(ratios):.2f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.log10(ratios[ratios > 0]), bins=50, color="seagreen", edgecolor="white")
    ax.set_title("log10(adjacent period ratio) -- peas-in-a-pod check")
    ax.set_xlabel("log10(P_outer / P_inner)")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/04_period_ratio_distribution.png", dpi=120)
    plt.close()

    print("\n" + "=" * 60)
    print("7. ECCENTRICITY DISTRIBUTION (should skew low if peas-in-a-pod holds)")
    print("=" * 60)
    fig, ax = plt.subplots(figsize=(7, 4))
    df["pl_orbeccen"].dropna().hist(bins=40, ax=ax, color="darkorange", edgecolor="white")
    ax.set_title("Eccentricity distribution")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/05_eccentricity_distribution.png", dpi=120)
    plt.close()

    print("\n" + "=" * 60)
    print("8. CORRELATION MATRIX (numeric features)")
    print("=" * 60)
    corr = df[ALL_COLS].corr()
    print(corr.round(2))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(ALL_COLS)))
    ax.set_yticks(range(len(ALL_COLS)))
    ax.set_xticklabels(ALL_COLS, rotation=45, ha="right")
    ax.set_yticklabels(ALL_COLS)
    plt.colorbar(im)
    ax.set_title("Feature correlation matrix")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/06_correlation_matrix.png", dpi=120)
    plt.close()

    print(f"\nAll plots saved to ./{save_dir}/")
    return {"systems": systems, "period_ratios": ratios}


if __name__ == "__main__":
    print("Import and call run_eda(df) with your loaded, filtered dataframe.")
