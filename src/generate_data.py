"""
generate_data.py
-----------------
Generates a synthetic but realistic campus placement dataset.

Why synthetic? Public placement datasets on Kaggle are tiny (~200-600 rows),
inconsistent, and everyone's seen them. This script generates a larger,
noisier, more realistic dataset using domain logic (what actually correlates
with placement: CGPA, DSA/coding skill, internships, projects, communication,
backlogs) plus randomness, so the model has to genuinely learn signal from
noise instead of memorizing a tiny CSV.

You can swap this out for a real Kaggle dataset later if you want -- the
rest of the pipeline (preprocess.py, train.py, app.py) doesn't care where
the CSV came from as long as the column names match.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_STUDENTS = 2500

BRANCHES = ["CSE", "ECE", "Mechanical", "Civil", "Electrical", "Chemical"]
BRANCH_TECH_BIAS = {  # how much a branch nudges "tech skill" up/down
    "CSE": 0.35, "ECE": 0.15, "Electrical": 0.05,
    "Mechanical": -0.10, "Chemical": -0.15, "Civil": -0.20,
}


def generate(n=N_STUDENTS, seed=RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    branch = rng.choice(BRANCHES, size=n, p=[0.30, 0.20, 0.15, 0.13, 0.12, 0.10])

    cgpa = np.clip(rng.normal(7.4, 0.9, n), 5.0, 10.0).round(2)

    internships = rng.choice([0, 1, 2, 3], size=n, p=[0.35, 0.35, 0.20, 0.10])
    projects = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.10, 0.25, 0.30, 0.25, 0.10])
    backlogs = rng.choice([0, 1, 2, 3], size=n, p=[0.65, 0.20, 0.10, 0.05])

    tech_bias = np.array([BRANCH_TECH_BIAS[b] for b in branch])
    coding_score = np.clip(
        rng.normal(55 + tech_bias * 40, 18, n) + projects * 3 + internships * 2, 0, 100
    ).round(1)
    communication_score = np.clip(rng.normal(60, 15, n), 0, 100).round(1)

    certifications = rng.poisson(1.2, n)
    extracurricular = rng.choice([0, 1], size=n, p=[0.55, 0.45])

    # ---- Latent "placement propensity" (logistic combination of features) ----
    z = (
        -1.55
        + 0.85 * (cgpa - 7)
        + 0.020 * (coding_score - 55)
        + 0.012 * (communication_score - 60)
        + 0.55 * internships
        + 0.30 * projects
        - 0.9 * backlogs
        + 0.15 * certifications
        + 0.25 * extracurricular
        + tech_bias * 1.2
        + rng.normal(0, 0.9, n)  # irreducible noise -- real life is messy
    )
    prob = 1 / (1 + np.exp(-z))
    placed = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "branch": branch,
        "cgpa": cgpa,
        "internships": internships,
        "projects_completed": projects,
        "backlogs": backlogs,
        "coding_score": coding_score,
        "communication_score": communication_score,
        "certifications": certifications,
        "extracurricular": extracurricular,
        "placed": placed,
    })

    # sprinkle a few missing values -- real datasets always have some,
    # and handling them is part of what makes preprocess.py worth showing
    for col in ["coding_score", "communication_score"]:
        missing_idx = rng.choice(df.index, size=int(0.02 * n), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/placement_data.csv", index=False)
    print(f"Saved {len(df)} rows to data/placement_data.csv")
    print(df["placed"].value_counts(normalize=True).round(3))
