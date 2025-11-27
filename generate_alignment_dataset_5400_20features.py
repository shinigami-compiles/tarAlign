"""
Generate synthetic TarAlign dataset with ONLY the 20 core features
+ aligned_label, using 60 users × 90 days = 5400 rows.

- Columns match your web-app 20-core schema:
    user_id, date,
    minutes_goal,
    consistency_index,
    sleep_hours,
    avg_minutes_last_week,
    avg_consistency_last_week,
    avg_sleep_last_week,
    avg_minutes_last_month,
    momentum_last_month,
    friction_last_month,
    avg_sleep_last_month,
    baseline_goal_minutes,
    baseline_consistency,
    baseline_sleep_hours,
    task_switch_avg,
    exercise_avg,
    day_of_week,
    is_weekend,
    goal_weight_career,
    goal_weight_fitness,
    goal_weight_learning,
    aligned_label

- aligned_label is derived from an internal "alignment_score"
  and thresholded at the median → ~50/50 class balance.

"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
RANDOM_SEED = 42

NUM_USERS = 60
DAYS_PER_USER = 90
TOTAL_ROWS = NUM_USERS * DAYS_PER_USER

START_DATE = datetime(2024, 1, 1)

OUT_CSV = "alignment_dataset_5400_20features.csv"


rng = np.random.default_rng(RANDOM_SEED)


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def clip(x, lo, hi):
    return float(np.clip(x, lo, hi))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# -------------------------------------------------------------------
# USER-LEVEL BASELINE GENERATION
# -------------------------------------------------------------------
def gen_user_baseline(user_idx: int) -> dict:
    """
    Generate a stable baseline profile per user:
    - baseline goal minutes, consistency, sleep
    - goal weights (career/fitness/learning)
    """
    user_id = f"user_{user_idx:03d}"

    # Baseline goal minutes: between 30 and 120
    baseline_goal_minutes = rng.uniform(30, 120)

    # Baseline consistency: between 0.35 and 0.9
    baseline_consistency = rng.uniform(0.35, 0.90)

    # Baseline sleep hours: around 7 ± 0.7
    baseline_sleep_hours = clip(rng.normal(7.0, 0.7), 5.5, 9.0)

    # Goal weights using a simple Dirichlet-like sampling
    raw_weights = rng.uniform(0.2, 1.0, size=3)
    raw_weights = raw_weights / raw_weights.sum()
    goal_weight_career, goal_weight_fitness, goal_weight_learning = raw_weights

    return {
        "user_id": user_id,
        "baseline_goal_minutes": baseline_goal_minutes,
        "baseline_consistency": baseline_consistency,
        "baseline_sleep_hours": baseline_sleep_hours,
        "goal_weight_career": goal_weight_career,
        "goal_weight_fitness": goal_weight_fitness,
        "goal_weight_learning": goal_weight_learning,
    }


# -------------------------------------------------------------------
# DAY-LEVEL ROW GENERATION
# -------------------------------------------------------------------
def gen_day_row(user_baseline: dict, day_offset: int) -> dict:
    """
    Generate a single day's behaviour for a given user:
    - minutes_goal, consistency_index, sleep_hours
    - momentum, friction_index, task_switch_rate, exercise_minutes
    - day_of_week, is_weekend
    - alignment_score (continuous 0-1)
    """
    date = START_DATE + timedelta(days=day_offset)
    day_of_week = date.weekday()  # 0=Mon, 6=Sun
    is_weekend = 1 if day_of_week >= 5 else 0

    b_goal = user_baseline["baseline_goal_minutes"]
    b_cons = user_baseline["baseline_consistency"]
    b_sleep = user_baseline["baseline_sleep_hours"]

    gw_career = user_baseline["goal_weight_career"]
    gw_fitness = user_baseline["goal_weight_fitness"]
    gw_learning = user_baseline["goal_weight_learning"]

    # -------------------------------------------------------------
    # minutes_goal: baseline ± noise, with mild weekend effects
    # -------------------------------------------------------------
    # Slight trend over time to allow variety
    trend_factor = 1.0 + rng.normal(0, 0.05) + (day_offset / (DAYS_PER_USER * 20.0))
    base_minutes = b_goal * trend_factor

    # Weekends: some users work less, some more
    weekend_boost = rng.normal(-0.1, 0.15) if is_weekend else rng.normal(0.0, 0.1)

    minutes_goal = base_minutes * (1.0 + weekend_boost)
    minutes_goal = clip(minutes_goal, 0.0, 240.0)

    # -------------------------------------------------------------
    # consistency_index: baseline ± noise
    # -------------------------------------------------------------
    consistency_index = b_cons + rng.normal(0.0, 0.10)
    # slight relation to how much they worked vs baseline
    consistency_index += 0.05 * ((minutes_goal / (b_goal + 1e-3)) - 1.0)
    consistency_index = float(np.clip(consistency_index, 0.0, 1.0))

    # -------------------------------------------------------------
    # sleep_hours: baseline ± noise, with some weekend drift
    # -------------------------------------------------------------
    sleep_hours = b_sleep + rng.normal(0.0, 0.6)
    if is_weekend:
        # On weekends people may sleep a bit more or less
        sleep_hours += rng.normal(0.3, 0.5)
    sleep_hours = clip(sleep_hours, 4.0, 10.0)

    # -------------------------------------------------------------
    # task_switch_rate: inversely related to consistency
    # -------------------------------------------------------------
    base_switch = 0.8 - consistency_index + rng.normal(0.0, 0.08)
    task_switch_rate = float(np.clip(base_switch, 0.0, 1.0))

    # -------------------------------------------------------------
    # exercise_minutes: roughly tied to fitness goal weight
    # -------------------------------------------------------------
    exercise_base = 15 + 60 * gw_fitness + rng.normal(0.0, 10.0)
    exercise_minutes = clip(exercise_base, 0.0, 120.0)

    # -------------------------------------------------------------
    # momentum: function of consistency, goal minutes, exercise
    # -------------------------------------------------------------
    norm_minutes = minutes_goal / 120.0
    norm_exercise = exercise_minutes / 60.0

    momentum_raw = (
        0.45 * consistency_index +
        0.30 * norm_minutes +
        0.20 * norm_exercise +
        rng.normal(0.0, 0.05)
    )
    momentum = float(np.clip(momentum_raw, 0.0, 1.0))

    # -------------------------------------------------------------
    # friction_index: opposite of momentum + some noise
    # -------------------------------------------------------------
    friction_raw = 0.8 - momentum + rng.normal(0.0, 0.08)
    friction_index = float(np.clip(friction_raw, 0.0, 1.0))

    # -------------------------------------------------------------
    # alignment_score: combination of all important signals
    # -------------------------------------------------------------
    ratio_goal_vs_baseline = minutes_goal / (b_goal + 1e-3)

    alignment_logit = (
        2.2 * consistency_index +
        1.4 * (ratio_goal_vs_baseline - 0.8) +
        1.0 * ((sleep_hours - 6.5) / 1.5) +
        1.5 * (momentum - 0.5) -
        1.6 * (friction_index - 0.5) -
        1.1 * (task_switch_rate - 0.5) +
        0.8 * (exercise_minutes / 60.0) +
        rng.normal(0.0, 0.6)   # noise for diversity
    )

    alignment_score = float(sigmoid(alignment_logit))

    return {
        "user_id": user_baseline["user_id"],
        "date": date.strftime("%Y-%m-%d"),
        "day_index": day_offset,
        "minutes_goal": minutes_goal,
        "consistency_index": consistency_index,
        "sleep_hours": sleep_hours,
        "momentum": momentum,
        "friction_index": friction_index,
        "task_switch_rate": task_switch_rate,
        "exercise_minutes": exercise_minutes,
        "baseline_goal_minutes": b_goal,
        "baseline_consistency": b_cons,
        "baseline_sleep_hours": b_sleep,
        "goal_weight_career": gw_career,
        "goal_weight_fitness": gw_fitness,
        "goal_weight_learning": gw_learning,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "alignment_score": alignment_score,
    }


# -------------------------------------------------------------------
# MAIN GENERATION PIPELINE
# -------------------------------------------------------------------
def main():
    all_rows = []

    for u in range(NUM_USERS):
        user_baseline = gen_user_baseline(u)
        for d in range(DAYS_PER_USER):
            row = gen_day_row(user_baseline, d)
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    assert len(df) == TOTAL_ROWS, f"Expected {TOTAL_ROWS} rows, got {len(df)}"

    # Convert date and sort for rolling
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)

    # ----------------------------------------------------------------
    # 1) Rolling aggregates to create the 20-core summary features
    # ----------------------------------------------------------------
    group = df.groupby("user_id", group_keys=False)

    # Weekly rolling (7 days, including current day)
    df["avg_minutes_last_week"] = group["minutes_goal"].transform(
        lambda s: s.rolling(window=7, min_periods=1).mean()
    )
    df["avg_consistency_last_week"] = group["consistency_index"].transform(
        lambda s: s.rolling(window=7, min_periods=1).mean()
    )
    df["avg_sleep_last_week"] = group["sleep_hours"].transform(
        lambda s: s.rolling(window=7, min_periods=1).mean()
    )

    # Monthly rolling (30 days, including current day)
    df["avg_minutes_last_month"] = group["minutes_goal"].transform(
        lambda s: s.rolling(window=30, min_periods=1).mean()
    )
    df["momentum_last_month"] = group["momentum"].transform(
        lambda s: s.rolling(window=30, min_periods=1).mean()
    )
    df["friction_last_month"] = group["friction_index"].transform(
        lambda s: s.rolling(window=30, min_periods=1).mean()
    )
    df["avg_sleep_last_month"] = group["sleep_hours"].transform(
        lambda s: s.rolling(window=30, min_periods=1).mean()
    )
    df["task_switch_avg"] = group["task_switch_rate"].transform(
        lambda s: s.rolling(window=30, min_periods=1).mean()
    )
    df["exercise_avg"] = group["exercise_minutes"].transform(
        lambda s: s.rolling(window=30, min_periods=1).mean()
    )

    # ----------------------------------------------------------------
    # 2) Balanced class labels from alignment_score
    # ----------------------------------------------------------------
    median_score = df["alignment_score"].median()
    df["aligned_label"] = (df["alignment_score"] >= median_score).astype(int)

    print("\nClass balance (aligned_label):")
    print(df["aligned_label"].value_counts(normalize=True))

    # ----------------------------------------------------------------
    # 3) Keep only the 20-core schema + label
    # ----------------------------------------------------------------
    cols_20 = [
        "user_id",
        "date",
        "minutes_goal",
        "consistency_index",
        "sleep_hours",
        "avg_minutes_last_week",
        "avg_consistency_last_week",
        "avg_sleep_last_week",
        "avg_minutes_last_month",
        "momentum_last_month",
        "friction_last_month",
        "avg_sleep_last_month",
        "baseline_goal_minutes",
        "baseline_consistency",
        "baseline_sleep_hours",
        "task_switch_avg",
        "exercise_avg",
        "day_of_week",
        "is_weekend",
        "goal_weight_career",
        "goal_weight_fitness",
        "goal_weight_learning",
        "aligned_label",
    ]

    df_20 = df[cols_20].copy()

    # Convert date back to string for CSV consistency
    df_20["date"] = df_20["date"].dt.strftime("%Y-%m-%d")

    # ----------------------------------------------------------------
    # 4) Save CSV
    # ----------------------------------------------------------------
    df_20.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Generated 20-feature dataset: {OUT_CSV}")
    print(f"Rows: {len(df_20)}, Unique users: {df_20['user_id'].nunique()}")
    print("\nAligned label distribution (counts):")
    print(df_20["aligned_label"].value_counts())
    print("\nSample rows:")
    print(df_20.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
