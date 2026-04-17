"""
ML-based Funnel Impact Simulator

Trains logistic regression models on user features (traffic_source, device,
country) to predict per-user conversion probabilities at each funnel stage:
  - P(signup | visit)
  - P(activation | signup)
  - P(purchase  | activation)

Simulates an "experiment lift" by multiplicatively boosting each user's
predicted log-odds for the targeted stage, then aggregates the new expected
funnel counts user-by-user. Returns model quality metrics and feature
importances so users can see WHY the model behaves the way it does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score


FEATURE_COLS = ["traffic_source", "device", "country"]

STAGE_DEFS = [
    ("conv1", "visited", "signed_up", "Visit \u2192 Signup"),
    ("conv2", "signed_up", "activated", "Signup \u2192 Activation"),
    ("conv3", "activated", "purchased", "Activation \u2192 Purchase"),
]


def _prepare_features(user_flags: pd.DataFrame) -> tuple:
    """Build a one-hot encoded feature matrix from categorical user attrs."""
    df = user_flags.copy()
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown").astype(str)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X = encoder.fit_transform(df[FEATURE_COLS])
    feature_names = encoder.get_feature_names_out(FEATURE_COLS).tolist()
    return X, feature_names, encoder


def _train_stage_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
) -> dict:
    """Train a logistic regression for a single stage transition.

    Returns predicted probabilities for ALL rows in X plus quality metrics.
    """
    n = len(y)
    n_pos = int(y.sum())
    n_neg = int(n - n_pos)

    if n < 20 or n_pos < 5 or n_neg < 5:
        prob = float(n_pos / n) if n > 0 else 0.0
        return {
            "trained": False,
            "reason": (
                f"Not enough data to train (need \u226520 samples with \u22655 of each class; "
                f"got {n} samples, {n_pos} positive)."
            ),
            "probs": np.full(n, prob),
            "auc": None,
            "accuracy": None,
            "n_samples": n,
            "n_positive": n_pos,
            "feature_importance": pd.DataFrame(),
            "intercept": 0.0,
            "coefs": np.zeros(X.shape[1]),
        }

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, solver="liblinear")
    model.fit(X_tr, y_tr)

    y_pred_prob = model.predict_proba(X_te)[:, 1]
    y_pred = model.predict(X_te)

    try:
        auc = float(roc_auc_score(y_te, y_pred_prob))
    except ValueError:
        auc = None
    acc = float(accuracy_score(y_te, y_pred))

    full_probs = model.predict_proba(X)[:, 1]
    coefs = model.coef_[0]
    intercept = float(model.intercept_[0])

    fi = (
        pd.DataFrame({"feature": feature_names, "coefficient": coefs})
        .assign(abs_coef=lambda d: d["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
        .reset_index(drop=True)
    )

    return {
        "trained": True,
        "reason": None,
        "probs": full_probs,
        "auc": auc,
        "accuracy": acc,
        "n_samples": n,
        "n_positive": n_pos,
        "feature_importance": fi,
        "intercept": intercept,
        "coefs": coefs,
    }


def train_funnel_models(user_flags: pd.DataFrame) -> dict:
    """Train one logistic regression per funnel transition.

    Each transition is trained on the conditioned subset (e.g. activation
    model is trained only on users who signed up).
    """
    X_all, feature_names, encoder = _prepare_features(user_flags)

    out = {
        "feature_names": feature_names,
        "encoder": encoder,
        "X_all": X_all,
        "stages": {},
    }

    for key, source_col, target_col, label in STAGE_DEFS:
        if source_col not in user_flags.columns or target_col not in user_flags.columns:
            continue

        mask = user_flags[source_col].astype(bool).values
        X_sub = X_all[mask]
        y_sub = user_flags.loc[mask, target_col].astype(int).values

        stage_result = _train_stage_model(X_sub, y_sub, feature_names)
        stage_result["label"] = label
        stage_result["mask"] = mask
        out["stages"][key] = stage_result

    return out


def _apply_lift_to_probs(probs: np.ndarray, lift: float) -> np.ndarray:
    """Apply a lift via odds-ratio update: new_odds = old_odds * (1 + lift).

    This is the standard Bayesian way to apply a relative lift to a
    probability. It's monotonic in `lift` (positive lift => higher
    probability for every user), automatically clips to [0, 1], and
    treats high-probability users with diminishing returns
    (a 15% odds boost on p=0.9 yields a smaller absolute change than
     on p=0.5, which is realistic for a "UX improvement" lever).
    """
    if lift == 0:
        return probs.copy()
    p = np.clip(probs, 1e-6, 1 - 1e-6)
    odds = p / (1 - p)
    new_odds = odds * (1 + lift)
    new_probs = new_odds / (1 + new_odds)
    return np.clip(new_probs, 0.0, 1.0)


def simulate_with_models(
    user_flags: pd.DataFrame,
    models: dict,
    lift1: float,
    lift2: float,
    lift3: float,
) -> dict:
    """Run the ML-based what-if simulation.

    Returns baseline + simulated counts and per-user expected outcomes.
    """
    n_users = len(user_flags)
    visited_mask = user_flags["visited"].astype(bool).values
    visited = int(visited_mask.sum())

    stages = models["stages"]
    s1, s2, s3 = stages.get("conv1"), stages.get("conv2"), stages.get("conv3")

    actual_signed = int(user_flags["signed_up"].astype(bool).sum())
    actual_active = int(user_flags["activated"].astype(bool).sum())
    actual_purchased = int(user_flags["purchased"].astype(bool).sum())

    def _ratio(stage, lift):
        """Multiplicative ratio: model's predicted positives WITH lift vs WITHOUT."""
        if stage is None:
            return 1.0 + lift
        if not stage["trained"]:
            return 1.0 + lift
        baseline_sum = float(stage["probs"].sum())
        if baseline_sum <= 0:
            return 1.0 + lift
        lifted_sum = float(_apply_lift_to_probs(stage["probs"], lift).sum())
        return lifted_sum / baseline_sum

    ratio1 = _ratio(s1, lift1)
    ratio2 = _ratio(s2, lift2)
    ratio3 = _ratio(s3, lift3)

    sim_signed = actual_signed * ratio1
    sim_signed = min(sim_signed, float(visited))

    sim_active = actual_active * ratio1 * ratio2
    sim_active = min(sim_active, sim_signed)

    sim_purchase = actual_purchased * ratio1 * ratio2 * ratio3
    sim_purchase = min(sim_purchase, sim_active)

    total_revenue = float(user_flags["revenue"].sum())
    revenue_per_purchase = (
        total_revenue / actual_purchased if actual_purchased > 0 else 0.0
    )
    sim_revenue = sim_purchase * revenue_per_purchase

    return {
        "baseline": {
            "visited": visited,
            "signed_up": actual_signed,
            "activated": actual_active,
            "purchased": actual_purchased,
            "total_revenue": total_revenue,
        },
        "simulated": {
            "visited": visited,
            "signed_up": sim_signed,
            "activated": sim_active,
            "purchased": sim_purchase,
            "total_revenue": sim_revenue,
        },
        "ratios": {"conv1": ratio1, "conv2": ratio2, "conv3": ratio3},
        "lifts": (lift1, lift2, lift3),
        "revenue_per_purchase": revenue_per_purchase,
    }


def compute_ml_deltas(simulation: dict) -> dict:
    b = simulation["baseline"]
    s = simulation["simulated"]
    delta_signups = s["signed_up"] - b["signed_up"]
    delta_activations = s["activated"] - b["activated"]
    delta_purchases = s["purchased"] - b["purchased"]
    delta_revenue = s["total_revenue"] - b["total_revenue"]

    base_overall = (b["purchased"] / b["visited"]) if b["visited"] > 0 else 0
    sim_overall = (s["purchased"] / s["visited"]) if s["visited"] > 0 else 0
    overall_lift_pct = (
        ((sim_overall - base_overall) / base_overall * 100)
        if base_overall > 0
        else 0
    )

    return {
        "delta_signups": delta_signups,
        "delta_activations": delta_activations,
        "delta_purchases": delta_purchases,
        "delta_revenue": delta_revenue,
        "overall_lift_pct": overall_lift_pct,
        "baseline_overall": base_overall,
        "simulated_overall": sim_overall,
    }


def generate_ml_insight(
    models: dict,
    simulation: dict,
    deltas: dict,
) -> str:
    lift1, lift2, lift3 = simulation["lifts"]
    if lift1 == 0 and lift2 == 0 and lift3 == 0:
        return (
            "Adjust the sliders or pick a preset to run the ML model and "
            "see projected outcomes. The model is trained on user features "
            "(traffic source, device, country) per funnel stage."
        )

    parts = []
    parts.append(
        f"Model predicts **{deltas['delta_purchases']:+,.0f} additional purchases** "
        f"and **${deltas['delta_revenue']:+,.0f} additional revenue** under this scenario."
    )

    stage_aucs = []
    for key, label in [("conv1", "Signup"), ("conv2", "Activation"), ("conv3", "Purchase")]:
        s = models["stages"].get(key)
        if s and s["trained"] and s["auc"] is not None:
            stage_aucs.append(f"{label} AUC={s['auc']:.2f}")
    if stage_aucs:
        parts.append("Model quality: " + ", ".join(stage_aucs) + ".")

    top_drivers = []
    for key, label in [("conv1", "signup"), ("conv2", "activation"), ("conv3", "purchase")]:
        s = models["stages"].get(key)
        if s and s["trained"] and not s["feature_importance"].empty:
            top = s["feature_importance"].iloc[0]
            direction = "drives" if top["coefficient"] > 0 else "hurts"
            top_drivers.append(f"`{top['feature']}` {direction} {label}")
    if top_drivers:
        parts.append("Top learned drivers: " + "; ".join(top_drivers) + ".")

    return " ".join(parts)
