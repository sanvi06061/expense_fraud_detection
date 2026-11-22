"""
explainability.py
Explainable AI (XAI) and Human Decision Support Engine for Financial Anomaly Detection.
Directly addresses Research Gap 5 and Table 1 Papers [11], [15], [17], [27], [29].
Computes global feature importances, local instance risk attributions, and
executive investigator narratives.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any


def get_global_feature_importances(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Extracts and ranks global feature importances from tree-based or hybrid models.
    """
    importances = None

    # Handle pipeline or raw model
    clf = model
    if hasattr(model, "pipeline") and hasattr(model.pipeline, "named_steps"):
        clf = model.pipeline.named_steps.get("clf", model)
    elif hasattr(model, "classifier"):
        clf = model.classifier

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        # Fallback uniform/variance based
        importances = np.ones(len(feature_names)) / len(feature_names)

    # Align lengths
    names = feature_names[:len(importances)] if len(feature_names) >= len(importances) else [f"f_{i}" for i in range(len(importances))]
    df_imp = pd.DataFrame({
        "feature": names,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    df_imp["relative_pct"] = (df_imp["importance"] / df_imp["importance"].sum() * 100).round(2)
    return df_imp


def compute_local_instance_attribution(
    row: pd.Series,
    baseline_means: pd.Series,
    baseline_stds: pd.Series,
    feature_importances: pd.DataFrame,
    top_k: int = 6
) -> pd.DataFrame:
    """
    Computes local feature contribution (risk shift) for a single transaction instance
    relative to legitimate transaction baselines.
    Positive attribution indicates the feature increased fraud risk.
    Negative attribution indicates the feature provided mitigating/legitimate evidence.
    """
    imp_dict = dict(zip(feature_importances["feature"], feature_importances["importance"]))

    attributions = []
    for feat in feature_importances["feature"]:
        if feat not in row.index:
            continue
        val = float(row[feat])
        mean_val = float(baseline_means.get(feat, 0.0))
        std_val = float(baseline_stds.get(feat, 1.0))
        if std_val < 1e-5:
            std_val = 1.0

        weight = imp_dict.get(feat, 0.01)
        # Directional shift: high value increases risk for velocity, amount, night, etc.
        z_score = (val - mean_val) / std_val
        
        # Invert direction for features where higher value is safer (e.g. cardPresent, cvv_match)
        if feat in ["cardPresent", "cvv_match"]:
            z_score = -z_score

        # Approximate local attribution score
        contribution = round(float(z_score * weight * 10.0), 3)

        attributions.append({
            "feature": feat,
            "actual_value": round(val, 2),
            "baseline_mean": round(mean_val, 2),
            "risk_contribution": contribution,
            "direction": "Risk Escalator (+)" if contribution > 0 else "Risk Mitigator (-)"
        })

    df_local = pd.DataFrame(attributions)
    # Sort by absolute impact
    df_local["abs_impact"] = df_local["risk_contribution"].abs()
    df_local = df_local.sort_values("abs_impact", ascending=False).head(top_k).drop(columns=["abs_impact"]).reset_index(drop=True)
    return df_local


def generate_executive_investigation_brief(
    row: pd.Series,
    risk_score: float,
    threshold: float,
    local_attrib: pd.DataFrame,
    rule_flags: str
) -> str:
    """
    Generates a structured, human-in-the-loop audit memorandum for compliance investigators.
    """
    decision = "🚨 ESCALATE FOR IMMEDIATE INVESTIGATION" if risk_score >= (threshold * 100) else "✅ PASS / MONITOR"
    
    top_drivers = local_attrib[local_attrib["risk_contribution"] > 0]
    top_driver_bullets = "\n".join([
        f"- **{r['feature']}**: observed value `{r['actual_value']}` vs typical `{r['baseline_mean']}` (+{r['risk_contribution']:.2f} risk points)"
        for _, r in top_drivers.head(3).iterrows()
    ]) if not top_drivers.empty else "- No statistical anomaly driver exceeded alert thresholds."

    brief = f"""### Transaction Audit & Risk Memorandum
**Decision Status**: {decision}
- **Assessed Risk Score**: `{risk_score:.1f} / 100` (Calibrated Decision Threshold: `{threshold * 100:.1f}`)
- **Account / Transaction ID**: `{row.get('accountNumber', 'N/A')}` | Amount: `${float(row.get('transactionAmount', 0)):,.2f}`
- **Merchant**: `{row.get('merchantName', 'N/A')}` (`{row.get('merchantCategoryCode', 'N/A')}`)
- **Timestamp**: `{row.get('transactionDateTime', 'N/A')}`

#### Top Local Statistical Risk Drivers:
{top_driver_bullets}

#### Behavioral Rule Flags:
- {rule_flags}

**Investigator Recommendation**:
{"Place a temporary authorization hold and send immediate two-factor verification to cardholder." if risk_score >= (threshold*100) else "Permit transaction under standard continuous telemetry monitoring."}
"""
    return brief
