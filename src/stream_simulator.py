"""
stream_simulator.py
Real-time streaming simulation and two-tier cost-aware screening architecture.
Directly addresses Research Gap 6 and Table 1 Papers [3], [9], [12]
(Abbassi et al. 2024, Shamsinejad et al. 2026, Zelasko 2026).
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Generator


class TwoTierCostAwareDetector:
    """
    Two-Tier Cost-Aware Fraud Screening:
    - Tier 1 (Lightweight Filter): Sub-millisecond rule & isolation check.
      Bypasses 90%+ benign transactions instantly to conserve computational resources.
    - Tier 2 (Deep Hybrid Engine): Comprehensive ML ensemble & XAI invoked only for suspects.
    """
    def __init__(self, tier1_fast_model, tier2_deep_model, tier1_cutoff: float = 0.20):
        self.fast_model = tier1_fast_model
        self.deep_model = tier2_deep_model
        self.tier1_cutoff = tier1_cutoff
        self.stats = {
            "total_screened": 0,
            "tier1_fast_passed": 0,
            "tier2_deep_escalated": 0,
            "total_time_ms": 0.0,
            "baseline_full_time_ms": 0.0
        }

    def score_transaction(self, X_row: pd.DataFrame) -> Dict[str, Any]:
        """Scores a single incoming transaction through the two-tier cost-aware pipeline."""
        t_start = time.perf_counter()
        self.stats["total_screened"] += 1

        # Tier 1: Fast lightweight screening
        fast_score = float(self.fast_model.predict_anomaly_score(X_row)[0])

        if fast_score < self.tier1_cutoff:
            # Passed fast filter
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            self.stats["tier1_fast_passed"] += 1
            self.stats["total_time_ms"] += latency_ms
            return {
                "final_risk_score": round(fast_score * 50.0, 1), # scaled low risk
                "tier_used": "Tier 1 (Fast Lightweight Filter)",
                "latency_ms": round(latency_ms, 3),
                "status": "APPROVED_AUTOMATIC",
                "escalated": False
            }
        else:
            # Escalated to Tier 2 Deep Evaluation
            self.stats["tier2_deep_escalated"] += 1
            deep_prob = float(self.deep_model.predict_proba(X_row)[0, 1])
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            self.stats["total_time_ms"] += latency_ms
            return {
                "final_risk_score": round(deep_prob * 100.0, 1),
                "tier_used": "Tier 2 (Deep Hybrid Engine)",
                "latency_ms": round(latency_ms, 3),
                "status": "FLAGGED_FOR_REVIEW" if deep_prob >= 0.5 else "APPROVED_VERIFIED",
                "escalated": True
            }

    def benchmark_stream(self, X_sample: pd.DataFrame) -> Dict[str, Any]:
        """Runs batch simulation to compare two-tier cost-aware latency vs naive full-tier latency."""
        n = len(X_sample)
        if n == 0:
            return {}

        # 1. Benchmark Naive Full Deep Model on all transactions
        t0 = time.perf_counter()
        _ = self.deep_model.predict_proba(X_sample)
        full_deep_time = time.perf_counter() - t0
        avg_full_latency_ms = (full_deep_time / n) * 1000.0

        # 2. Benchmark Two-Tier Architecture
        t0 = time.perf_counter()
        fast_scores = self.fast_model.predict_anomaly_score(X_sample)
        escalated_idx = np.where(fast_scores >= self.tier1_cutoff)[0]
        if len(escalated_idx) > 0:
            _ = self.deep_model.predict_proba(X_sample.iloc[escalated_idx])
        two_tier_time = time.perf_counter() - t0
        avg_two_tier_latency_ms = (two_tier_time / n) * 1000.0

        speedup_pct = max(0.0, (avg_full_latency_ms - avg_two_tier_latency_ms) / max(1e-5, avg_full_latency_ms) * 100.0)

        return {
            "transactions_simulated": n,
            "tier1_fast_pass_count": int(n - len(escalated_idx)),
            "tier1_fast_pass_pct": round((n - len(escalated_idx)) / n * 100.0, 1),
            "tier2_escalated_count": int(len(escalated_idx)),
            "tier2_escalated_pct": round(len(escalated_idx) / n * 100.0, 1),
            "naive_full_latency_ms": round(avg_full_latency_ms, 3),
            "two_tier_latency_ms": round(avg_two_tier_latency_ms, 3),
            "latency_reduction_pct": round(speedup_pct, 1),
            "throughput_boost": f"{avg_full_latency_ms / max(1e-5, avg_two_tier_latency_ms):.1f}x"
        }
