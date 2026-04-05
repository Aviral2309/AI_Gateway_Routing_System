"""
Retrain the query classifier on real logged data.
Run weekly after collecting real traffic.

Usage: python scripts/retrain_classifier.py
"""
import asyncio
import sys
import os
import pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    print("=" * 55)
    print("  Nexus — Classifier Retraining")
    print("=" * 55)

    from app.db.session import get_db
    from app.db.models import RequestLog
    from sqlalchemy import select

    async with get_db() as db:
        r = await db.execute(
            select(RequestLog.query, RequestLog.tier, RequestLog.quality_score)
            .where(RequestLog.success == True)
            .order_by(RequestLog.created_at.desc())
            .limit(5000)
        )
        rows = r.all()

    print(f"\nFetched {len(rows)} examples from DB")

    if len(rows) < 30:
        print(f"Not enough data. Need ≥30. Have {len(rows)}. Skipping.")
        return

    from app.classifier.feature_extractor import FeatureExtractor
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    tier_map = {"simple": 0, "medium": 1, "complex": 2}
    data = [(q, tier_map[t], s or 0.5) for q, t, s in rows if t in tier_map]

    extractor = FeatureExtractor()
    X = np.array([extractor.extract(q).to_vector() for q, _, _ in data])
    y = np.array([l for _, l, _ in data])
    w = np.array([s for _, _, s in data])

    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8,
        class_weight="balanced", random_state=42
    )
    model.fit(Xs, y, sample_weight=w)

    scores = cross_val_score(model, Xs, y, cv=5, scoring="accuracy")
    print(f"\nCross-val accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    names = {0: "simple", 1: "medium", 2: "complex"}
    u, c  = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for ui, ci in zip(u, c):
        print(f"  {names[ui]}: {ci} ({ci/len(y)*100:.1f}%)")

    path = "classifier_model.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    print(f"\n✓ Saved to {path}")
    print("  Restart the server to load the new model.")


if __name__ == "__main__":
    asyncio.run(main())
