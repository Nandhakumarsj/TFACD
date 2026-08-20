from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Below this, distances are treated as "all clients identical" rather than
# dividing through by a ~zero median and producing meaningless huge OOD scores.
_DEGENERATE_SCALE = 1e-12


@dataclass
class DetectionMetrics:
    """Per-round evidence about HOW a detection decision was reached, not just
    what it decided. Persisted to the trust log so a run's artifacts are
    self-describing - previously `cluster_method` lived only in config, so an
    old trust log couldn't tell you which technique produced it.
    """

    cluster_method: str
    n_clients: int
    n_components: int
    explained_variance_ratio: float
    silhouette: float | None
    benign_cluster_size: int
    distance_fallback_used: bool
    degenerate: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_method": self.cluster_method,
            "n_clients": self.n_clients,
            "n_components": self.n_components,
            "explained_variance_ratio": self.explained_variance_ratio,
            "silhouette": self.silhouette,
            "benign_cluster_size": self.benign_cluster_size,
            "distance_fallback_used": self.distance_fallback_used,
            "degenerate": self.degenerate,
        }


@dataclass
class DetectionResult:
    benign_mask: np.ndarray
    round_scores: np.ndarray
    trust_scores: np.ndarray
    projected_updates: np.ndarray
    ood_scores: np.ndarray
    metrics: DetectionMetrics


@dataclass
class PCAClusterEMAFilter:
    """FedDMC-inspired baseline, not an exact FedDMC reproduction.

    It combines PCA, a standard clustering method, and an EMA history correction.
    The paper's custom BTBCN should be reproduced separately if exact comparison is required.

    Three distinct per-client signals come out of `detect()`, and they answer
    different questions - reporting one in place of another overstates what was measured:

    - `round_scores` (0/1): did THIS round's clustering put the client in the
      benign cluster? Memoryless.
    - `ood_scores` (>=0): how far is this client's update from the cohort's robust
      center in PCA space, as a multiple of the cohort's MEDIAN distance? 1.0 is
      typical, 2.0 is twice as far out as the median client. Continuous and
      unbounded above, so it ranks clients even when clustering puts them all in
      one bucket - which is exactly the case where `round_scores` conveys nothing.
    - `trust_scores` (0..1): the EMA-smoothed history of `round_scores`. This is
      the per-client personalization signal, and the only one of the three that
      carries across rounds. Gate 4 measured that a single bad round does NOT sink
      a client here by design - it takes several consistent rounds to cross
      `reject_below_trust`, which is why robust aggregation is kept as
      defense-in-depth rather than relying on this alone.
    """

    pca_components: int = 5
    cluster_method: str = "agglomerative"
    min_benign_fraction: float = 0.5
    ema_alpha: float = 0.3
    reject_below_trust: float = 0.35
    history: dict[str, float] = field(default_factory=dict)

    def _update_trust(self, client_ids: list[str], round_scores: np.ndarray) -> np.ndarray:
        trust = np.zeros(len(client_ids), dtype=np.float64)
        for index, client_id in enumerate(client_ids):
            old = self.history.get(client_id, 1.0)
            new = self.ema_alpha * round_scores[index] + (1.0 - self.ema_alpha) * old
            self.history[client_id] = float(new)
            trust[index] = new
        return trust

    def detect(self, client_ids: list[str], update_vectors: np.ndarray) -> DetectionResult:
        if len(client_ids) != len(update_vectors):
            raise ValueError("client_ids and update_vectors length mismatch")
        n_clients = len(client_ids)
        if n_clients < 3:
            # Too few clients for clustering to mean anything: accept all, but
            # still advance EMA trust so history stays continuous across rounds
            # where a federation temporarily shrinks.
            scores = np.ones(n_clients, dtype=np.float64)
            trust = self._update_trust(client_ids, scores)
            metrics = DetectionMetrics(
                cluster_method="none:too-few-clients", n_clients=n_clients, n_components=0,
                explained_variance_ratio=0.0, silhouette=None, benign_cluster_size=n_clients,
                distance_fallback_used=False, degenerate=True,
            )
            return DetectionResult(
                np.ones(n_clients, dtype=bool), scores, trust, update_vectors, np.zeros(n_clients, dtype=np.float64), metrics
            )

        scaled = StandardScaler().fit_transform(update_vectors)
        components = max(1, min(self.pca_components, n_clients - 1, scaled.shape[1]))
        pca = PCA(n_components=components, random_state=0)
        # Not a rare synthetic case: client updates naturally converge toward
        # each other as federated training approaches convergence, driving
        # StandardScaler's per-feature variance toward 0 - sklearn's own PCA
        # then divides by a ~0 total_var and emits a RuntimeWarning on every
        # such round. Suppressed here because the NaN it produces is handled
        # explicitly below, not silently swallowed.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
            projected = pca.fit_transform(scaled)
        explained_variance_ratio = float(pca.explained_variance_ratio_.sum())
        if not np.isfinite(explained_variance_ratio):
            # Verified live: an unguarded NaN here makes json.dumps() silently
            # emit the invalid JSON literal `NaN` into the trust log, which a
            # strict JSON parser - e.g. any non-Python reader - rejects.
            explained_variance_ratio = 0.0

        if self.cluster_method == "dbscan":
            labels = DBSCAN(eps=1.5, min_samples=max(2, int(np.ceil(n_clients * 0.2)))).fit_predict(projected)
            valid = labels[labels >= 0]
            if len(valid) == 0:
                benign_cluster = -1
            else:
                values, counts = np.unique(valid, return_counts=True)
                benign_cluster = int(values[np.argmax(counts)])
        else:
            labels = AgglomerativeClustering(n_clusters=2).fit_predict(projected)
            values, counts = np.unique(labels, return_counts=True)
            benign_cluster = int(values[np.argmax(counts)])

        round_scores = (labels == benign_cluster).astype(np.float64)

        # Continuous OOD signal, computed for every round rather than only in the
        # fallback branch below - it is the one per-client signal that still
        # discriminates when clustering collapses everyone into one label.
        center = np.median(projected, axis=0)
        distances = np.linalg.norm(projected - center, axis=1)
        median_distance = float(np.median(distances))
        degenerate = median_distance <= _DEGENERATE_SCALE
        # A ~zero median means the cohort is effectively identical (or >half sit
        # exactly on the center); dividing by it would manufacture enormous OOD
        # scores out of floating-point noise.
        ood_scores = np.zeros(n_clients, dtype=np.float64) if degenerate else distances / median_distance

        # If clustering labels too few clients benign, retain the closest clients to the robust center.
        minimum = max(1, int(np.ceil(self.min_benign_fraction * n_clients)))
        distance_fallback_used = int(round_scores.sum()) < minimum
        if distance_fallback_used:
            keep = np.argsort(distances)[:minimum]
            round_scores[:] = 0.0
            round_scores[keep] = 1.0

        # silhouette_score is only defined for 2..n-1 distinct labels; DBSCAN can
        # legitimately produce a single label (all noise, or all one cluster).
        unique_labels = np.unique(labels)
        silhouette: float | None = None
        if 2 <= len(unique_labels) <= n_clients - 1:
            try:
                silhouette = float(silhouette_score(projected, labels))
            except ValueError:
                silhouette = None

        trust = self._update_trust(client_ids, round_scores)
        benign_mask = trust >= self.reject_below_trust
        metrics = DetectionMetrics(
            cluster_method=self.cluster_method, n_clients=n_clients, n_components=components,
            explained_variance_ratio=explained_variance_ratio, silhouette=silhouette,
            benign_cluster_size=int(round_scores.sum()), distance_fallback_used=distance_fallback_used,
            degenerate=degenerate,
        )
        return DetectionResult(benign_mask, round_scores, trust, projected, ood_scores, metrics)
