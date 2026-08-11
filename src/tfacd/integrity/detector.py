from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class DetectionResult:
    benign_mask: np.ndarray
    round_scores: np.ndarray
    trust_scores: np.ndarray
    projected_updates: np.ndarray


@dataclass
class PCAClusterEMAFilter:
    """FedDMC-inspired baseline, not an exact FedDMC reproduction.

    It combines PCA, a standard clustering method, and an EMA history correction.
    The paper's custom BTBCN should be reproduced separately if exact comparison is required.
    """

    pca_components: int = 5
    cluster_method: str = "agglomerative"
    min_benign_fraction: float = 0.5
    ema_alpha: float = 0.3
    reject_below_trust: float = 0.35
    history: dict[str, float] = field(default_factory=dict)

    def detect(self, client_ids: list[str], update_vectors: np.ndarray) -> DetectionResult:
        if len(client_ids) != len(update_vectors):
            raise ValueError("client_ids and update_vectors length mismatch")
        n_clients = len(client_ids)
        if n_clients < 3:
            scores = np.ones(n_clients, dtype=np.float64)
            return DetectionResult(np.ones(n_clients, dtype=bool), scores, scores, update_vectors)

        scaled = StandardScaler().fit_transform(update_vectors)
        components = max(1, min(self.pca_components, n_clients - 1, scaled.shape[1]))
        projected = PCA(n_components=components, random_state=0).fit_transform(scaled)

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
        # If clustering labels too few clients benign, retain the closest clients to the robust center.
        minimum = max(1, int(np.ceil(self.min_benign_fraction * n_clients)))
        if int(round_scores.sum()) < minimum:
            center = np.median(projected, axis=0)
            distance = np.linalg.norm(projected - center, axis=1)
            keep = np.argsort(distance)[:minimum]
            round_scores[:] = 0.0
            round_scores[keep] = 1.0

        trust = np.zeros(n_clients, dtype=np.float64)
        for index, client_id in enumerate(client_ids):
            old = self.history.get(client_id, 1.0)
            new = self.ema_alpha * round_scores[index] + (1.0 - self.ema_alpha) * old
            self.history[client_id] = float(new)
            trust[index] = new
        benign_mask = trust >= self.reject_below_trust
        return DetectionResult(benign_mask, round_scores, trust, projected)
