"""Small shims so the package runs on the older scientific stack some services pin
(Python 3.9, numpy 1.22, pandas 1.1.5, scikit-learn 0.24)."""
from __future__ import annotations

import inspect


def agglomerative(n_clusters=None, metric: str = "euclidean", **kw):
    """``sklearn.cluster.AgglomerativeClustering`` with the distance argument named for the installed version
    (``metric`` since scikit-learn 1.2, ``affinity`` before)."""
    from sklearn.cluster import AgglomerativeClustering
    params = inspect.signature(AgglomerativeClustering.__init__).parameters
    key = "metric" if "metric" in params else "affinity"
    return AgglomerativeClustering(n_clusters=n_clusters, **{key: metric}, **kw)
