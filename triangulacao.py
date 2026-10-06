"""
Triangulação: CHD (Reinert, lexical) × BERTopic (semântico, embeddings).

As duas partições dos mesmos segmentos são comparadas com:
- ARI (índice de Rand ajustado): 0 = acaso, 1 = partições idênticas
- NMI (informação mútua normalizada): 0 = independentes, 1 = idênticas
- tabela cruzada classes × tópicos e o tópico predominante de cada classe

Concordância alta reforça que as classes não são artefato do método lexical;
divergências apontam classes a examinar com mais cuidado.

  pip install -r requirements-topicos.txt   (bertopic, sentence-transformers…)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import comb, log
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def contingency(a: Sequence[int], b: Sequence[int]):
    la, lb = sorted(set(a)), sorted(set(b))
    ia, ib = {x: i for i, x in enumerate(la)}, {x: i for i, x in enumerate(lb)}
    table = np.zeros((len(la), len(lb)), dtype=int)
    for x, y in zip(a, b):
        table[ia[x], ib[y]] += 1
    return la, lb, table


def adjusted_rand_index(a: Sequence[int], b: Sequence[int]) -> float:
    _, _, t = contingency(a, b)
    n = int(t.sum())
    if n < 2:
        return 1.0
    sum_ij = sum(comb(int(v), 2) for v in t.ravel())
    sum_a = sum(comb(int(v), 2) for v in t.sum(axis=1))
    sum_b = sum(comb(int(v), 2) for v in t.sum(axis=0))
    expected = sum_a * sum_b / comb(n, 2)
    max_index = (sum_a + sum_b) / 2
    if max_index == expected:
        return 1.0
    return (sum_ij - expected) / (max_index - expected)


def normalized_mutual_info(a: Sequence[int], b: Sequence[int]) -> float:
    """NMI com normalização pela média aritmética das entropias (padrão do scikit-learn)."""
    _, _, t = contingency(a, b)
    n = t.sum()
    if n == 0:
        return 1.0
    pa, pb, p = t.sum(axis=1) / n, t.sum(axis=0) / n, t / n
    ha = -sum(x * log(x) for x in pa if x > 0)
    hb = -sum(x * log(x) for x in pb if x > 0)
    mi = sum(p[i, j] * log(p[i, j] / (pa[i] * pb[j]))
             for i in range(t.shape[0]) for j in range(t.shape[1]) if p[i, j] > 0)
    if ha == 0 and hb == 0:
        return 1.0
    denom = (ha + hb) / 2
    return float(mi / denom) if denom > 0 else 0.0


@dataclass
class Triangulation:
    method: str
    n_segments: int
    ari: float
    nmi: float
    class_ids: List[int]
    topic_ids: List[int]
    table: List[List[int]]                     # classes × tópicos
    topic_words: Dict[int, List[str]] = field(default_factory=dict)
    best_topic: Dict[int, Dict] = field(default_factory=dict)   # classe → {tópico, % da classe}
    outliers: int = 0

    def summary(self) -> str:
        return (f"ARI = {self.ari:.2f} e NMI = {self.nmi:.2f} entre as classes da CHD e os tópicos "
                f"do {self.method} (n = {self.n_segments} segmentos"
                + (f"; {self.outliers} segmentos sem tópico excluídos" if self.outliers else "")
                + f"). {interpret_ari(self.ari)}")

    def to_dict(self) -> Dict:
        return asdict(self)


def interpret_ari(ari: float) -> str:
    if ari >= 0.65:
        return "Alta convergência: os dois métodos identificam essencialmente os mesmos agrupamentos."
    if ari >= 0.35:
        return "Convergência moderada: parte das classes tem correspondência semântica clara."
    if ari >= 0.10:
        return "Convergência baixa: examine as classes sem tópico predominante."
    return "Sem convergência além do acaso: os agrupamentos lexicais e semânticos divergem."


def compare(chd: Sequence[int], topics: Sequence[int], method: str = "BERTopic",
            topic_words: Optional[Dict[int, List[str]]] = None) -> Triangulation:
    """Compara partições; ignora segmentos sem classe (0) e outliers do BERTopic (-1)."""
    pairs = [(c, t) for c, t in zip(chd, topics) if c and c > 0 and t is not None and t >= 0]
    outliers = sum(1 for c, t in zip(chd, topics) if c and c > 0 and (t is None or t < 0))
    if not pairs:
        raise ValueError("Nenhum segmento com classe e tópico para comparar")
    a, b = [p[0] for p in pairs], [p[1] for p in pairs]
    la, lb, t = contingency(a, b)
    best = {}
    for i, cid in enumerate(la):
        j = int(t[i].argmax())
        best[cid] = {"topico": lb[j], "pct_classe": round(100 * t[i, j] / t[i].sum(), 1)}
    return Triangulation(method, len(pairs), adjusted_rand_index(a, b), normalized_mutual_info(a, b),
                         la, lb, t.tolist(), topic_words or {}, best, outliers)


# ---------------------------------------------------------------------------
# BERTopic
# ---------------------------------------------------------------------------

def run_bertopic(texts: List[str], n_topics: Optional[int] = None, seed: int = 42,
                 embedding_model: str = DEFAULT_EMBEDDING_MODEL, language_stopwords: str = "pt"):
    """
    Ajusta o BERTopic. Com n_topics, usa KMeans (mesmo número de grupos da CHD,
    comparação direta); sem, usa HDBSCAN (número de tópicos automático, com outliers -1).
    Retorna (tópicos por segmento, palavras por tópico).
    """
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
    except ImportError as e:
        raise ImportError("Triangulação requer: pip install -r requirements-topicos.txt") from e

    stop = None
    if language_stopwords == "pt":
        from corpus import EXTRA_STOPWORDS
        from translator import _STOPWORDS
        stop = sorted(_STOPWORDS["pt"] | set(EXTRA_STOPWORDS["pt"])
                      | {"a", "o", "e", "de", "do", "da", "dos", "das", "um", "uma",
                         "os", "as", "que", "em", "para", "por", "com", "se"})

    n = len(texts)
    umap_model = UMAP(n_neighbors=max(2, min(15, n - 1)), n_components=5, min_dist=0.0,
                      metric="cosine", random_state=seed)
    kwargs = {}
    if n_topics:
        from sklearn.cluster import KMeans

        kwargs["hdbscan_model"] = KMeans(n_clusters=n_topics, random_state=seed, n_init=10)
    else:
        from hdbscan import HDBSCAN

        kwargs["hdbscan_model"] = HDBSCAN(min_cluster_size=max(3, n // 50), prediction_data=True)
    model = BERTopic(
        embedding_model=SentenceTransformer(embedding_model),
        umap_model=umap_model,
        vectorizer_model=CountVectorizer(stop_words=stop, min_df=1),
        calculate_probabilities=False,
        **kwargs,
    )
    topics, _ = model.fit_transform(texts)
    words = {
        int(t): [w for w, _ in (model.get_topic(t) or [])][:8]
        for t in set(topics) if t != -1
    }
    return [int(t) for t in topics], words


# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------

def plot_crosstab(tri: Triangulation, names: Dict[int, str], path: Path) -> Path:
    """Mapa de calor classes × tópicos (% de cada classe), rampa sequencial de um só tom."""
    from matplotlib.colors import LinearSegmentedColormap

    from analises import SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, _mpl

    plt = _mpl()
    t = np.asarray(tri.table, dtype=float)
    pct = 100 * t / np.maximum(t.sum(axis=1, keepdims=True), 1)
    ramp = LinearSegmentedColormap.from_list(
        "azul", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(figsize=(1.1 * len(tri.topic_ids) + 3.5, 0.55 * len(tri.class_ids) + 1.8))
    ax.imshow(pct, cmap=ramp, vmin=0, vmax=100, aspect="auto")
    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            if t[i, j]:
                ax.text(j, i, f"{pct[i, j]:.0f}%", ha="center", va="center", fontsize=8,
                        color="#ffffff" if pct[i, j] > 55 else TEXT_PRIMARY)
    ax.set_yticks(range(len(tri.class_ids)),
                  [f"Classe {c} — {names.get(c, '')}".rstrip(" —")[:38] for c in tri.class_ids],
                  fontsize=8, color=TEXT_PRIMARY)
    ax.set_xticks(range(len(tri.topic_ids)),
                  [f"T{tp}\n{' · '.join(tri.topic_words.get(tp, [])[:2])}" for tp in tri.topic_ids],
                  fontsize=7, color=TEXT_SECONDARY)
    ax.set_xticks(np.arange(-0.5, len(tri.topic_ids)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(tri.class_ids)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(f"CHD × {tri.method}: % de cada classe por tópico  (ARI {tri.ari:.2f} · NMI {tri.nmi:.2f})",
                 loc="left", fontsize=10, color=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path
