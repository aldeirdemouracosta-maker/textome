"""
Análises complementares à CHD, no estilo do IRaMuTeQ:
- estatísticas textuais (segmentos, ocorrências, formas, hápax)
- distribuição das classes
- AFC (análise fatorial de correspondência) classes × formas
- análise de similitude (coocorrência + árvore máxima)
- nuvem de palavras

Os cálculos usam a matriz segmentos × formas exportada do R (ou qualquer
matriz equivalente), então podem ser testados sem R.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

# Paleta categórica validada (modo claro): cada classe tem sempre a mesma cor
# em todos os gráficos. Classes além da 8ª ficam em cinza neutro (nunca uma
# 9ª cor gerada) e se distinguem pelo marcador e pelo rótulo.
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER_COLOR = "#8a8984"
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h", "<", ">", "p", "*"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def class_color(cid: int) -> str:
    return CLASS_COLORS[cid - 1] if 1 <= cid <= len(CLASS_COLORS) else OTHER_COLOR


def class_marker(cid: int) -> str:
    return MARKERS[(cid - 1) % len(MARKERS)]


@dataclass
class CorpusMatrix:
    """Matriz segmentos × formas (contagens) + classe de cada segmento (0 = não classificado)."""
    counts: sparse.csr_matrix
    terms: List[str]
    groups: List[int]

    def __post_init__(self):
        self.counts = sparse.csr_matrix(self.counts)
        if self.counts.shape != (len(self.groups), len(self.terms)):
            raise ValueError(
                f"Matriz {self.counts.shape} incompatível com {len(self.groups)} segmentos "
                f"e {len(self.terms)} formas"
            )


@dataclass
class TextStats:
    segments: int
    classified: int
    occurrences: int
    forms: int
    hapax: int

    @property
    def hapax_pct_forms(self) -> float:
        return 100 * self.hapax / self.forms if self.forms else 0.0

    @property
    def classified_pct(self) -> float:
        return 100 * self.classified / self.segments if self.segments else 0.0


def text_stats(m: CorpusMatrix) -> TextStats:
    freq = np.asarray(m.counts.sum(axis=0)).ravel()
    return TextStats(
        segments=m.counts.shape[0],
        classified=sum(1 for g in m.groups if g and g > 0),
        occurrences=int(freq.sum()),
        forms=int((freq > 0).sum()),
        hapax=int((freq == 1).sum()),
    )


# ---------------------------------------------------------------------------
# AFC
# ---------------------------------------------------------------------------

@dataclass
class CAResult:
    class_ids: List[int]
    terms: List[str]
    row_coords: np.ndarray     # classes × fatores
    col_coords: np.ndarray     # formas × fatores
    inertia_pct: np.ndarray    # % de inércia por fator
    term_class: List[int]      # classe em que a forma é mais característica


def class_term_table(m: CorpusMatrix) -> Tuple[List[int], np.ndarray]:
    class_ids = sorted({g for g in m.groups if g and g > 0})
    groups = np.asarray([g or 0 for g in m.groups])
    table = np.vstack([
        np.asarray(m.counts[groups == cid].sum(axis=0)).ravel() for cid in class_ids
    ]) if class_ids else np.zeros((0, len(m.terms)))
    return class_ids, table


def correspondence_analysis(table: np.ndarray, n_factors: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """AFC clássica por SVD. Retorna (coord. linhas, coord. colunas, % inércia)."""
    N = np.asarray(table, dtype=float)
    total = N.sum()
    if total == 0:
        raise ValueError("Tabela vazia")
    P = N / total
    r = P.sum(axis=1)
    c = P.sum(axis=0)
    keep_r, keep_c = r > 0, c > 0
    P, r, c = P[keep_r][:, keep_c], r[keep_r], c[keep_c]
    S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    U, sv, Vt = np.linalg.svd(S, full_matrices=False)
    eig = sv ** 2
    k = min(n_factors, int((eig > 1e-12).sum()))
    if k == 0:
        raise ValueError("A AFC precisa de pelo menos 2 classes com vocabulário distinto")
    rows = np.zeros((len(keep_r), k))
    cols = np.zeros((len(keep_c), k))
    rows[keep_r] = (U[:, :k] * sv[:k]) / np.sqrt(r)[:, None]
    cols[keep_c] = (Vt[:k].T * sv[:k]) / np.sqrt(c)[:, None]
    pct = 100 * eig[:k] / eig.sum()
    return rows, cols, pct


def run_ca(m: CorpusMatrix, terms_per_class: int = 12,
           class_forms: Optional[Dict[int, List[Tuple[str, float]]]] = None) -> Optional[CAResult]:
    class_ids, table = class_term_table(m)
    if len(class_ids) < 3:
        return None  # com 2 classes há só 1 fator: o plano F1×F2 não existe
    rows, cols, pct = correspondence_analysis(table, 2)
    if rows.shape[1] < 2:
        return None

    # Formas mostradas: as mais características de cada classe (χ² do rainette)
    # ou, sem elas, as mais frequentes de cada classe.
    index = {t: i for i, t in enumerate(m.terms)}
    chosen: Dict[str, int] = {}
    for pos, cid in enumerate(class_ids):
        if class_forms and class_forms.get(cid):
            candidates = [f for f, _ in class_forms[cid] if f in index]
        else:
            order = np.argsort(-table[pos])
            candidates = [m.terms[j] for j in order if table[pos, j] > 0]
        for term in candidates[:terms_per_class]:
            chosen.setdefault(term, cid)
    terms = list(chosen)
    idx = [index[t] for t in terms]
    return CAResult(class_ids, terms, rows, cols[idx], pct, [chosen[t] for t in terms])


# ---------------------------------------------------------------------------
# Similitude
# ---------------------------------------------------------------------------

def cooccurrence_graph(m: CorpusMatrix, top_n: int = 50, min_cooc: int = 2):
    """Grafo de coocorrência (nº de segmentos com as duas formas) e sua árvore máxima."""
    import networkx as nx

    presence = (m.counts > 0).astype(np.int32)
    freq = np.asarray(presence.sum(axis=0)).ravel()
    top = [j for j in np.argsort(-freq)[:top_n] if freq[j] > 0]
    sub = presence[:, top]
    cooc = (sub.T @ sub).toarray()
    g = nx.Graph()
    for a, j in enumerate(top):
        g.add_node(m.terms[j], freq=int(freq[j]))
    for a in range(len(top)):
        for b in range(a + 1, len(top)):
            if cooc[a, b] >= min_cooc:
                g.add_edge(m.terms[top[a]], m.terms[top[b]], weight=int(cooc[a, b]))
    tree = nx.maximum_spanning_tree(g, weight="weight") if g.number_of_edges() else g.copy()
    tree.remove_nodes_from([n for n in list(tree) if tree.degree(n) == 0])
    return g, tree


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def _setup_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "text.color": TEXT_PRIMARY,
                         "axes.labelcolor": TEXT_SECONDARY, "figure.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE})
    return plt


def plot_class_sizes(sizes: Dict[int, int], names: Dict[int, str], path: Path) -> Path:
    plt = _mpl()
    ids = sorted(sizes)
    total = sum(sizes.values()) or 1
    pct = [100 * sizes[c] / total for c in ids]
    labels = [f"Classe {c} — {names.get(c, '')}".rstrip(" —")[:55] for c in ids]
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(ids) + 1.0))
    _setup_axes(ax)
    y = np.arange(len(ids))[::-1]
    ax.barh(y, pct, height=0.62, color=[class_color(c) for c in ids],
            edgecolor=SURFACE, linewidth=2)
    for yi, p, c in zip(y, pct, ids):
        ax.text(p + 0.8, yi, f"{p:.1f}% ({sizes[c]})", va="center", fontsize=8, color=TEXT_PRIMARY)
    ax.set_yticks(y, labels, fontsize=8, color=TEXT_PRIMARY)
    ax.set_xlim(0, max(pct) * 1.25 + 1)
    ax.set_xlabel("% dos segmentos classificados")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Distribuição dos segmentos por classe", loc="left", fontsize=10, color=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_afc(ca: CAResult, names: Dict[int, str], path: Path) -> Path:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    _setup_axes(ax)
    ax.axhline(0, color=GRID, linewidth=0.8, zorder=0)
    ax.axvline(0, color=GRID, linewidth=0.8, zorder=0)
    xy = ca.col_coords
    for cid in ca.class_ids:
        sel = [i for i, c in enumerate(ca.term_class) if c == cid]
        if not sel:
            continue
        ax.scatter(xy[sel, 0], xy[sel, 1], s=28, marker=class_marker(cid), color=class_color(cid),
                   edgecolor=SURFACE, linewidth=1, zorder=3,
                   label=f"Classe {cid} — {names.get(cid, '')}".rstrip(" —")[:45])
    pad_x = 0.12 * (np.ptp(xy[:, 0]) or 1)
    pad_y = 0.08 * (np.ptp(xy[:, 1]) or 1)
    ax.set_xlim(xy[:, 0].min() - pad_x, xy[:, 0].max() + 2.5 * pad_x)
    ax.set_ylim(xy[:, 1].min() - pad_y, xy[:, 1].max() + pad_y)
    omitted = place_labels(ax, xy[:, 0], xy[:, 1], ca.terms)
    if omitted:
        ax.text(0.0, -0.1, f"{omitted} rótulo(s) omitido(s) por sobreposição; lista completa no classes.json.",
                transform=ax.transAxes, fontsize=7, color=TEXT_SECONDARY)
    ax.set_xlabel(f"Fator 1 ({ca.inertia_pct[0]:.1f}% da inércia)")
    ax.set_ylabel(f"Fator 2 ({ca.inertia_pct[1]:.1f}% da inércia)")
    ax.set_title("AFC — formas características por classe", loc="left", fontsize=10, color=TEXT_PRIMARY)
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_similitude(tree, term_class: Dict[str, int], names: Dict[int, str], path: Path,
                    seed: int = 42) -> Path:
    import networkx as nx

    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.set_axis_off()
    if tree.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "Coocorrências insuficientes para o grafo de similitude",
                ha="center", va="center", color=TEXT_SECONDARY)
    else:
        pos = layout_components(tree, seed=seed)
        ax.margins(0.08)
        weights = np.array([d["weight"] for _, _, d in tree.edges(data=True)] or [1])
        widths = 0.6 + 3.4 * (weights - weights.min()) / (np.ptp(weights) or 1)
        nx.draw_networkx_edges(tree, pos, ax=ax, width=widths, edge_color="#b9b8b2")
        freqs = np.array([tree.nodes[n]["freq"] for n in tree])
        sizes = 40 + 460 * (freqs - freqs.min()) / (np.ptp(freqs) or 1)
        colors = [class_color(term_class[n]) if n in term_class else OTHER_COLOR for n in tree]
        nx.draw_networkx_nodes(tree, pos, ax=ax, node_size=sizes, node_color=colors,
                               edgecolors=SURFACE, linewidths=1.5)
        nodes = list(pos)
        omitted = place_labels(ax, [pos[n][0] for n in nodes], [pos[n][1] for n in nodes], nodes,
                               fontsize=7.5, above=True)
        if omitted:
            ax.text(0.0, -0.02, f"{omitted} rótulo(s) omitido(s) por sobreposição.",
                    transform=ax.transAxes, fontsize=7, color=TEXT_SECONDARY)
        present = sorted({term_class[n] for n in tree if n in term_class})
        handles = [plt.Line2D([], [], marker="o", linestyle="", color=class_color(c),
                              label=f"Classe {c} — {names.get(c, '')}".rstrip(" —")[:45]) for c in present]
        if any(n not in term_class for n in tree):
            handles.append(plt.Line2D([], [], marker="o", linestyle="", color=OTHER_COLOR,
                                      label="sem classe predominante"))
        ax.legend(handles=handles, fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Análise de similitude (árvore máxima de coocorrência)", loc="left",
                 fontsize=10, color=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_wordcloud(m: CorpusMatrix, path: Path, max_words: int = 120, seed: int = 42) -> Path:
    from wordcloud import WordCloud

    freq = np.asarray(m.counts.sum(axis=0)).ravel()
    words = {m.terms[j]: float(freq[j]) for j in np.argsort(-freq)[:max_words] if freq[j] > 0}
    # Um só tom (sequencial azul): tamanho e intensidade codificam a frequência.
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
    top = max(words.values()) if words else 1

    def color_func(word, **_):
        return ramp[min(len(ramp) - 1, int(len(ramp) * words.get(word, 0) / (top + 1e-9)))]

    wc = WordCloud(width=1600, height=900, background_color=SURFACE, random_state=seed,
                   prefer_horizontal=0.95, color_func=color_func, collocations=False)
    wc.generate_from_frequencies(words or {"(vazio)": 1})
    wc.to_file(str(path))
    return path


def place_labels(ax, xs: Sequence[float], ys: Sequence[float], labels: Sequence[str],
                 fontsize: float = 7.0, above: bool = False) -> int:
    """
    Posiciona rótulos sem sobreposição (busca gulosa entre posições candidatas
    ao redor de cada ponto, incluindo empilhar para pontos coincidentes).
    Retorna quantos rótulos foram omitidos por falta de espaço.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = []
    omitted = 0
    area = ax.get_window_extent(renderer)
    if above:
        offsets = [(0, 5), (0, -12)] + [(dx, dy) for dy in (5, -12, 14, -21) for dx in (-25, 25)]
    else:
        offsets = [(4, 3), (4, -9), (-4, 3), (-4, -9)] + [(4, 3 - 9 * n) for n in range(2, 7)] \
            + [(4, 3 + 9 * n) for n in range(1, 5)] + [(-4, 3 - 9 * n) for n in range(2, 5)]
    for x, y, label in zip(xs, ys, labels):
        for dx, dy in offsets:
            ha = "center" if above else ("left" if dx >= 0 else "right")
            t = ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                            fontsize=fontsize, color=TEXT_PRIMARY, ha=ha, va="bottom", zorder=5)
            bb = t.get_window_extent(renderer).expanded(1.04, 1.1)
            inside = (bb.x0 >= area.x0 and bb.x1 <= area.x1 and bb.y0 >= area.y0 and bb.y1 <= area.y1)
            if inside and not any(bb.overlaps(o) for o in placed):
                placed.append(bb)
                break
            t.remove()
        else:
            omitted += 1
    return omitted


def term_classes(m: CorpusMatrix,
                 class_forms: Optional[Dict[int, List[Tuple[str, float]]]] = None) -> Dict[str, int]:
    """
    Classe em que cada forma é mais característica: a do maior χ² do rainette
    quando disponível; senão, a de maior razão observado/esperado na matriz.
    """
    result: Dict[str, int] = {}
    class_ids, table = class_term_table(m)
    if class_ids:
        totals = table.sum(axis=1, keepdims=True)
        expected = totals * table.sum(axis=0, keepdims=True) / max(table.sum(), 1)
        ratio = np.divide(table, expected, out=np.zeros_like(table, dtype=float), where=expected > 0)
        best = ratio.argmax(axis=0)
        for j, term in enumerate(m.terms):
            if table[:, j].sum() > 0:
                result[term] = class_ids[best[j]]
    if class_forms:
        best_chi: Dict[str, float] = {}
        for cid, forms in class_forms.items():
            for f, chi2 in forms:
                if chi2 > best_chi.get(f, float("-inf")):
                    best_chi[f], result[f] = chi2, cid
    return result


def layout_components(graph, seed: int = 42) -> Dict[str, Tuple[float, float]]:
    """Desenha cada componente separadamente e os organiza em linhas (maior primeiro)."""
    import networkx as nx

    comps = sorted((graph.subgraph(c).copy() for c in nx.connected_components(graph)),
                   key=lambda g: -g.number_of_nodes())
    pos: Dict[str, Tuple[float, float]] = {}
    cursor_x, cursor_y, row_h, row_w = 0.0, 0.0, 0.0, 0.0
    max_w = 2.2 * np.sqrt(max(comps[0].number_of_nodes(), 1)) if comps else 1
    for comp in comps:
        n = comp.number_of_nodes()
        size = np.sqrt(n)
        if n == 1:
            local = {next(iter(comp)): np.zeros(2)}
        elif n == 2:
            a, b = comp
            local = {a: np.array([-0.5, 0.0]), b: np.array([0.5, 0.0])}
        else:
            local = nx.kamada_kawai_layout(comp, weight=None)
        arr = np.array(list(local.values()))
        arr = arr - arr.mean(axis=0)
        span = np.ptp(arr, axis=0).max() or 1
        arr = arr / span * size
        w = np.ptp(arr[:, 0]) + 1.0
        h = np.ptp(arr[:, 1]) + 1.0
        if cursor_x + w > max_w and cursor_x > 0:
            cursor_x, cursor_y, row_h = 0.0, cursor_y - row_h, 0.0
        offset = np.array([cursor_x - arr[:, 0].min(), cursor_y - arr[:, 1].max()])
        for node, xy in zip(local, arr):
            pos[node] = tuple(xy + offset)
        cursor_x += w
        row_h = max(row_h, h)
    return pos


@dataclass
class AnalysisOutput:
    stats: TextStats
    figures: Dict[str, Path] = field(default_factory=dict)
    ca: Optional[CAResult] = None
    warnings: List[str] = field(default_factory=list)


FIGURE_TITLES = {
    "dendrograma": "Dendrograma da CHD com as formas mais características de cada classe",
    "classes": "Distribuição dos segmentos de texto por classe",
    "afc": "Análise fatorial de correspondência (AFC)",
    "similitude": "Análise de similitude",
    "nuvem": "Nuvem de palavras do corpus",
}


def run_all(m: CorpusMatrix, out_dir: Path, sizes: Dict[int, int], names: Dict[int, str],
            class_forms: Optional[Dict[int, List[Tuple[str, float]]]] = None,
            dendrogram_png: Optional[bytes] = None, seed: int = 42) -> AnalysisOutput:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = AnalysisOutput(stats=text_stats(m))

    if dendrogram_png:
        p = out_dir / "dendrograma.png"
        p.write_bytes(dendrogram_png)
        result.figures["dendrograma"] = p
    if sizes:
        result.figures["classes"] = plot_class_sizes(sizes, names, out_dir / "classes.png")

    try:
        result.ca = run_ca(m, class_forms=class_forms)
        if result.ca:
            result.figures["afc"] = plot_afc(result.ca, names, out_dir / "afc.png")
        else:
            result.warnings.append("AFC não gerada: são necessárias pelo menos 3 classes.")
    except ValueError as e:
        result.warnings.append(f"AFC não gerada: {e}")

    term_class = term_classes(m, class_forms)
    _, tree = cooccurrence_graph(m)
    result.figures["similitude"] = plot_similitude(tree, term_class, names, out_dir / "similitude.png", seed)
    result.figures["nuvem"] = plot_wordcloud(m, out_dir / "nuvem.png", seed=seed)
    return result
