"""
Validação humana das classes e concordância (kappa de Cohen).

1. Revisão dos nomes: para cada classe o pesquisador aceita, edita ou rejeita
   o nome sugerido pela IA (taxa de aceitação).
2. Atribuição às cegas: segmentos sorteados (fora dos que foram mostrados à IA)
   são atribuídos pelo pesquisador — e opcionalmente pela IA — à classe cujo
   nome/descrição melhor os descreve, sem ver a classe da CHD. A concordância
   com a CHD indica se as classes e seus nomes são interpretáveis.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

NONE_LABEL = 0  # "nenhuma classe / não sei"

KAPPA_SCALE = [  # Landis & Koch (1977); κ < 0 é "pobre"
    (0.20, "leve"),
    (0.40, "razoável"),
    (0.60, "moderada"),
    (0.80, "substancial"),
    (1.01, "quase perfeita"),
]


def interpret_kappa(kappa: Optional[float]) -> str:
    if kappa is None:
        return "indefinido"
    if kappa < 0:
        return "pobre"
    for limit, label in KAPPA_SCALE:
        if kappa <= limit:
            return label
    return "quase perfeita"


@dataclass
class Agreement:
    n: int
    observed: float              # concordância observada (proporção)
    kappa: Optional[float]
    interpretation: str
    labels: List[int]
    matrix: List[List[int]]      # linhas = avaliador A, colunas = avaliador B


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> Agreement:
    if len(a) != len(b):
        raise ValueError("As duas listas de avaliações precisam ter o mesmo tamanho")
    n = len(a)
    labels = sorted(set(a) | set(b))
    if n == 0:
        return Agreement(0, 0.0, None, "indefinido", labels, [])
    index = {lab: i for i, lab in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for x, y in zip(a, b):
        matrix[index[x]][index[y]] += 1
    po = sum(matrix[i][i] for i in range(len(labels))) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[lab] * cb[lab] for lab in labels) / (n * n)
    kappa = None if pe == 1 else (po - pe) / (1 - pe)
    return Agreement(n, po, kappa, interpret_kappa(kappa), labels, matrix)


@dataclass
class NameDecision:
    class_id: int
    ai_name: str
    final_name: str
    decision: str  # "aceito", "editado" ou "rejeitado"


@dataclass
class Item:
    item_id: int
    text: str
    chd_class: int
    human: Optional[int] = None
    ai: Optional[int] = None


@dataclass
class ValidationSession:
    names: List[NameDecision] = field(default_factory=list)
    items: List[Item] = field(default_factory=list)
    descriptions: Dict[int, str] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    ai_model: Optional[str] = None

    # ----- nomes -----
    def final_names(self) -> Dict[int, str]:
        return {d.class_id: d.final_name for d in self.names}

    def acceptance(self) -> Dict[str, int]:
        c = Counter(d.decision for d in self.names)
        return {k: c.get(k, 0) for k in ("aceito", "editado", "rejeitado")}

    # ----- concordância -----
    def _pairs(self, attr_a: str, attr_b: str) -> Tuple[List[int], List[int]]:
        a, b = [], []
        for it in self.items:
            va = it.chd_class if attr_a == "chd" else getattr(it, attr_a)
            vb = it.chd_class if attr_b == "chd" else getattr(it, attr_b)
            if va is not None and vb is not None:
                a.append(va)
                b.append(vb)
        return a, b

    def agreements(self) -> Dict[str, Agreement]:
        out = {}
        for key, (x, y) in {
            "pesquisador × CHD": ("human", "chd"),
            "IA × CHD": ("ai", "chd"),
            "pesquisador × IA": ("human", "ai"),
        }.items():
            a, b = self._pairs(x, y)
            if a:
                out[key] = cohen_kappa(a, b)
        return out

    # ----- persistência -----
    def to_json(self) -> str:
        data = asdict(self)
        data["resultados"] = {
            "aceitacao_nomes": self.acceptance(),
            "concordancia": {k: asdict(v) for k, v in self.agreements().items()},
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "ValidationSession":
        data = json.loads(raw)
        return cls(
            names=[NameDecision(**d) for d in data.get("names", [])],
            items=[Item(**i) for i in data.get("items", [])],
            descriptions={int(k): v for k, v in data.get("descriptions", {}).items()},
            created_at=data.get("created_at", ""),
            ai_model=data.get("ai_model"),
        )


def decide_name(class_id: int, ai_name: str, final_name: Optional[str]) -> NameDecision:
    final = (final_name or "").strip()
    if final == "" or final == ai_name.strip():
        return NameDecision(class_id, ai_name, ai_name, "aceito")
    if final == "-":
        return NameDecision(class_id, ai_name, f"Classe {class_id}", "rejeitado")
    return NameDecision(class_id, ai_name, final, "editado")


def sample_items(
    segments: Sequence[Tuple[str, int]],
    exclude: Sequence[str] = (),
    per_class: int = 5,
    seed: int = 42,
) -> List[Item]:
    """Sorteio estratificado por classe, excluindo os segmentos já mostrados à IA."""
    rng = random.Random(seed)
    excluded = set(exclude)
    by_class: Dict[int, List[str]] = {}
    for text, cid in segments:
        if cid and cid > 0 and text not in excluded:
            by_class.setdefault(cid, [])
            if text not in by_class[cid]:
                by_class[cid].append(text)
    chosen: List[Tuple[str, int]] = []
    for cid in sorted(by_class):
        pool = by_class[cid]
        chosen += [(t, cid) for t in rng.sample(pool, min(per_class, len(pool)))]
    rng.shuffle(chosen)
    return [Item(i + 1, t, cid) for i, (t, cid) in enumerate(chosen)]


def assign_with_ai(session: ValidationSession, assign: Callable[[str, Dict[int, str]], Optional[int]]) -> None:
    """assign(texto, {classe: 'nome — descrição'}) -> classe escolhida (ou 0)."""
    options = {
        cid: f"{name} — {session.descriptions.get(cid, '')}".rstrip(" —")
        for cid, name in session.final_names().items()
    }
    for it in session.items:
        try:
            it.ai = assign(it.text, options)
        except Exception:
            it.ai = NONE_LABEL


# ---------------------------------------------------------------------------
# Relatórios
# ---------------------------------------------------------------------------

def _label(cid: int, names: Dict[int, str]) -> str:
    return "nenhuma/não sei" if cid == NONE_LABEL else f"{cid}. {names.get(cid, f'Classe {cid}')}"[:40]


def method_text(session: ValidationSession) -> str:
    acc = session.acceptance()
    total = sum(acc.values()) or 1
    n_items = len(session.items)
    n_classes = len(session.names)
    parts = [
        f"Os nomes sugeridos pela IA para as {n_classes} classes foram revisados pelo pesquisador: "
        f"{acc['aceito']} aceitos ({100 * acc['aceito'] / total:.0f}%), {acc['editado']} editados e "
        f"{acc['rejeitado']} rejeitados."
    ]
    if n_items:
        parts.append(
            f"Para avaliar a interpretabilidade das classes, {n_items} segmentos de texto sorteados "
            "(estratificados por classe e distintos dos apresentados à IA) foram atribuídos às cegas "
            "à classe cujo nome e descrição melhor os representava; a concordância com a "
            "classificação da CHD foi medida pelo kappa de Cohen (interpretação de Landis e Koch, 1977)."
        )
    for key, ag in session.agreements().items():
        k = "indefinido" if ag.kappa is None else f"κ = {ag.kappa:.2f}"
        parts.append(f"Concordância {key}: {k} ({ag.interpretation}); "
                     f"concordância observada de {100 * ag.observed:.0f}% (n = {ag.n}).")
    return " ".join(parts)


def build_markdown(session: ValidationSession) -> str:
    names = session.final_names()
    lines = ["# Validação das classes", "", method_text(session), "", "## Nomes das classes", "",
             "| Classe | Sugestão da IA | Nome final | Decisão |", "|---|---|---|---|"]
    lines += [f"| {d.class_id} | {d.ai_name} | {d.final_name} | {d.decision} |" for d in session.names]
    for key, ag in session.agreements().items():
        lines += ["", f"## Matriz de confusão — {key}", ""]
        header = [_label(c, names) for c in ag.labels]
        lines.append("| | " + " | ".join(header) + " |")
        lines.append("|---" * (len(header) + 1) + "|")
        for lab, row in zip(header, ag.matrix):
            lines.append(f"| **{lab}** | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def build_docx(session: ValidationSession, path: str | Path) -> Path:
    from docx import Document as Docx

    names = session.final_names()
    doc = Docx()
    doc.add_heading("Validação das classes", level=0)
    doc.add_paragraph(method_text(session))
    doc.add_heading("Nomes das classes", level=1)
    t = doc.add_table(rows=1, cols=4)
    t.style = "Light Grid Accent 1"
    for cell, h in zip(t.rows[0].cells, ["Classe", "Sugestão da IA", "Nome final", "Decisão"]):
        cell.text = h
    for d in session.names:
        cells = t.add_row().cells
        for cell, v in zip(cells, [d.class_id, d.ai_name, d.final_name, d.decision]):
            cell.text = str(v)
    for key, ag in session.agreements().items():
        doc.add_heading(f"Matriz de confusão — {key}", level=1)
        k = "indefinido" if ag.kappa is None else f"{ag.kappa:.2f}"
        doc.add_paragraph(f"κ = {k} ({ag.interpretation}); concordância observada "
                          f"{100 * ag.observed:.0f}% (n = {ag.n}). Linhas: primeiro avaliador; "
                          "colunas: segundo.")
        header = [_label(c, names) for c in ag.labels]
        m = doc.add_table(rows=1, cols=len(header) + 1)
        m.style = "Light Grid Accent 1"
        for cell, h in zip(m.rows[0].cells[1:], header):
            cell.text = h
        for lab, row in zip(header, ag.matrix):
            cells = m.add_row().cells
            cells[0].text = lab
            for cell, v in zip(cells[1:], row):
                cell.text = str(v)
    path = Path(path)
    doc.save(path)
    return path


def write_validation(session: ValidationSession, out_dir: str | Path) -> List[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / "validacao.json", out / "validacao.md"]
    paths[0].write_text(session.to_json(), encoding="utf-8")
    paths[1].write_text(build_markdown(session), encoding="utf-8")
    try:
        paths.append(build_docx(session, out / "validacao.docx"))
    except ImportError:
        pass
    return paths


def session_from_outputs(result_dir: str | Path, per_class: int = 5, seed: int = 42) -> ValidationSession:
    """Monta a sessão a partir de classes.json gerado pelo pipeline."""
    data = json.loads((Path(result_dir) / "classes.json").read_text(encoding="utf-8"))
    session = ValidationSession()
    shown: List[str] = []
    for c in data["classes"]:
        interp = c.get("interpretacao") or {}
        session.names.append(NameDecision(c["id"], c["nome"], c["nome"], "aceito"))
        session.descriptions[c["id"]] = interp.get("descricao", "")
        shown += c.get("segmentos", [])
    segments = [(s["texto"], s["classe"]) for s in data.get("todos_segmentos", [])]
    if not segments:  # resultados antigos: só os segmentos de exemplo
        segments = [(t, c["id"]) for c in data["classes"] for t in c.get("segmentos", [])]
        shown = []
    session.items = sample_items(segments, exclude=shown, per_class=per_class, seed=seed)
    return session
