"""
rainette_bridge.py
Ponte Python ↔ rainette (R) via rpy2.
Executa CHD (método Reinert) e extrai formas + segmentos por classe.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.rinterface_lib.embedded import RRuntimeError
    RPY2_AVAILABLE = True
except ImportError:
    RPY2_AVAILABLE = False
    ro = None
    RRuntimeError = Exception


class RainetteBridge:
    """
    Executa Classificação Hierárquica Descendente com rainette
    e devolve formas características + segmentos por classe.
    """

    def __init__(self):
        if not RPY2_AVAILABLE:
            raise ImportError(
                "rpy2 não está instalado. Instale com: pip install rpy2\n"
                "Além disso, R e os pacotes rainette + quanteda precisam estar disponíveis."
            )
        self.res = None
        self.dtm = None
        self.corpus = None
        self.groups: List[int] = []
        self.k: Optional[int] = None
        self._check_r_packages()

    def _check_r_packages(self) -> None:
        """Verifica se quanteda e rainette estão instalados no R."""
        try:
            importr("quanteda")
            importr("rainette")
        except Exception as e:
            raise RuntimeError(
                "Pacotes R necessários não encontrados.\n"
                "No R, execute:\n"
                '  install.packages(c("quanteda", "rainette"))\n'
                f"Detalhe: {e}"
            ) from e

    def run(
        self,
        texts: List[str],
        k: int = 6,
        segment_size: int = 40,
        min_segment_size: int = 12,
        min_split_members: int = 10,
        language: str = "pt",
        min_docfreq: int = 5,
        n_terms: int = 20,
        seed: int = 42,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Pipeline completo de classificação.

        Retorna:
        {
          1: {"forms": [("palavra", 42.5), ...], "segments": ["texto...", ...]},
          ...
        }
        """
        if not texts:
            raise ValueError("Lista de textos vazia.")

        self.k = k
        lang = language if language in ("pt", "en", "fr", "es", "de", "it") else "en"

        with localconverter(ro.default_converter + pandas2ri.converter):
            ro.globalenv["texts_py"] = ro.StrVector(texts)

        r_code = f"""
        library(quanteda)
        library(rainette)
        set.seed({int(seed)})

        corp <- corpus(texts_py)
        corp <- split_segments(corp, segment_size = {int(segment_size)})

        tok <- tokens(corp, remove_punct = TRUE, remove_numbers = TRUE)
        tok <- tokens_tolower(tok)  # antes das stopwords: "É" no início da frase também sai
        tok <- tokens_remove(tok, stopwords("{lang}"))

        dtm <- dfm(tok)
        dtm <- dfm_trim(dtm, min_docfreq = {int(min_docfreq)})

        if (ndoc(dtm) < 4) {{
            stop("Poucos segmentos após pré-processamento. Ajuste segment_size ou o corpus.")
        }}

        res <- rainette(
            dtm,
            k = {int(k)},
            min_segment_size = {int(min_segment_size)},
            min_split_members = {int(min_split_members)}
        )

        groups <- cutree_rainette(res, k = {int(k)})
        stats <- rainette_stats(
            groups,
            dtm,
            measure = "chi2",
            n_terms = {int(n_terms)},
            show_negative = FALSE
        )
        segment_texts <- as.character(corp)
        """

        try:
            ro.r(r_code)
        except RRuntimeError as e:
            raise RuntimeError(f"Erro ao executar rainette no R:\n{e}") from e

        self.res = ro.r["res"]
        self.dtm = ro.r["dtm"]
        self.corpus = ro.r["corp"]
        # Segmentos não classificados vêm como NA do R → 0
        self.groups = [
            int(g) if g is not None and int(g) > 0 else 0  # NA_integer_ é negativo
            for g in list(ro.r["groups"])
        ]

        return self._extract_classes(n_terms=n_terms)

    def _extract_classes(self, n_terms: int = 20) -> Dict[int, Dict[str, Any]]:
        classes: Dict[int, Dict[str, Any]] = {}
        stats_list = ro.r["stats"]

        for i, class_id in enumerate(range(1, self.k + 1)):
            forms: List[Tuple[str, float]] = []
            try:
                df_r = stats_list.rx2(i + 1)
                with localconverter(ro.default_converter + pandas2ri.converter):
                    df = ro.conversion.rpy2py(df_r)
                if "feature" in df.columns and "chi2" in df.columns:
                    df = df.sort_values("chi2", ascending=False).head(n_terms)
                    forms = list(
                        zip(
                            [str(x) for x in df["feature"].tolist()],
                            [float(x) for x in df["chi2"].tolist()],
                        )
                    )
            except Exception:
                forms = []

            segments = self._get_segments_for_class(class_id)
            classes[class_id] = {"forms": forms, "segments": segments}

        return classes

    def _get_segments_for_class(
        self, class_id: int, max_segments: int = 8
    ) -> List[str]:
        indices = [i for i, g in enumerate(self.groups) if g == class_id]
        if not indices:
            return []
        segment_texts = list(ro.r["segment_texts"])
        selected: List[str] = []
        for idx in indices:
            if len(selected) >= max_segments:
                break
            if 0 <= idx < len(segment_texts):
                text = str(segment_texts[idx]).strip()
                if text and text not in selected:  # sem repetir segmentos idênticos
                    selected.append(text)
        return selected

    def get_segments(self) -> List[Tuple[str, int]]:
        """Todos os segmentos de texto com a classe da CHD (0 = não classificado)."""
        texts = [str(t).strip() for t in ro.r["segment_texts"]]
        return list(zip(texts, self.groups))

    def get_matrix(self):
        """Matriz segmentos × formas (a mesma usada na CHD) + classe de cada segmento."""
        from scipy import sparse

        from analises import CorpusMatrix

        ro.r("""
        dtm_t <- as(as(dtm, "CsparseMatrix"), "TsparseMatrix")
        dtm_i <- dtm_t@i
        dtm_j <- dtm_t@j
        dtm_x <- dtm_t@x
        dtm_feat <- featnames(dtm)
        dtm_dim <- dim(dtm)
        """)
        i = np.asarray(ro.r["dtm_i"], dtype=int)
        j = np.asarray(ro.r["dtm_j"], dtype=int)
        x = np.asarray(ro.r["dtm_x"], dtype=float)
        n_docs, n_terms = (int(v) for v in ro.r["dtm_dim"])
        counts = sparse.csr_matrix((x, (i, j)), shape=(n_docs, n_terms))
        terms = [str(t) for t in ro.r["dtm_feat"]]
        groups = [g if isinstance(g, int) and g > 0 else 0 for g in self.groups]
        return CorpusMatrix(counts, terms, groups)

    def save_dendrogram(self, path: str, n_terms: int = 15) -> Optional[str]:
        """Salva o dendrograma do rainette (PNG). Retorna None ou a mensagem de erro."""
        ro.globalenv["dendro_path"] = ro.StrVector([str(path)])
        try:
            ro.r(f"""
            p <- rainette_plot(res, dtm, k = {int(self.k)}, n_terms = {int(n_terms)},
                               measure = "chi2", show_negative = FALSE)
            png(dendro_path, width = 2400, height = 1500, res = 200)
            if (inherits(p, "ggplot")) print(p) else grid::grid.draw(p)
            invisible(dev.off())
            """)
            return None
        except RRuntimeError as e:
            try:
                ro.r("if (dev.cur() > 1) invisible(dev.off())")
            except Exception:
                pass
            return str(e)

    def get_group_sizes(self) -> Dict[int, int]:
        counts = Counter(self.groups)
        return {int(k): v for k, v in counts.items() if k}
