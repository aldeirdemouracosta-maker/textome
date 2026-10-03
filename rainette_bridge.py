"""
rainette_bridge.py
Ponte Python ↔ rainette (R) via rpy2.
Executa CHD (método Reinert) e extrai formas + segmentos por classe.
"""

from __future__ import annotations

from collections import Counter
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

        corp <- corpus(texts_py)
        corp <- split_segments(corp, segment_size = {int(segment_size)})

        tok <- tokens(corp, remove_punct = TRUE, remove_numbers = TRUE)
        tok <- tokens_remove(tok, stopwords("{lang}"))
        tok <- tokens_tolower(tok)

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
        self.groups = [int(g) for g in list(ro.r["groups"])]

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
        selected = []
        for idx in indices[:max_segments]:
            if 0 <= idx < len(segment_texts):
                text = str(segment_texts[idx]).strip()
                if text:
                    selected.append(text)
        return selected

    def get_group_sizes(self) -> Dict[int, int]:
        counts = Counter(self.groups)
        return {int(k): v for k, v in counts.items() if k is not None}
