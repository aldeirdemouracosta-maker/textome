"""
Teste real do BERTopic no corpus de exemplo: os tópicos semânticos devem
recuperar os temas anotados nas variáveis *tema_ dos cabeçalhos.

  python exemplos/teste_bertopic.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpus import parse_corpus  # noqa: E402
from triangulacao import compare, run_bertopic  # noqa: E402

docs = parse_corpus((Path(__file__).with_name("corpus_demo.txt")).read_text(encoding="utf-8"))
temas = [re.search(r"\*tema_(\w+)", d.header).group(1) for d in docs]
ids = {t: i + 1 for i, t in enumerate(sorted(set(temas)))}
gold = [ids[t] for t in temas]

topics, words = run_bertopic([d.text for d in docs], n_topics=len(ids), seed=42)
tri = compare(gold, topics, topic_words=words, method="BERTopic (KMeans)")
print(tri.summary())
for cid in tri.class_ids:
    b = tri.best_topic[cid]
    tema = next(t for t, i in ids.items() if i == cid)
    print(f"  {tema:<12} → T{b['topico']} ({b['pct_classe']:.0f}%): {', '.join(words.get(b['topico'], [])[:5])}")
assert tri.ari > 0.2, f"ARI muito baixo: {tri.ari:.2f}"
