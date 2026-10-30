import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pytest  # noqa: E402

from src.benchmark.corpus import Sentence, GoldTriple  # noqa: E402


@pytest.fixture
def toy_sentences() -> list[Sentence]:
    return [
        Sentence(
            sentence_id="s1",
            sentence="O Brasil exporta soja.",
            doc_id=1,
            phrase_index=0,
            gold=[GoldTriple("O Brasil", "exporta", "soja")],
        ),
        Sentence(
            sentence_id="s2",
            sentence="A capital do Brasil é Brasília.",
            doc_id=2,
            phrase_index=0,
            gold=[GoldTriple("A capital do Brasil", "é", "Brasília")],
        ),
        Sentence(
            sentence_id="s3",
            sentence="Maria comprou pão e queijo.",
            doc_id=3,
            phrase_index=0,
            gold=[
                GoldTriple("Maria", "comprou", "pão"),
                GoldTriple("Maria", "comprou", "queijo"),
            ],
        ),
    ]
