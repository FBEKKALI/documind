"""Banc d'évaluation du RAG : hit-rate@k, MRR et couverture de mots-clés.

Format de eval_questions.json :
[
  {"question": "...", "expected_source": "fichier.pdf", "expected_keywords": ["mot1", "mot2"]}
]
"""
import argparse
import json
from pathlib import Path

import rag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="eval_questions.json")
    parser.add_argument("--k", type=int, default=rag.TOP_K)
    parser.add_argument("--answers", action="store_true",
                        help="génère aussi les réponses et mesure la couverture de mots-clés")
    args = parser.parse_args()

    cases = json.loads(Path(args.file).read_text(encoding="utf-8"))
    hits, rr_sum, kw_scores = 0, 0.0, []

    for case in cases:
        passages = rag.retrieve(case["question"], args.k)
        sources = [p["source"] for p in passages]
        expected = case.get("expected_source")
        rank = sources.index(expected) + 1 if expected in sources else None
        if rank:
            hits += 1
            rr_sum += 1 / rank

        line = f"- {case['question'][:70]:70} | source rank: {rank or '-'}"
        keywords = case.get("expected_keywords", [])
        if args.answers and keywords:
            text = rag.answer(case["question"], args.k)["answer"].lower()
            cov = sum(kw.lower() in text for kw in keywords) / len(keywords)
            kw_scores.append(cov)
            line += f" | keywords: {cov:.0%}"
        print(line)

    n = len(cases)
    print(f"\nQuestions: {n} | k={args.k}")
    print(f"hit@{args.k}: {hits / n:.2%}")
    print(f"MRR: {rr_sum / n:.3f}")
    if kw_scores:
        print(f"keyword coverage: {sum(kw_scores) / len(kw_scores):.2%}")


if __name__ == "__main__":
    main()
