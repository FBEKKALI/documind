# DocuMind – Assistant RAG avec citations et évaluation

Assistant de questions-réponses sur vos propres documents (PDF, Markdown, TXT).
Chaque réponse est **ancrée dans les documents** et **cite ses sources** (fichier + page).
Le projet inclut un **banc d'évaluation** qui mesure la qualité de la récupération (retrieval), pour itérer sur le chunking et le paramètre `k` avec des chiffres plutôt qu'au feeling.

## Fonctionnalités

- Ingestion de PDF / MD / TXT, découpage en chunks avec recouvrement
- Index vectoriel local FAISS (aucune base externe à installer)
- Génération avec LangChain, fournisseur au choix : **Azure OpenAI** ou **OpenAI**
- Réponses citées `[1]`, `[2]` avec les passages utilisés
- Refus explicite de répondre quand le contexte ne contient pas l'information
- API REST Flask + petite interface web
- Évaluation : hit-rate@k, MRR et couverture de mots-clés attendus
- Dockerfile prêt à déployer (ex. Azure Container Apps / App Service)

## Architecture

```
documents ──> ingest.py ──> chunks ──> embeddings ──> index FAISS
                                                          │
question ──> app.py (Flask) ──> rag.py: retrieve top-k ───┘
                                   │
                                   └─> prompt + contexte ──> LLM ──> réponse + sources
```

## Démarrage rapide

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # puis renseigner vos clés

mkdir -p docs               # déposer vos PDF / MD / TXT dedans
python ingest.py            # construit l'index dans ./index
python app.py               # http://localhost:5000
```

Appel API :

```bash
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quelle est la politique de télétravail ?", "k": 4}'
```

## Évaluation

1. Copier `eval_questions.example.json` en `eval_questions.json` et l'adapter à vos documents :
   chaque entrée contient la question, la source attendue et des mots-clés attendus.
2. Lancer :

```bash
python evaluate.py                 # métriques de retrieval uniquement (rapide, peu coûteux)
python evaluate.py --answers       # ajoute la couverture de mots-clés dans les réponses générées
python evaluate.py --k 6
```

Sortie type : `hit@k`, `MRR`, `keyword coverage`. Comparez plusieurs configurations
(`CHUNK_SIZE`, `CHUNK_OVERLAP`, `k`) et gardez la meilleure.

## Configuration (`.env`)

| Variable | Rôle |
|---|---|
| `PROVIDER` | `azure` ou `openai` |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | paramètres de découpage |
| `TOP_K` | nombre de passages récupérés |
| `AZURE_OPENAI_*` | endpoint, clé, déploiements chat et embeddings |
| `OPENAI_API_KEY`, `OPENAI_CHAT_MODEL`, `OPENAI_EMBEDDING_MODEL` | mode OpenAI |

## Docker

```bash
docker build -t documind .
docker run --env-file .env -p 5000:5000 -v $(pwd)/index:/app/index documind
```

## Pistes d'amélioration

- Recherche hybride (BM25 + vectorielle) et reranking
- Évaluation de la fidélité des réponses avec un LLM juge
- Streaming des réponses et historique de conversation
- Authentification et ingestion par upload depuis l'interface
- Déploiement Azure avec Key Vault pour les secrets

## Stack

Python · LangChain · FAISS · Flask · Azure OpenAI / OpenAI · Docker
