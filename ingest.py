"""Construit l'index vectoriel à partir du dossier docs/."""
from rag import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR, build_index

if __name__ == "__main__":
    DOCS_DIR.mkdir(exist_ok=True)
    n_docs, n_chunks = build_index()
    print(f"{n_docs} page(s)/document(s) indexé(s) en {n_chunks} chunks "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
