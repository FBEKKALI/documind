"""API Flask + interface web minimale."""
from flask import Flask, jsonify, request

import rag

app = Flask(__name__)

PAGE = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><title>DocuMind</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#222}
 input{width:75%;padding:.6rem;font-size:1rem} button{padding:.6rem 1rem;font-size:1rem}
 .src{background:#f4f4f6;border-radius:6px;padding:.5rem .8rem;margin:.4rem 0;font-size:.9rem}
 #answer{white-space:pre-wrap;margin-top:1rem}
</style></head><body>
<h1>DocuMind</h1>
<p>Posez une question sur vos documents.</p>
<input id="q" placeholder="Votre question..." autofocus>
<button onclick="ask()">Envoyer</button>
<div id="answer"></div><div id="sources"></div>
<script>
async function ask(){
  const q=document.getElementById('q').value.trim(); if(!q) return;
  document.getElementById('answer').textContent='...'; document.getElementById('sources').innerHTML='';
  const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
  const d=await r.json();
  if(d.error){document.getElementById('answer').textContent=d.error;return;}
  document.getElementById('answer').textContent=d.answer;
  document.getElementById('sources').innerHTML=d.sources.map(s=>
    '<div class="src"><b>['+s.id+'] '+s.source+(s.page!==null?' p.'+(s.page+1):'')+'</b><br>'+
    s.excerpt.replace(/</g,'&lt;')+'...</div>').join('');
}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')ask()});
</script></body></html>"""


@app.get("/")
def home():
    return PAGE


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/ask")
def ask():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify(error="Le champ 'question' est requis."), 400
    k = int(payload.get("k", rag.TOP_K))
    try:
        return jsonify(rag.answer(question, k))
    except FileNotFoundError:
        return jsonify(error="Index introuvable. Lancez d'abord : python ingest.py"), 503


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
