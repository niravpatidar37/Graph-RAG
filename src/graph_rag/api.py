from __future__ import annotations

from functools import lru_cache

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .pipeline import CloudGraphRAG

app = FastAPI(title="Graph RAG API", version="0.1.0")


@app.get("/", response_class=HTMLResponse)
def home() -> str:
        return """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Graph RAG</title>
    <style>
        :root { color-scheme: light; font-family: Georgia, serif; }
        body { margin: 0; background: #eef2f0; color: #17221f; }
        main { max-width: 900px; margin: 0 auto; padding: 8vh 24px; }
        h1 { font-size: clamp(2.5rem, 7vw, 5.5rem); line-height: .95; margin: 0 0 12px; }
        p { color: #52635e; font: 1rem/1.5 system-ui, sans-serif; }
        form { display: flex; gap: 10px; margin: 32px 0; }
        input { flex: 1; min-width: 0; padding: 15px 16px; border: 1px solid #aabbb4; border-radius: 6px; font-size: 1rem; }
        button { padding: 15px 22px; border: 0; border-radius: 6px; background: #17624f; color: white; font-weight: 700; cursor: pointer; }
        button:disabled { opacity: .55; cursor: wait; }
        section { margin-top: 24px; padding-top: 20px; border-top: 1px solid #c4d0cb; }
        pre { white-space: pre-wrap; font: 1rem/1.55 system-ui, sans-serif; }
        ul { padding-left: 20px; font-family: system-ui, sans-serif; }
        .muted { color: #687a73; }
    </style>
</head>
<body>
<main>
    <p class="muted">EVIDENCE-GROUNDED KNOWLEDGE SEARCH</p>
    <h1>Ask the graph.</h1>
    <p>Questions are answered from indexed documents and their relationships.</p>
    <form id="query-form">
        <input id="question" required placeholder="Ask a question about the indexed documents" autocomplete="off">
        <button id="submit" type="submit">Ask</button>
    </form>
    <section>
        <h2>Answer</h2>
        <pre id="answer" class="muted">Your answer will appear here.</pre>
    </section>
    <section><h2>Sources</h2><ul id="sources"><li class="muted">No sources yet.</li></ul></section>
    <section><h2>Graph facts</h2><ul id="facts"><li class="muted">No graph facts yet.</li></ul></section>
</main>
<script>
const form = document.getElementById('query-form');
const input = document.getElementById('question');
const button = document.getElementById('submit');
const answer = document.getElementById('answer');
const sources = document.getElementById('sources');
const facts = document.getElementById('facts');
function fill(list, values, formatter) {
    list.replaceChildren();
    (values.length ? values : ['None returned.']).forEach(value => {
        const item = document.createElement('li');
        item.textContent = formatter ? formatter(value) : value;
        list.appendChild(item);
    });
}
form.addEventListener('submit', async event => {
    event.preventDefault();
    button.disabled = true;
    answer.textContent = '';
    fill(sources, []);
    fill(facts, []);
    try {
        const response = await fetch('/query/stream', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question: input.value, limit: 8}) });
        if (!response.ok) throw new Error('Query failed');
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const events = buffer.split('\\n\\n');
            buffer = events.pop();
            for (const raw of events) {
                if (!raw.startsWith('data: ')) continue;
                const data = JSON.parse(raw.slice(6));
                if (data.type === 'evidence') {
                    fill(sources, data.sources || [], source => source.document_id + ' (score ' + Number(source.score).toFixed(3) + ')\\n' + source.text);
                    fill(facts, data.graph_facts || []);
                } else if (data.type === 'token') {
                    answer.textContent += data.text;
                } else if (data.type === 'error') {
                    throw new Error(data.detail || 'Query failed');
                }
            }
        }
    } catch (error) {
        answer.textContent = error.message;
    } finally { button.disabled = false; }
});
</script>
</body>
</html>"""


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    limit: int = Field(default=8, ge=1, le=50)


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, object]]
    graph_facts: list[str]
    retrieved_entities: list[str]


@lru_cache(maxsize=1)
def get_pipeline() -> CloudGraphRAG:
    return CloudGraphRAG.from_env()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict[str, object]:
    return get_pipeline().metrics.snapshot()


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        return QueryResponse(**get_pipeline().query_result(request.question, request.limit))
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/query/stream")
def query_stream(request: QueryRequest) -> StreamingResponse:
    def events():
        try:
            for event in get_pipeline().query_stream(request.question, request.limit):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as error:  # a broken stream must still close with an error event, not a silent truncation
            yield f"data: {json.dumps({'type': 'error', 'detail': str(error)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


def main() -> None:
    import uvicorn

    uvicorn.run("graph_rag.api:app", host="0.0.0.0", port=8000, reload=False)