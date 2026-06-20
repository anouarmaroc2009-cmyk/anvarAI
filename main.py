import json
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import config
from agent import CodingAgent

app = FastAPI(title="Anvar AI - Coding Agent", version="1.0.0")
agent = CodingAgent()

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def index():
        return FileResponse(str(static_dir / "index.html"))


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


class ChatGoogleRequest(BaseModel):
    message: str
    google_token: str


class ChatApiKeyRequest(BaseModel):
    message: str
    api_key: str


class ToolExecuteRequest(BaseModel):
    name: str
    args: dict


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/config")
async def get_config():
    return {
        "googleClientId": config.GOOGLE_CLIENT_ID,
        "hasServerKey": bool(config.GOOGLE_API_KEY),
        "hasBigPickle": bool(config.OPENCODE_API_KEY),
        "model": config.MODEL,
    }


@app.post("/chat/bp")
async def chat_bp(req: ChatRequest):
    chunks = []
    for chunk in agent.process_with_bigpickle(req.message):
        if chunk["type"] == "text":
            chunks.append(chunk["content"])
        elif chunk["type"] == "error":
            raise HTTPException(500, chunk["content"])
    return {"response": "".join(chunks)}


@app.post("/chat/bp/stream")
async def chat_bp_stream(req: ChatRequest):
    async def event_stream():
        for chunk in agent.process_with_bigpickle(req.message, stream=True):
            yield f"data: {json.dumps(chunk)}\n\n"
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post("/tools/execute")
async def tool_execute(req: ToolExecuteRequest):
    result = CodingAgent.execute_tool(req.name, req.args)
    return {"result": result}


@app.post("/chat/google")
async def chat_google(req: ChatGoogleRequest):
    chunks = []
    for chunk in agent.process_with_token(req.message, req.google_token):
        if chunk["type"] == "text":
            chunks.append(chunk["content"])
        elif chunk["type"] == "error":
            raise HTTPException(500, chunk["content"])
    return {"response": "".join(chunks)}


@app.post("/chat/google/stream")
async def chat_google_stream(req: ChatGoogleRequest):
    async def event_stream():
        for chunk in agent.process_with_token(req.message, req.google_token):
            yield f"data: {json.dumps(chunk)}\n\n"
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post("/chat/apikey")
async def chat_apikey(req: ChatApiKeyRequest):
    chunks = []
    for chunk in agent.process_with_api_key(req.message, req.api_key):
        if chunk["type"] == "text":
            chunks.append(chunk["content"])
        elif chunk["type"] == "error":
            raise HTTPException(500, chunk["content"])
    return {"response": "".join(chunks)}


@app.post("/chat/apikey/stream")
async def chat_apikey_stream(req: ChatApiKeyRequest):
    async def event_stream():
        for chunk in agent.process_with_api_key(req.message, req.api_key, stream=True):
            yield f"data: {json.dumps(chunk)}\n\n"
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if config.OPENCODE_API_KEY:
        chunks = []
        for chunk in agent.process_with_bigpickle(req.message):
            if chunk["type"] == "text":
                chunks.append(chunk["content"])
            elif chunk["type"] == "error":
                raise HTTPException(500, chunk["content"])
        return ChatResponse(response="".join(chunks))
    if not config.GOOGLE_API_KEY:
        raise HTTPException(
            500,
            "No API key configured — set OPENCODE_API_KEY (free at opencode.ai/zen) "
            "or GOOGLE_API_KEY (free at aistudio.google.com/apikey)",
        )
    try:
        result = agent.process_message(req.message)
        return ChatResponse(response=result)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if config.OPENCODE_API_KEY:
        async def bp_stream():
            for chunk in agent.process_with_bigpickle(req.message, stream=True):
                yield f"data: {json.dumps(chunk)}\n\n"
        return StreamingResponse(
            bp_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )
    if not config.GOOGLE_API_KEY:
        raise HTTPException(500, "No API key configured")

    async def event_stream():
        for chunk in agent.process_message(req.message, stream=True):
            yield f"data: {json.dumps(chunk)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/history")
async def history():
    return {"messages": agent.get_history()}


@app.post("/reset")
async def reset():
    agent.reset()
    return {"status": "conversation reset"}


import os
# Railway sets $PORT - use it if available
_railway_port = os.environ.get("PORT")
if _railway_port:
    config.PORT = int(_railway_port)

if __name__ == "__main__":
    import uvicorn
    print(f"AI Coding Agent starting - Model: {config.MODEL} on port {config.PORT}")
    uvicorn.run("main:app", host="0.0.0.0", port=config.PORT, reload=True)
