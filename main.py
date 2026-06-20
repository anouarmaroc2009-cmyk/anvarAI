import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import config
from agent import CodingAgent

app = FastAPI(title="Anvar AI - Coding Agent", version="2.0.0")
agent = CodingAgent()

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def index():
        return FileResponse(str(static_dir / "index.html"))


class ChatRequest(BaseModel):
    message: str


class ToolExecuteRequest(BaseModel):
    name: str
    args: dict


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/config")
async def get_config():
    return {"hasBigPickle": bool(config.OPENCODE_API_KEY), "model": config.MODEL}


@app.post("/chat")
async def chat(req: ChatRequest):
    if not config.OPENCODE_API_KEY:
        raise HTTPException(500, "OPENCODE_API_KEY not set (get one free at opencode.ai/zen)")
    chunks = []
    for chunk in agent.process_message(req.message):
        if chunk["type"] == "text":
            chunks.append(chunk["content"])
        elif chunk["type"] == "error":
            raise HTTPException(500, chunk["content"])
    return {"response": "".join(chunks)}


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if not config.OPENCODE_API_KEY:
        raise HTTPException(500, "OPENCODE_API_KEY not set")

    async def event_stream():
        for chunk in agent.process_message(req.message, stream=True):
            yield f"data: {json.dumps(chunk)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post("/tools/execute")
async def tool_execute(req: ToolExecuteRequest):
    result = agent.execute_tool(req.name, req.args)
    return {"result": result}


import os
_railway_port = os.environ.get("PORT")
if _railway_port:
    config.PORT = int(_railway_port)

if __name__ == "__main__":
    import uvicorn
    print(f"Anvar AI starting (Big Pickle) on port {config.PORT}")
    uvicorn.run("main:app", host="0.0.0.0", port=config.PORT, reload=True)
