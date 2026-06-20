import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import config
from agent import CodingAgent

IS_VERCEL = os.environ.get("VERCEL", "") == "1"

app = FastAPI(title="AI Coding Agent", version="1.0.0")


class HistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[HistoryItem]] = []


class ChatResponse(BaseModel):
    response: str
    history: List[dict]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not config.ANTHROPIC_API_KEY:
        raise HTTPException(500, "ANTHROPIC_API_KEY not set - add it in Vercel Environment Variables")
    try:
        agent = CodingAgent()
        for h in (req.history or []):
            agent.messages.append({"role": h.role, "content": h.content})
        result = agent.process_message(req.message)
        history_out = [
            {"role": m["role"], "content": m["content"] if isinstance(m["content"], str) else "(tool call)"}
            for m in agent.messages
        ]
        return ChatResponse(response=result, history=history_out)
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=True)
