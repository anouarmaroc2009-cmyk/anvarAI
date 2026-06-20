from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import config
from agent import CodingAgent

app = FastAPI(title="AI Coding Agent", version="1.0.0")
agent = CodingAgent()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not config.ANTHROPIC_API_KEY:
        raise HTTPException(500, "ANTHROPIC_API_KEY not set")
    try:
        result = agent.process_message(req.message)
        return ChatResponse(response=result)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/history")
async def history():
    raw = agent.get_history()
    cleaned = []
    for m in raw:
        if isinstance(m["content"], list):
            items = []
            for b in m["content"]:
                if isinstance(b, dict):
                    items.append(b)
                elif b.type == "tool_use":
                    items.append({"type": "tool_use", "name": b.name, "input": b.input})
                else:
                    items.append({"type": "text", "text": b.text})
            cleaned.append({"role": m["role"], "content": items})
        else:
            cleaned.append({"role": m["role"], "content": m["content"]})
    return {"messages": cleaned}


@app.post("/reset")
async def reset():
    agent.reset()
    return {"status": "conversation reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=True)
