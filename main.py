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
    if not config.GOOGLE_API_KEY:
        raise HTTPException(
            500,
            "GOOGLE_API_KEY not set - add it in Railway Environment Variables "
            "(get one free at https://aistudio.google.com/apikey)",
        )
    try:
        result = agent.process_message(req.message)
        return ChatResponse(response=result)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/history")
async def history():
    return {"messages": agent.get_history()}


@app.post("/reset")
async def reset():
    agent.reset()
    return {"status": "conversation reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=True)
