import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from anthropic import Anthropic

app = FastAPI(title="EDDY CHATBOT API")

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

SYSTEM_PROMPT = (
    "You are EDDY CHATBOT, Your Intelligent Personal Assistant, "
    "powered by EDDY YUNG MEDIA. Answer questions on any topic "
    "clearly and helpfully."
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

@app.get("/")
def home():
    return {"message": "EDDY CHATBOT API is running"}

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        reply = client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[m.model_dump() for m in req.messages],
        )
        return {"reply": reply.content[0].text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
