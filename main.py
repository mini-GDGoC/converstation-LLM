from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# .env 불러오기
load_dotenv()

# FastAPI 인스턴스
app = FastAPI()

# 요청 바디 모델
class ChatRequest(BaseModel):
    message: str

# LLM 설정 (OpenAI)
llm = ChatOpenAI(
    model="gpt-3.5-turbo",  # 또는 "gpt-4", "gpt-4o"
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")  # 여기에 .env 키 들어감
)

@app.post("/chat")
async def chat(req: ChatRequest):
    response = llm.invoke([HumanMessage(content=req.message)])
    return {"reply": response.content}

@app.get("/")
def read_root():
    return {"message": f"Update"}