from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from modules.llm_model import init_model, get_model
from modules.database import get_db
from modules.models import MenuItem
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
    model="gpt-3.5-turbo",  # 또는 "gpt-4", "gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")  # 여기에 .env 키 들어감
)


@app.on_event("startup")
async def startup():
    init_model()
    print("Model init")
    # print(get_model().invoke("하이")) #실제 모델 동작 하는지 테스트 (주석 처리 해놓지 않으면 리로드 할 때마다 계속 호출 해서 api 사용량 까먹음)

    db = get_db()
    print("DB init")
    # for i in db.query(MenuItem).all():
    #     print(i.id, i.parent_id, i.name, i.description, i.emoji, i.keywords)


@app.post("/chat")
async def chat(req: ChatRequest):
    response = llm.invoke([HumanMessage(content=req.message)])
    return {"reply": response.content}


@app.get("/")
def read_root():
    return {"message": f"Update"}
