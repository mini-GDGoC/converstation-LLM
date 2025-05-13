from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from modules.llm_model import init_model, get_model
from modules.database import get_db
from modules.models import MenuItem
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from fastapi.responses import JSONResponse
from langchain_core.messages import BaseMessage
from langchain.prompts import PromptTemplate

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

custom_prompt = PromptTemplate.from_template(""" 
                                             너는 디지털 기기 사용을 어려워 하시는 노인들을 도와주는 AI야.
                                             너는 절대 딱딱하게 말하지 않고, 부드러운 말투로 이야기해. 존댓말을 사용해야 해.
                                             영어로 된 단어들은 최대한 한국어로 풀어서 이해하기 쉽게 설명해줘.
                                            
                                             대화 기록:
                                             {history}
                                             사용자: {input}
                                             AI:
                                             """)

# 새 ConversationChain 생성
memory = ConversationBufferMemory(return_messages=True)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=custom_prompt,
    verbose=True
)

@app.post("/chat")
async def chat(req: ChatRequest):
    response = conversation.predict(input=req.message)
    return {"reply": response}

@app.get("/chat-history")
async def chat_history():
    history = [
        {"role": m.type, "content": m.content}
        for m in memory.chat_memory.messages
    ]
    return JSONResponse(content={"history": history})


# 결제 완료 되면 memory 초기화
@app.post("/reset-chat")
async def reset_chat():
    memory.clear()
    return {"message": "대화 내용이 초기화되었습니다."}

@app.get("/")
def read_root():
    return {"message": f"Update"}
