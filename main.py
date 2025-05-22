from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from modules.llm_model import init_model, get_model
from modules.database import get_db, get_menu_info
from modules.models import MenuItem
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from fastapi.responses import JSONResponse
from langchain_core.messages import BaseMessage
from langchain.prompts import PromptTemplate
from modules.tts import get_tts, TTS_testReq
from modules.stt import get_stt, STT_testReq

# .env 불러오기
load_dotenv()

# FastAPI 인스턴스
app = FastAPI()


# 요청 바디 모델
class ChatRequest(BaseModel):
    message: str


# LLM 설정 (OpenAI)
llm = ChatOpenAI(
    model="gpt-4o",  # 또는 "gpt-4", "gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")  # 여기에 .env 키 들어감
)


# 새 ConversationChain 생성
memory = ConversationBufferMemory(return_messages=True)

@app.on_event("startup")
async def startup():
    init_model()
    print("Model init")
    # print(get_model().invoke("하이")) #실제 모델 동작 하는지 테스트 (주석 처리 해놓지 않으면 리로드 할 때마다 계속 호출 해서 api 사용량 까먹음)

    db = get_db()
    print("DB init")
    global conversation
    menu_info = get_menu_info()
    custom_prompt = PromptTemplate.from_template(f""" 
                                             너는 디지털 기기가 익숙하지 않은 어르신들을 도와주는 따뜻한 AI 도우미야.
- 항상 존댓말을 사용하고, 말투는 부드럽고 친절해야 해. 손자처럼 친근한 말투로 해줘.
- 어려운 기술 용어나 영어 표현은 쓰지 말고, 쉬운 단어로 바꿔서 설명해줘.
- 어르신이 메뉴에 없는 항목을 말하셔도, “없습니다”라고 단정하지 말고 **비슷한 메뉴나 상위 분류 기준으로 자연스럽게 유도해줘.**
- 주문이 다 끝난 것 같으면 "결제를 진행합니다"라고 말해줘.
---

### 🔁 트리 기반 추천 방식

- **사용자가 상위 메뉴(예: 햄버거)** 를 언급하면:
  - `햄버거` 메뉴의 하위 분류인 **소고기 / 닭고기 / 새우** 중 어떤 재료가 좋으신지 질문해줘.
  - 예: “햄버거가 드시고 싶으시군요! 소고기, 닭고기, 새우 중에 어떤 고기를 넣은 햄버거가 좋으세요?”

- **사용자가 중간 분류(예: 소고기)** 를 선택하면:
  - 그 하위 메뉴들(예: 고기 두 장, 달달한 소스, 치즈 많이 등)을 기준으로 질문을 유도해줘.
  - 예: “소고기가 들어간 햄버거는 더블패티버거, 불고기버거, 더블치즈버거 같은 메뉴가 있어요. 어떤 스타일이 더 끌리세요?”

- AI는 항상 트리 구조를 기억하고, 선택지가 있다면 하위 메뉴를 보여주고 질문으로 유도해줘.

---

### 🧠 지금까지의 메뉴 분류 구조는 다음과 같아:
{menu_info}

---
                                             - 이모티콘은 사용하지 말아줘.
                                        
                                             대화 기록:
                                             {{history}}
                                             사용자: {{input}}
                                             AI:
                                             """)  
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=custom_prompt,
        verbose=True
    )

    # for i in db.query(MenuItem).all():
    #     print(i.id, i.parent_id, i.name, i.description, i.emoji, i.keywords)



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


@app.post("/test_tts")
async def test_tts(req: TTS_testReq):
    return get_tts(req.fileName, req.text)



@app.post("/stt-test")
async def stt_test(req: STT_testReq):
    return get_stt(req.ileName)