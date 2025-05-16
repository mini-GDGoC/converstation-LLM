from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import easyocr
from PIL import Image
import io
import time
import pytesseract

import matplotlib.pyplot as plt
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

# .env 불러오기
load_dotenv()

# FastAPI 인스턴스
app = FastAPI()

# ocr reader
reader = easyocr.Reader(['ko', 'en'])

# 요청 바디 모델
class ChatRequest(BaseModel):
    message: str
    visible_buttons: list[str] = []


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
    start_time = time.time()
    response = conversation.predict(input=req.message)
    total_time = round(time.time() - start_time, 4)

    return JSONResponse(content={"response": response, "process_time": total_time})

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

@app.post("/ocr-test")
async def ocr_test(file: UploadFile = File(...)):
    start_time = time.time()
    # image load
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)

    # opencv용 bgr로 변환
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 3. CLAHE 객체 생성 (clipLimit 높일수록 대비 강해짐)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)

    # 밝은 영역만 마스킹 (threshold 적용)
    _, bright_mask = cv2.threshold(contrast_enhanced, 100, 255, cv2.THRESH_BINARY)
    bright_only = cv2.bitwise_and(image_bgr, image_bgr, mask=bright_mask)
    bright_rgb = cv2.cvtColor(bright_only, cv2.COLOR_BGR2RGB)

    # detect text and position
    results = reader.readtext(bright_rgb)

    buttons = []
    for (bbox, text, prob) in results:
        if prob > 0.5:  # 신뢰도 기준
            (tl, tr, br, bl) = bbox
            x_min = int(min(tl[0], bl[0]))
            y_min = int(min(tl[1], tr[1]))
            x_max = int(max(tr[0], br[0]))
            y_max = int(max(bl[1], br[1]))
            buttons.append({
                "text": text,
                "bbox": {
                    "x": x_min,
                    "y": y_min,
                    "width": x_max - x_min,
                    "height": y_max - y_min
                }
            })

    visible_button_texts = [b['text'] for b in buttons]
    conversation.prompt.partial_variables = {"visible_buttons": ', '.join(visible_button_texts)}

    # LLM에게 질문 추천 요청
    question_prompt = f"지금 화면에 보이는 메뉴 항목은 다음과 같아: {', '.join(visible_button_texts)}. 이걸 보고 어르신에게 어떤 질문을 하면 좋을까? 한문장 정도의 질문으로 해줘."
    suggested_question = conversation.predict(input=question_prompt)

    total_time = round(time.time() - start_time, 4)
    return JSONResponse(content={
        "buttons": buttons,
        "suggested_question": suggested_question,
        "process_time": total_time
    })