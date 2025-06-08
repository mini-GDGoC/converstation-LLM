from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from modules.dto import ButtonRequest, QuestionRequest, ScrollRequest
import os
import json
import re

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from modules.database import get_menu_info, get_menu_info_for_prompt
from modules.prompt import test1, test2, test3

from pydantic import BaseModel, ValidationError
from typing import List, Optional


# .env 불러오기
load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

# 메모리 저장소
store = {}

# 메뉴 디비
menu_db = get_menu_info_for_prompt()

class KioskResponse(BaseModel):
    matched_button: Optional[str]
    follow_up_question: str
    choices: List[str]
    action: str  # click | ask | scroll


def get_session_state(session_id: str):
    if session_id not in store:
        store[session_id] = {
            "history": InMemoryChatMessageHistory(),
            "visible_buttons": [],
            "side_bar_exists": False,
            "side_bar_point": None,
            "question": "",
            "screen_type": ""
        }
    return store[session_id]

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    return get_session_state(session_id)["history"]

# 통합 프롬프트 템플릿
prompt = ChatPromptTemplate.from_messages([
    ("system", test2),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# 체인 생성
chain = prompt | llm

# 메시지 히스토리와 함께 실행 가능한 체인 생성
conversation_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

def extract_json_from_llm(raw_response):
    text = raw_response.content.strip()
    json_str = re.sub(r"^```json|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r'\{.*\}', json_str, re.DOTALL)
    if not match:
        raise ValueError("JSON 응답을 찾을 수 없습니다.")
    json_block = match.group(0)

    try:
        parsed = json.loads(json_block)
        return KioskResponse(**parsed).dict()
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"JSON 파싱/검증 실패: {str(e)}")

# API 1: 화면 스크린샷을 통한 visible_buttons 전달 및 질문 생성
async def handle_screen_input(request: QuestionRequest):
    try:
        session = get_session_state("default_session")
        session["visible_buttons"] = request.visible_buttons
        session["side_bar_exists"] = request.side_bar_exists

        print("Visible buttons:", session["visible_buttons"])
        visible_buttons_str = [b["text"] for b in request.visible_buttons]

        raw_response = conversation_chain.invoke(
            {
                "input": "",
                "visible_buttons": visible_buttons_str,
                "question": "",
                "screen_type": "",
                "menu_db": menu_db['menu_items'],
                "side_bar_exists": request.side_bar_exists,
            },
            config={"configurable": {"session_id": "default_session"}}
        )
        # print("menu_db:", menu_db['hierarchy_text'])
        print("Raw response:", raw_response.content)
        response = extract_json_from_llm(raw_response)

        session["question"] = response.get("follow_up_question", "")
        session["screen_type"] = response.get("screen_type", "")
        print("screen_type:", session["screen_type"])

        return JSONResponse(content={"response": response})

    except Exception as e:
        return JSONResponse(
            content={"error": f"처리 중 오류가 발생했습니다: {str(e)}"}, 
            status_code=505
        )

# API 2: 사용자 발화만 받아서 버튼 선택 or 추가 질문
async def handle_user_input(request: ButtonRequest):
    try:
        session = get_session_state("default_session")
        visible_buttons_str = [b["text"] for b in session["visible_buttons"]]

        raw_response = conversation_chain.invoke(
            {
                "input": request.message,
                "visible_buttons": visible_buttons_str,
                "question": session["question"],
                "screen_type": session["screen_type"],
                "menu_db": menu_db['menu_items'],
                "side_bar_exists": session.get("side_bar_exists", False),
            },
            config={"configurable": {"session_id": "default_session"}}
        )

        response = extract_json_from_llm(raw_response)

        if not response.get("matched_button") and response.get("follow_up_question"):
            session["question"] = response["follow_up_question"]

        return JSONResponse(content={"response": response})

    except Exception as e:
        return JSONResponse(
            content={"error": f"처리 중 오류가 발생했습니다: {str(e)}"}, 
            status_code=505
        )


# API 3: 스크롤 변경 후 이미지만 받아옴. 이전 답변을 토대로 대답해야함
async def scroll_action(request: ScrollRequest):
    try:
        session = get_session_state("default_session")
        session["visible_buttons"] = request.visible_buttons
        session["side_bar_exists"] = request.side_bar_exists

        print("Visible buttons:", session["visible_buttons"])
        visible_buttons_str = ", ".join([b["text"] for b in request.visible_buttons])

        raw_response = conversation_chain.invoke(
            {
                "input": request.message,
                "visible_buttons": visible_buttons_str,
                "question": "",
                "screen_type": "",
                "menu_db": menu_db['menu_items'],
                "side_bar_exists": request.side_bar_exists,
            },
            config={"configurable": {"session_id": "default_session"}}
        )
        # print("menu_db:", menu_db['hierarchy_text'])
        print("Raw response:", raw_response.content)
        response = extract_json_from_llm(raw_response)

        if not response.get("matched_button") and response.get("follow_up_question"):
            session["question"] = response["follow_up_question"]

        return JSONResponse(content={"response": response})

    except Exception as e:
        return JSONResponse(
            content={"error": f"처리 중 오류가 발생했습니다: {str(e)}"}, 
            status_code=505
        )



async def reset_conversation_memory():
    try:
        if "default_session" in store:
            store["default_session"]["history"].clear()
            store["default_session"]["visible_buttons"] = []
            store["default_session"]["question"] = ""
            store["default_session"]["screen_type"] = ""
        return {"message": "대화 내용이 초기화되었습니다."}
    except Exception as e:
        return {"error": f"초기화 중 오류가 발생했습니다: {str(e)}"}


