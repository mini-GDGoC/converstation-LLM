from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from modules.dto import ChatRequest
import os
import json
import re

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# .env 불러오기
load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.9,
    api_key=os.getenv("OPENAI_API_KEY")
)


# 메모리 저장소
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 프롬프트 템플릿 수정
prompt = ChatPromptTemplate.from_messages([
    ("system", """다음은 사용자의 터치스크린 UI 화면에서 추출한 텍스트 요소 목록이야:
    {visible_buttons}

1. 이 화면이 어떤 목적인 화면인지 아래 중에서 판단해줘:
    1-1. 메뉴를 선택하는 화면 (특정 메뉴 이름들로 이루어져있다면 메뉴를 선택하는 화면이야)
    1-2. 옵션을 선택하는 화면 (예: 크기, 음료, 포장여부 등)
    1-3. 결제를 진행하는 화면
    1-4. 그 외

2. 너는 친절한 어르신 도우미야. 다음 기준으로 행동해줘:

- 만약 "메뉴 선택 화면"이라면:
  - "question" 과, "choice"는 빈 문자열로 응답해줘

- 만약 "옵션 선택 화면"이라면:
  - 어떤 옵션을 선택해야 하는지 질문을 만들어줘
  - 노인 친화적으로 영어로 된 단어는 풀어서 설명하고, 최대한 자세하게 질문을 만들어줘
  - 최대한 선택지를 포함해서 질문 만들어줘
  - 화면의 요소들 중에서 누를 버튼을 "choices" 리스트로 만들어줘
  - 예: "음료를 선택해주세요", "사이즈를 선택해주세요", "세트는 음료와 다른 추가 메뉴를 함께 먹을 수 있어요"

- 만약 "결제 화면"이라면:
  - 결제를 유도하는 한 문장을 생성해줘
  - 화면의 요소들 중에서 누를 버튼을 "choices" 리스트로 만들어줘
  - 예: "주문이 완료되었어요. 결제를 진행해볼까요?"

- 만약 "그 외" 라면:
  - 화면에 보이는 상황에서 필요한 질문을 생성해줘
  - 노인 친화적으로 영어로 된 단어는 풀어서 설명하고, 최대한 자세하게 질문을 만들어줘
  - 화면의 요소들 중에서 누를 버튼을 "choices" 리스트로 만들어줘

- 반드시 다음 형식으로 JSON으로 응답해줘:
"screen_type": "menu_select / option_select / payment / other",
"question": "어떤 메뉴를 선택하시겠어요?",
"choices": ["불고기버거 세트", "치즈버거 세트", "감자튀김"] // 없으면 []
```
"""),
    MessagesPlaceholder(variable_name="history"),
    # ("human", "{input}")
])

# 체인 생성
chain = prompt | llm

# 메시지 히스토리와 함께 실행 가능한 체인 생성
conversation_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    # input_messages_key="input",
    history_messages_key="history"
)
async def divide_question(message: ChatRequest):
    try:
        visible_buttons = ", ".join(message.visible_buttons)
        
        raw_response = conversation_chain.invoke(
            {
                # "input": message.message,
                "visible_buttons": visible_buttons
            },
            config={"configurable": {"session_id": "default_session"}}
        )
        
        # LLM 응답에서 JSON만 추출
        text = raw_response.content.strip()

        # ```json ... ``` 제거
        json_str = re.sub(r"^```json|```$", "", text.strip(), flags=re.MULTILINE).strip()

        # JSON 파싱
        parsed = json.loads(json_str)

        return JSONResponse(content={"response": parsed})
        # response.content에서 실제 응답 추출
        # return JSONResponse(content={"response": response.content})
        
    except Exception as e:
        return JSONResponse(
            content={"error": f"처리 중 오류가 발생했습니다: {str(e)}"}, 
            status_code=500
        )

async def reset_divide_memory():
    try:
        # 세션 히스토리 초기화
        if "default_session" in store:
            store["default_session"].clear()
        return {"message": "대화 내용이 초기화되었습니다."}
    except Exception as e:
        return {"error": f"초기화 중 오류가 발생했습니다: {str(e)}"}
