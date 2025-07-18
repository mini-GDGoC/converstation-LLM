from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from modules.dto import ButtonRequest
import os
import json
import re

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from modules.database import get_menu_info

# .env 불러오기
load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.9,
    api_key=os.getenv("OPENAI_API_KEY")
)


# 메모리 저장소
store = {}

# 메뉴 디비
menu_db = get_menu_info()

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 프롬프트 템플릿 수정
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    너는 노인들의 디지털 서비스 사용을 도와주는 도우미야. 노인이 원하는 목적을 이루기 위해 도와준다고 생각해.
주어진 화면의 요소들을 가지고 어떤 질문을 할 수 있는지 질문을 만들어줘야해.
최대한 영어는 풀어서 설명하고, 자세하게 이야기 해줘.
     
다음은 사용자의 터치스크린 UI 화면에서 추출된 버튼 목록이야:
{visible_buttons}

이전 질문:
"{question}"

사용자의 발화:
"{input}"

화면 타입:
"{screen_type}"     

위 내용을 기준으로 아래 규칙에 따라 행동해줘:

1. 사용자의 발화에서 추출된 버튼(`visible_buttons`)과 일치하거나 비슷한 항목이 있으면:
   - 문맥상 사용자가 원하는 동작을 하는 버튼이 있으면 해당하는 버튼이 있다고 생각해줘.
   - 해당 버튼을 `matched_button` 항목으로 반환해줘.
   - `question`이랑 `choices`는 비워서 반환해줘.
   - 예: 사용자가 "불고기버거 주세요"라고 말했고 버튼에 "불고기버거 세트"가 있으면 그것도 매칭된 걸로 간주해.
        사용자가 "햄버거 먹고싶어"라고 말했고 버튼에 "버거"가 있으면 그것도 매칭된 걸로 간주해

2. 사용자의 발화가 대답인데, 일치하는 버튼이 없다면:
   a. 만약에 화면 타입이 "menu_select"라면
     - {menu_db}의 계층을 참고해서 이전 질문과 사용자의 발화를 통해 다음 질문과 선택지를 만들어줘
     - 선택지는 자식 노드들로 이루어져있으면 좋겠어
     - 이때 질문은 `follow_up_question` 항복으로 반환하고, 선택지는 `choices`항목으로 변환해줘
   - `visible_buttons`에서 하나를 고를 수 있도록 적절한 추가 질문을 `follow_up_question` 항목에 작성해줘.
   - 질문은 어르신도 이해할 수 있도록 친절하고 쉬운 말로 작성해줘.

3. 사용자의 발화가 질문이라면:
   - 사용자의 질문에 대해 대답과 함께, 주문을 하기 위한 추가 질문을 `follow_up_question`항목에 작성해줘.

4. 아래 JSON 형식으로만 응답해줘:
  "matched_button": "불고기버거 세트",   // 없으면 null
  "follow_up_question": "사이드나 음료도 함께 드릴까요, 버거만 드릴까요?"  // 없으면 빈 문자열
  "choices:": [] // 없으면 빈 리스트

단, `matched_button`과 `follow_up_question` 중 하나는 무조건 채워져있어야해. `choices`는 있을 수도 상황에 따라 있을 수도 있고, 없을 수도 있어

### 활용 예시

- `visible_buttons = ["불고기버거 세트", "치즈버거 세트", "감자튀김"]`
- `input = "불고기 주세요"`

→ 예상 응답:
  "matched_button": "불고기버거 세트",
  "follow_up_question": ""
  "choices:":[]


"""),
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

async def get_button(request: ButtonRequest):
    try:
        visible_buttons = ", ".join(request.visible_buttons)
        
        # "default_session" 대신 request.session_id 사용
        raw_response = conversation_chain.invoke(
            {
                "input": request.message,
                "visible_buttons": visible_buttons,
                "question": request.question,
                "screen_type": request.screen_type,
                "menu_db": menu_db
            },
            config={"configurable": {"session_id": request.session_id}}
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

# reset_button_memory 함수도 수정
async def reset_button_memory(session_id: str = None):
    try:
        if session_id:
            # 특정 세션만 초기화
            if session_id in store:
                store[session_id].clear()
                return {"message": f"세션 {session_id}의 대화 내용이 초기화되었습니다."}
            else:
                return {"message": f"세션 {session_id}이 존재하지 않습니다."}
        else:
            # 모든 세션 초기화
            count = 0
            for sess_id in store:
                store[sess_id].clear()
                count += 1
            return {"message": f"모든 세션({count}개)의 대화 내용이 초기화되었습니다."}
    except Exception as e:
        return {"error": f"초기화 중 오류가 발생했습니다: {str(e)}"}
