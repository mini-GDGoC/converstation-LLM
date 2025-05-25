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
    ("system", """
    다음은 사용자의 터치스크린 UI 화면에서 추출된 버튼 목록이야:
{visible_buttons}

사용자의 발화:
"{input}"

위 내용을 기준으로 아래 규칙에 따라 행동해줘:

1. 사용자의 발화에서 추출된 버튼(`visible_buttons`)과 일치하거나 비슷한 항목이 있으면:
   - 문맥상 사용자가 원하는 동작을하는 버튼이 있으면 해당하는 버튼이 있다고 생각해줘.
   - 해당 버튼을 `matched_button` 항목으로 반환해줘.
   - 예: 사용자가 "불고기버거 주세요"라고 말했고 버튼에 "불고기버거 세트"가 있으면 그것도 매칭된 걸로 간주해.
        사용자가 "햄버거 먹고싶어"라고 말했고 버튼에 "버거"가 있으면 그것도 매칭된 걸로 간주해.

2. 일치하는 버튼이 없다면:
   - 사용자의 의도를 바탕으로 적절한 **추가 질문**을 `follow_up_question` 항목에 작성해줘.
   - 질문은 어르신도 이해할 수 있도록 친절하고 쉬운 말로 작성해줘.

3. 사용자의 발화가 질문이라면:
   - 사용자의 질문에 대해 대답과 함께 추가 질문을 `follow_up_question`항목에 작성해줘.

3. 아래 JSON 형식으로만 응답해줘:
  "matched_button": "불고기버거 세트",   // 없으면 null
  "follow_up_question": "세트로 드릴까요, 버거만 드릴까요?"  // 없으면 빈 문자열

단, 두 항목중 하나는 반드시 채워져있어야해.

### ✅ 활용 예시

- `visible_buttons = ["불고기버거 세트", "치즈버거 세트", "감자튀김"]`
- `input = "불고기 주세요"`

→ 예상 응답:
  "matched_button": "불고기버거 세트",
  "follow_up_question": ""


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

async def get_button(message: ChatRequest):
    try:
        visible_buttons = ", ".join(message.visible_buttons)
        
        raw_response = conversation_chain.invoke(
            {
                "input": message.message,
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

async def reset_button_memory():
    try:
        # 세션 히스토리 초기화
        if "default_session" in store:
            store["default_session"].clear()
        return {"message": "대화 내용이 초기화되었습니다."}
    except Exception as e:
        return {"error": f"초기화 중 오류가 발생했습니다: {str(e)}"}
