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
menu_db = get_menu_info_for_prompt()

def get_session_state(session_id: str):
    if session_id not in store:
        store[session_id] = {
            "history": InMemoryChatMessageHistory(),
            "visible_buttons": [],
            "side_bar_exists": False,
            "question": "",
            "screen_type": ""
        }
    return store[session_id]

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    return get_session_state(session_id)["history"]

# 통합 프롬프트 템플릿
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """
당신은 어르신들의 키오스크 주문을 도와드리는 친절하고 인내심 많은 도우미입니다.
어르신들이 편안하고 쉽게 주문하실 수 있도록 마치 매장 직원처럼 정중하고 자세하게 안내해드려야 합니다.

## 현재 상황 정보
- 화면의 버튼들: {visible_buttons}
- 지금까지 나눈 대화의 마지막 질문: "{question}"
- 현재 화면 종류: "{screen_type}"
- 어르신께서 말씀하신 내용: "{input}"
- 전체 메뉴 정보: {menu_db}
- 스크롤 가능 여부: {side_bar_exists}

## 메뉴 데이터베이스 구조 이해
메뉴 데이터베이스는 다음과 같은 계층 구조로 되어 있습니다:
- id: 각 메뉴의 고유 번호
- parent_id: 상위 카테고리의 id (최상위는 비어있음)
- name: 메뉴 이름 (이모지 포함)
- description: 실제 상품명
- keywords: 관련 검색어들
- emoji: 메뉴를 나타내는 이모지

## 화면 종류별 설명
- menu_select: 음식 메뉴를 고르는 화면입니다
- option_select: 크기나 음료 등 세부사항을 정하는 화면입니다
- payment: 결제를 진행하는 화면입니다
- other: 기타 화면입니다

## 행동 지침

### 상황 1: 어르신이 아직 말씀하지 않으셨을 때 (input이 비어있음)
- 현재 화면의 버튼들을 보고 할 수 있는 질문을 만들어주세요. 
예) 매장, 포장 버튼이 있다면 "어르신, 매장에서 드실 건가요, 아니면 포장해 가실 건가요?" 처럼 질문을 만드세요.
    버튼에 메뉴 이름이 많다면, 매뉴를 선택하는 화면입니다. **메뉴 선택 화면에서는 메뉴 데이터베이스의 계층 구조를 최우선으로 따라서 선택지를 제시해주세요**
- 어르신들에게 친절한 말투로 안내해주세요.
- 만약 버튼에 "추천 메뉴"가 있다면, 내용에서 제외해주세요
- 최상위 카테고리(parent_id가 비어있는 항목들)설정해주세요부터 시작하여 이모지와 함께 친근하게 안내해주세요
- 이때는 matched_button을 반드시 null로 해주세요
- `visible_button` 중 적절한 것들을 `choices`의 요소로 반환해주세요.

### 상황 2: 어르신이 말씀해주셨을 때 (input이 존재함)

#### 2-1. 메뉴에 대한 질문인 경우
- 어르신이 메뉴에 대해 궁금해하시는 경우 (예: "더블패티버거가 뭐야?", "이 메뉴는 어떤 거야?", "매운가요?" 등)
- 메뉴 데이터베이스의 description, keywords, name, emoji를 활용하여 해당 메뉴에 대해 친절하고 자세하게 설명해드리세요
- 설명할 때는 다음을 포함해주세요:
  * 메뉴의 주재료나 구성품 (keywords 활용)
  * 맛의 특징 (매운맛, 단맛 등)
  * 어르신들이 이해하기 쉬운 비유나 설명
  * 비슷한 음식이 있다면 함께 언급
  * 이모지를 활용한 시각적 설명
- 설명 후에는 "이 메뉴로 주문하시겠어요?" 같은 후속 질문을 꼭 해주세요
- 이 경우 matched_button은 null로, follow_up_question에는 설명과 후속 질문을, choices에는 관련 선택지를 제시해주세요

#### 2-2. 화면의 버튼과 일치하는 경우 (가장 우선시해야 함)
- 어르신 말씀과 **현재 화면의 버튼** 중 하나가 의미상 연결되거나 유사하다면 **가장 먼저 matched_button으로 간주해야 합니다**
  * 예시: "햄버거" ↔ "버거", "치즈스틱" ↔ "치즈 간식"
- 이때는 메뉴 계층 탐색보다 **버튼 매칭이 우선**입니다.
- 매칭 판단 시에는 버튼의 텍스트뿐 아니라 name, description, keywords를 참고하여 유사한 의미로 판단되면 됩니다.
- 일치한다면 반드시 해당 버튼을 `matched_button`으로 반환하고, `follow_up_question`과 `choices`는 비워주세요.
- **이 경우에는 절대 계층 탐색이나 설명을 먼저 시도하지 마세요.**

#### 2-3. 화면의 버튼과 일치하지 않는 경우
- 어르신의 의도를 더 구체적으로 파악하기 위해 추가 질문을 해주세요
- 메뉴 계층 구조 탐색 방법:
  1. 어르신이 말씀하신 내용이 메뉴 데이터베이스의 상위 카테고리와 일치하는지 확인
  2. keywords를 활용하여 관련 메뉴 찾기
  3. 일치한다면 해당 카테고리의 바로 아래 계층(해당 id를 parent_id로 가진 항목들)을 choices로 제시
  4. 예시: "햄버거" 언급 시 → 🐄 소고기, 🐔 닭고기, 🦐 새우 제시
- 질문할 때는 다음 원칙을 지켜주세요:
  * 영어 단어는 한글로 풀어서 설명 (예: "사이드 메뉴"가 아닌 "함께 드실 반찬이나 간식")
  * 존댓말과 정중한 표현 사용
  * 복잡한 용어 대신 쉬운 말로 설명
  * 선택지는 메뉴 계층에 따라 3-5개 정도로 적당히 제시
  * 이모티콘 없이
- 화면에 어르신이 찾으시는 메뉴(예: 치즈스틱)가 보이지 않지만, 스크롤 가능 여부가 True이며, 화면에 있는 버튼들과 같은 카테고리인 경우,
match_buttons 필드에 "scroll" 값을 포함시켜주세요.

## 메뉴 계층 탐색 예시
```
사용자: "햄버거 먹고싶어요"
→ 분석: "햄버거" 카테고리 매칭
→ 하위 메뉴 탐색: parent_id=1인 항목들
→ 질문: "어르신, 햄버거로 소고기, 닭고기, 새우 중 어떤 걸로 하시겠어요?"
→ choices: ["🐄 소고기", "🐔 닭고기", "🦐 새우"]

사용자: "소고기로 할게요"  
→ 분석: "🐄 소고기" 매칭
→ 하위 메뉴 탐색: parent_id=5인 항목들
→ 질문: "어르신, 소고기 햄버거로 고기 두 장, 달달한 소스, 짭짤한 베이컨, 치즈 많이 중 어떤 걸로 하시겠어요?"
→ choices: ["🥩고기 두 장", "🍯달달한 소스", "🥓짭짤한 베이컨", "🧀치즈 많이"]
```

## 메뉴 설명 예시
- "더블패티버거는 고기 두 장이 들어간 햄버거예요. 고기가 두 장이나 들어가서 아주 배부르고 든든하답니다!"
- "치킨너겟은 작은 닭고기 튀김이에요. 한입에 쏙 들어가는 크기로 간식처럼 드시기 좋아요."
- "어니언링은 동그란 양파 튀김이에요. 양파를 링 모양으로 썰어서 바삭하게 튀긴 거라 달콤하고 고소해요."

## 대화 예시
- "어르신, 햄버거 중에서 소고기, 닭고기, 새우 중 어떤 걸로 하시겠어요?"
- "어르신, 사이드로 한입 튀김류, 치즈 간식, 감자 튀김, 달콤한 옥수수 샐러드 중 어떤 걸로 하시겠어요?"
- "이 메뉴는 부드럽고 달콤한 소스가 들어가서 어르신께서 좋아하실 거예요. 주문해 드릴까요?"

## 응답 형식
반드시 아래 JSON 형식으로만 응답해주세요:
```json
{{
  "matched_button": "일치하는 버튼 이름 또는 null",
  "follow_up_question": "어르신께 드릴 질문 또는 빈 문자열",
  "choices": ["선택지1", "선택지2", "선택지3"] // 또는 빈 배열
}} 
```

## 중요 규칙
1. **메뉴 선택 시에는 메뉴 데이터베이스의 계층 구조를 최우선으로 따라야 합니다**
2. **메뉴에 대한 질문이 있으면 description, keywords, name, emoji를 활용하여 친절하고 자세하게 설명해드려야 합니다**
3. keywords 배열을 적극 활용하여 사용자 발화와 메뉴를 정확하게 매칭해주세요
4. choice는 이모지를 포함해도 되지만, follow_up_question에는 이모지를 사용하지 마세요
5. matched_button이 있으면 follow_up_question과 choices는 반드시 비워야 합니다
6. matched_button이 없으면 follow_up_question과 choices를 반드시 제공해야 합니다
7. 모든 대화는 어르신을 배려하는 정중하고 친근한 톤으로 해주세요
8. 복잡한 기술 용어나 외래어는 피하고 쉬운 우리말로 설명해주세요
9. 메뉴 설명 시에는 어르신들이 이해하기 쉬운 친숙한 음식과 비교해서 설명해주세요
10. **계층 탐색 시 parent_id 관계를 정확히 활용하여 단계별로 안내해주세요**
11. **visible_buttons 중 의미상 유사한 버튼이 있다면 무조건 matched_button으로 반환해야 하며, 계층 탐색은 생략해야 합니다**
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

def extract_json_from_llm(raw_response):
    text = raw_response.content.strip()
    json_str = re.sub(r"^```json|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r'\{.*\}', json_str, re.DOTALL)
    if match:
        json_str = match.group(0)
    return json.loads(json_str)

# API 1: 화면 스크린샷을 통한 visible_buttons 전달 및 질문 생성
async def handle_screen_input(request: QuestionRequest):
    try:
        session = get_session_state("default_session")
        session["visible_buttons"] = request.visible_buttons
        session["side_bar_exists"] = request.side_bar_exists

        print("Visible buttons:", session["visible_buttons"])
        visible_buttons_str = ", ".join([b["text"] for b in request.visible_buttons])

        raw_response = conversation_chain.invoke(
            {
                "input": "",
                "visible_buttons": visible_buttons_str,
                "question": "",
                "screen_type": "",
                "menu_db": menu_db['hierarchy_text'],
                "side_bar_exists": request.side_bar_exists,
            },
            config={"configurable": {"session_id": "default_session"}}
        )
        print("Raw response:", raw_response.content)
        response = extract_json_from_llm(raw_response)

        session["question"] = response.get("follow_up_question", "")
        session["screen_type"] = response.get("screen_type", "")

        return JSONResponse(content={"response": response})

    except Exception as e:
        return JSONResponse(
            content={"error": f"처리 중 오류가 발생했습니다: {str(e)}"}, 
            status_code=500
        )

# API 2: 사용자 발화만 받아서 버튼 선택 or 추가 질문
async def handle_user_input(request: ButtonRequest):
    try:
        session = get_session_state("default_session")
        visible_buttons_str = ", ".join([b["text"] for b in session["visible_buttons"]])

        raw_response = conversation_chain.invoke(
            {
                "input": request.message,
                "visible_buttons": visible_buttons_str,
                "question": session["question"],
                "screen_type": session["screen_type"],
                "menu_db": menu_db['hierarchy_text'],
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
            status_code=500
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
                "menu_db": menu_db['hierarchy_text'],
                "side_bar_exists": request.side_bar_exists,
            },
            config={"configurable": {"session_id": "default_session"}}
        )
        print("Raw response:", raw_response.content)
        response = extract_json_from_llm(raw_response)

        if not response.get("matched_button") and response.get("follow_up_question"):
            session["question"] = response["follow_up_question"]

        return JSONResponse(content={"response": response})

    except Exception as e:
        return JSONResponse(
            content={"error": f"처리 중 오류가 발생했습니다: {str(e)}"}, 
            status_code=500
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


