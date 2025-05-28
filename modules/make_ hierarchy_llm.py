from fastapi.responses import JSONResponse
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from modules.dto import ChatRequest
import os

# .env 불러오기
load_dotenv()

# LLM 정의
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.9,
    api_key=os.getenv("OPENAI_API_KEY")
)

# 대화 메모리
memory = ConversationBufferMemory(return_messages=True)

# 프롬프트 정의
custom_prompt = PromptTemplate.from_template("""
당신은 사용자 경험(UX) 디자이너이자 노인 친화형 키오스크 설계 전문가입니다.
다음 조건에 따라 **노인을 위한 트리 구조 메뉴**를 생성해 주세요.

📌 반드시 지켜야 할 핵심 조건:

1. **모든 메뉴 항목은 절대 빠짐없이 포함**해야 합니다.

   * 누락된 메뉴가 단 1개라도 있으면 안 됩니다.
   * 메뉴 목록은 아래 “입력 메뉴” 항목에 모두 제시됩니다.
   * "입력 메뉴"의 각 칵테고리 별로 트리를 만듭니다.

2. **트리 구조로 분류**해 주세요.

   * 고기 종류나 특징(예: 소고기/닭고기/새우 등)을 기준으로 분기
   * 자식 노드는 최대 3\~4개를 넘지 않도록 분기해 주세요.
   * 음료일 경우 기본적으로 "차가운"/"뜨거운"으로 먼저 분기해주세요. 음료의 온도만 다를 경우 온도를 제외한 설명은 똑같이 유지해주세요.
   * 비슷한 종류(예: 튀김류, 치즈류)는 의미 있는 상위 그룹으로 묶어 주세요.
   * 메뉴의 주요 재료가 다르더라도, 조리 방식(예: 튀김), 먹는 방식(예: 한입 크기), 활용도(예: 간식용) 등이 유사한 경우에는 하나의 의미 있는 상위 그룹으로 묶어 주세요.
   * 예: 치킨너겟, 어니언링, 통오징어링 → '🍢한입 튀김류' 또는 '작은 튀김 간식' 등
   * 하위 메뉴가 1개만 있는 경우에는 중간 설명 없이 메뉴로 바로 연결합니다.
   * 예:

```
닭고기 -> 튀긴 닭고기 -> 치킨버거 (X)
닭고기 -> 치킨버거 (O)
```

* 마지막 리프 노드가 메뉴 이름이 되도록합니다.
* 예:

```
   ├─ 🍔 햄버거
│  ├─ 🐄 소고기
│  │  ├─ 🥩고기 두 장 → 더블패티버거 | 키워드: [고기, 두 장, 배부른]
│  │  ├─ 🍯달달한 소스 → 불고기버거 | 키워드: [불고기, 달콤한, 부드러운]
│  │  ├─ 🥓짭짤한 베이컨 → 베이컨버거 | 키워드: [베이컨, 짭짤한, 고소한]
│  │  └─ 🧀치즈 많이 → 더블치즈버거 | 키워드: [치즈, 진한 맛, 고기]
│  ├─ 🐔 닭고기
│  │  └─ 치킨버거 (🍗 튀긴 닭고기) | 키워드: [닭, 바삭한, 담백한]
│  └─ 🦐 새우
│     └─ 새우버거 (🦐 튀긴 새우) | 키워드: [새우, 바삭한, 해산물]   
```

3. **각 메뉴 항목에는 설명과 키워드를 추가**해 주세요.

   * 설명은 어르신도 이해할 수 있는 쉬운 표현으로 해주세요
   * 키워드는 어르신이 말로 검색하거나 누르기 쉽게, 최대한 쉬운 단어로 구성해주세요
   * 예

```
동그란 치즈볼 -> 동그란 치즈 튀김
```

* 유사 메뉴는 그룹화된 하위 항목(예: 치즈 → 치즈볼 포함)으로 정리해 주세요.
* 키워드에 같은게 있으면 최대한 그룹화해주세요.

4. **설명 앞에 직관적인 이모지를 붙여 주세요. 단, 의미가 명확할 때만 사용하고 억지로 붙이지 마세요.**

   * 예: 🧅 양파 튀김 → 어니언링
   * 의미 구분이 어려운 경우(예: '길쭉한 치즈')는 이모지를 생략합니다.

5. **비슷한 메뉴는 차별화된 설명으로 구분**해 주세요.

   * 예: 치즈스틱 → "긴 치즈 튀김", 롱치즈스틱 → "더 긴 치즈 튀김"

6. \*\*최종 출력은 시각적 트리 형태(들여쓰기 + ├─, └─)\*\*로 표시해 주세요.

   * 예:

     ```
     ├─ 🐄 소고기
     │  └─ 🧀치즈 많이 → 더블치즈버거
     ```

---

**입력 메뉴 목록**
{menu}

""")

# FastAPI에서 호출될 함수
async def make_hierarchy(menu: str):
    prompt = custom_prompt.format(menu=menu)
    response = llm.invoke(prompt)
    return JSONResponse(content={"response": response})

async def reset_hierarchy_memory():
    memory.clear()
    return {"message": "대화 내용이 초기화되었습니다."}