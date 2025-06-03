from fastapi import FastAPI, File, UploadFile, Form
import json
from dotenv import load_dotenv

from modules.get_action import get_action_from_audio
from modules.llm_model import init_model, get_model
from modules.database import get_db, get_menu_info
from modules.divide_question_llm import divide_question, reset_divide_memory
from modules.test_one_llm import handle_screen_input, handle_user_input, reset_conversation_memory, get_session_state, \
    scroll_action
from modules.ocr import run_ocr
from modules.get_question import get_question_from_image

from langchain.chains import ConversationChain
from fastapi.responses import JSONResponse
from modules.tts import get_tts, TTS_testReq
from modules.stt import get_stt, STT_testReq, get_stt_from_file_obj

from modules.dto import ChatRequest, ButtonRequest, QuestionRequest, ScrollRequest
from fastapi.middleware.cors import CORSMiddleware


# .env 불러오기
load_dotenv()

# FastAPI 인스턴스
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
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


@app.get("/")
def read_root():
    return {"message": f"Update"}


@app.post("/test_tts", deprecated=True)
async def test_tts(req: TTS_testReq):
    return get_tts(req.fileName, req.text)


@app.post("/stt-test", deprecated=True)
async def stt_test(req: STT_testReq):
    return get_stt(req.fileName)

@app.post("/ocr-test", deprecated=True)
async def ocr_test(file: UploadFile = File(...)):
    return await run_ocr(file)


@app.post("/get_button/chat", deprecated=True)
async def get_button_llm(req: ButtonRequest):
    return await handle_user_input(req)

@app.post("/reset-conversation", deprecated=True)
async def reset_button_llm():
    return await reset_conversation_memory()

@app.post("/divide_question/chat", deprecated=True)
async def divide_question_llm(req: QuestionRequest):
    return await handle_screen_input(req)

@app.post("/get-question", summary="1번 api 스크린샷을 보내고 사용자에게 보여줄 질문을 받는다.")
async def get_question(file: UploadFile = File(...)):
    """
    스크린 샷을 보내면 사용자에게 질문할 음성과 선택지들을 보내주는 api

        {
            "follow_up_question": follow_up_question,
            "choices": options,
            "tts_file": s3_url,
            "sidebar": scrollbar_exists
        }
    """
    return await get_question_from_image(file)

@app.post("/get_action", summary="사용자 응답을 바탕으로 좌표를 받거나 추가 질문들 받을 수도, 좌표는 버튼과 스크롤바의 정보를 알려줌")
async def get_action(file: UploadFile = File(...)):
    """
    사용자의 음성 파일을 주면 응답반환
    버튼을 찾은 경우 클릭이면 버튼 시작 좌표와 높이 정보줌
    사이드 바도 시작 좌표와 너비 높이 줌

        {
            "action": "click" or "scroll",
            "text": 버튼이름,
            "bbox":{
                "x": x1,
                "y": y1,
                "width": x2,
                "height": y2,
            }
        }

    버튼을 찾지 못한 경우

        {
            "follow_up_question_url": obj_url,
            "choices": [],
            "user_answer": user_answer,
        }
    """
    return await get_action_from_audio(file)

@app.post("/get-action-scroll", summary="모르겠음 여진이가 잘 설명 부탁")
async def get_action_scroll(file: UploadFile = File(...), message: str = Form(...)):
    """
    스크롤 api

        {
            "answer_text": answer_text,
            "choices": []  //option list,
            "answer_audio": tts_file_url,
            "action": action
        }
    """


    # 스크롤이 존재했을 경우, 스크롤 한 화면을 새롭게 받아서, 그 전 사용자의 답변을 기반으로 answer_text, answer_audio, action(click | scroll)) response

    # OCR 실행
    ocr_response = await run_ocr(file)
    ocr_data = ocr_response.body
    ocr_json = json.loads(ocr_data.decode())  # bytes → str → dict

    # llm 모델을 사용하여 질문 생성
    visible_buttons = [{"text": group["text"], "bbox": group["bbox"]}
                  for group in ocr_json.get("groups", [])]
    # sidebar_exists를 bool로 변환
    scrollbar_exists = ocr_json.get("sidebar_exists", False)
    scrollbar_exists_bool = bool(scrollbar_exists)
    print("Visible buttons:", visible_buttons, "scrollbar_exists:", scrollbar_exists)
    req = ScrollRequest(visible_buttons=visible_buttons, side_bar_exists=scrollbar_exists_bool, message=message)
    llm_response = await scroll_action(req)
    print("LLM Response:", llm_response)
    # llm_response.body는 bytes이므로 디코딩 후 파싱
    llm_json = json.loads(llm_response.body.decode())
    print("llm_json:", llm_json)
    response_data = llm_json.get("response", llm_json)
    answer_text = response_data.get("follow_up_question", "")
    options = response_data.get("choices", [])
    action = response_data.get("matched_button", None)

    tts_file = None
    if answer_text:
        tts_file = get_tts("question", answer_text)

    return JSONResponse(content={
        "answer_text": answer_text,
        "choices": options,
        "answer_audio": tts_file,
        "action": action
    })