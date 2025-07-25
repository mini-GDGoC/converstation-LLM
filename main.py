from fastapi import FastAPI, File, UploadFile, Form
import json
from dotenv import load_dotenv

from modules.get_action import get_action_from_audio, get_action_from_text
from modules.llm_model import init_model, get_model
from modules.database import get_db, get_menu_info
from modules.divide_question_llm import divide_question, reset_divide_memory

from modules.test_one_llm import handle_screen_input, handle_user_input, reset_conversation_memory, get_session_state, scroll_action
from modules.ocr import run_ocr
from modules.get_question import get_question_from_image

from langchain.chains import ConversationChain
from fastapi.responses import JSONResponse
from modules.tts import get_tts, TTS_testReq
# from modules.stt import get_stt, STT_testReq, get_stt_from_file_obj, jwt_token
from modules.stt import get_stt, STT_testReq, get_stt_from_file_obj


from fastapi.middleware.cors import CORSMiddleware

from modules.dto import ChatRequest, ButtonRequest, QuestionRequest, ScrollRequest, TestMessageRequest, TestScrollRequest

# .env 불러오기
load_dotenv()

# FastAPI 인스턴스 추가
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


@app.on_event("startup")
async def startup():
    init_model()
    print("Model init")
    # print(get_model().invoke("하이")) #실제 모델 동작 하는지 테스트 (주석 처리 해놓지 않으면 리로드 할 때마다 계속 호출 해서 api 사용량 까먹음)
    # a = jwt_token

    db = get_db()
    print("DB init")
    # for i in db.query(MenuItem).all():
    #     print(i.id, i.parent_id, i.name, i.description, i.emoji, i.keywords)


@app.get("/")
def read_root():
    return {"message": f"Update"}


@app.post("/test/tts", deprecated=True)
async def test_tts(req: TTS_testReq):
    return get_tts(req.fileName, req.text)


@app.post("/test/stt", deprecated=True)
async def stt_test(req: STT_testReq):
    return get_stt(req.fileName)

@app.post("/test/ocr", deprecated=True)
async def ocr_test(file: UploadFile = File(...)):
    return await run_ocr(file)


@app.post("/test/get_button/chat", deprecated=True)
async def get_button_llm(req: ButtonRequest):
    return await handle_user_input(req)

@app.post("/test/reset-conversation", deprecated=True)
async def reset_button_llm(session_id: str = None):
    return await reset_conversation_memory(session_id)

@app.post("/test/divide_question/chat", deprecated=True)
async def divide_question_llm(req: QuestionRequest):
    return await handle_screen_input(req)

@app.post("/test/get-action")
async def test_get_action(req: TestMessageRequest):
    return await get_action_from_text(req.message)

@app.post("/test/get-action-scroll")
async def test_get_action_scroll(image_file: UploadFile = File(...), audio_message: str = Form(...), session_id: str = Form("default_session")):
    # 스크롤이 존재했을 경우, 스크롤 한 화면을 새롭게 받아서, 그 전 사용자의 답변을 기반으로 answer_text, answer_audio, action(click | scroll)) response

    # OCR 실행
    ocr_response = await run_ocr(image_file)
    ocr_data = ocr_response.body
    ocr_json = json.loads(ocr_data.decode())  # bytes → str → dict

    # llm 모델을 사용하여 질문 생성
    visible_buttons = [{"text": group["text"], "bbox": group["bbox"]}
                  for group in ocr_json.get("groups", [])]
    # sidebar_exists를 bool로 변환
    scrollbar_exists = ocr_json.get("sidebar_exists", False)
    scrollbar_exists_bool = bool(scrollbar_exists)
    print("Visible buttons:", visible_buttons, "scrollbar_exists:", scrollbar_exists)
    
    # "default_session" 대신 session_id 사용
    session = get_session_state(session_id)
    session["visible_buttons"] = visible_buttons
    session["side_bar_exists"] = scrollbar_exists_bool
    if scrollbar_exists_bool:
        # 세션에 스크롤바 좌표 업데이트 해줌
        x, y, w, h = scrollbar_exists
        session["side_bar_point"] = (x, y, w, h)
    
    audio_message_to_test_request = TestMessageRequest(message=audio_message, session_id=session_id)
    return await test_get_action(audio_message_to_test_request)



@app.post("/get-question")
async def get_question(file: UploadFile = File(...), session_id: str = Form("default_session")):
    # if True:
    #     await reset_conversation_memory()


    """
    스크린 샷을 보내면 사용자에게 질문할 음성과 선택지들을 보내주는 api

        {
            "follow_up_question": follow_up_question,
            "choices": options,
            "tts_file": s3_url,
            "sidebar": scrollbar_exists
        }
    
    세트 사이드 -> 세트 음료 넘어갈 때

        {
            "text": "세트 음료",
            "bbox": {
                "x": 1380,
                "y": 391,
                "width": 138,
                "height": 36
            },
            "action": "click"
        }
    """
    return await get_question_from_image(file, session_id)

@app.post("/get_action", summary="사용자 응답을 바탕으로 좌표를 받거나 추가 질문들 받을 수도, 좌표는 버튼과 스크롤바의 정보를 알려줌")
async def get_action(file: UploadFile = File(...), session_id: str = Form("default_session")):
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
    result = await get_action_from_audio(file, session_id)
    return result if result is not None else {"버튼 찾을수 없음"}

@app.post("/get-action-scroll")
async def get_action_scroll(image_file: UploadFile = File(...), audio_file: UploadFile = File(...), session_id= Form("default_session")):

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
    ocr_response = await run_ocr(image_file)
    ocr_data = ocr_response.body
    ocr_json = json.loads(ocr_data.decode())  # bytes → str → dict

    # llm 모델을 사용하여 질문 생성
    visible_buttons = [{"text": group["text"], "bbox": group["bbox"]}
                  for group in ocr_json.get("groups", [])]
    # sidebar_exists를 bool로 변환
    scrollbar_exists = ocr_json.get("sidebar_exists", False)
    scrollbar_exists_bool = bool(scrollbar_exists)
    print("Visible buttons:", visible_buttons, "scrollbar_exists:", scrollbar_exists)

    session = get_session_state(session_id)
    session["visible_buttons"] = visible_buttons
    session["side_bar_exists"] = scrollbar_exists_bool
    if scrollbar_exists_bool:
        # 세션에 스크롤바 좌표 업데이트 해줌
        x, y, w, h = scrollbar_exists
        session["side_bar_point"] = (x, y, w, h)

    return await get_action(audio_file, session_id=session_id)
