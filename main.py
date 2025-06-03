from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np



import json


import matplotlib.pyplot as plt
from dotenv import load_dotenv

from modules.get_action import get_action_from_audio
from modules.llm_model import init_model, get_model
from modules.database import get_db, get_menu_info
from modules.divide_question_llm import divide_question, reset_divide_memory
from modules.test_one_llm import handle_screen_input, handle_user_input, reset_conversation_memory, get_session_state, scroll_action
from modules.ocr import run_ocr
from modules.get_question import get_question_from_image
import modules.s3 as s3

import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from fastapi.responses import JSONResponse
from langchain_core.messages import BaseMessage
from langchain.prompts import PromptTemplate
from modules.tts import get_tts, TTS_testReq
from modules.stt import get_stt, STT_testReq, get_stt_from_file_obj

from modules.dto import ChatRequest, ButtonRequest, QuestionRequest, ScrollRequest

# .env 불러오기
load_dotenv()

# FastAPI 인스턴스
app = FastAPI()


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


@app.post("/test_tts")
async def test_tts(req: TTS_testReq):
    return get_tts(req.fileName, req.text)


@app.post("/stt-test")
async def stt_test(req: STT_testReq):
    return get_stt(req.fileName)

@app.post("/ocr-test")
async def ocr_test(file: UploadFile = File(...)):
    return await run_ocr(file)


@app.post("/get_button/chat") 
async def get_button_llm(req: ButtonRequest):
    return await handle_user_input(req)

@app.post("/reset-conversation")
async def reset_button_llm():
    return await reset_conversation_memory()

@app.post("/divide_question/chat") 
async def divide_question_llm(req: QuestionRequest):
    return await handle_screen_input(req)

@app.post("/get-question")
async def get_question(file: UploadFile = File(...)):
    return await get_question_from_image(file)

@app.post("/get_action")
async def get_action(file: UploadFile = File(...)):
    return await get_action_from_audio(file)

@app.post("/get-action-scroll")
async def get_action_scroll(file: UploadFile = File(...),
    message: str = Form(...)):
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
