from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np


from PIL import Image
import io
import time
import json


import matplotlib.pyplot as plt
from dotenv import load_dotenv

from modules.get_action import get_action_from_audio
from modules.llm_model import init_model, get_model
from modules.database import get_db, get_menu_info
from modules.models import MenuItem
from modules.get_button_llm import get_button, reset_button_memory
from modules.divide_question_llm import divide_question, reset_divide_memory
from modules.test_one_llm import handle_screen_input, handle_user_input, reset_conversation_memory, get_session_state
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

from modules.dto import ChatRequest, ButtonRequest, QuestionRequest

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