from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np


from PIL import Image
import io
import time
# from paddleocr import PaddleOCR

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from modules.llm_model import init_model, get_model
from modules.database import get_db, get_menu_info
from modules.models import MenuItem
from modules.get_button_llm import get_button, reset_button_memory
from modules.divide_question_llm import divide_question, reset_divide_memory
from modules.test_one_llm import handle_screen_input, handle_user_input, reset_conversation_memory
from modules.ocr import ocr

import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from fastapi.responses import JSONResponse
from langchain_core.messages import BaseMessage
from langchain.prompts import PromptTemplate

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

# easyocr
@app.post("/ocr-test")
async def ocr_test(file: UploadFile = File(...)):
    return await ocr(file)


@app.post("/get_button/chat") 
async def get_button_llm(req: ButtonRequest):
    return await handle_user_input(req)

@app.post("/reset-conversation")
async def reset_button_llm():
    return await reset_conversation_memory()

@app.post("/divide_question/chat") 
async def divide_question_llm(req: QuestionRequest):
    return await handle_screen_input(req)

