import json
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from modules.dto import QuestionRequest
from modules.ocr import run_ocr
from modules.test_one_llm import handle_screen_input
from modules.tts import get_tts

async def get_question_from_image(file: UploadFile):
    # OCR 실행
    ocr_response = await run_ocr(file)
    ocr_data = ocr_response.body
    ocr_json = json.loads(ocr_data.decode())  # bytes → str → dict

    # llm 모델을 사용하여 질문 생성
    visible_buttons = [{"text": group["text"], "bbox": group["bbox"]}
                      for group in ocr_json.get("groups", [])]
    print("Visible buttons:", visible_buttons)
    req = QuestionRequest(visible_buttons=visible_buttons)
    llm_response = await handle_screen_input(req)
    print("LLM Response:", llm_response)
    # llm_response.body는 bytes이므로 디코딩 후 파싱
    llm_json = json.loads(llm_response.body.decode())
    print("llm_json:", llm_json)
    response_data = llm_json.get("response", llm_json)
    follow_up_question = response_data.get("follow_up_question", "")
    options = response_data.get("choices", [])

    tts_file = None
    if follow_up_question:
        tts_file = get_tts("question", follow_up_question)

    return JSONResponse(content={
        "follow_up_question": follow_up_question,
        "choices": options,
        "tts_file": tts_file
    })