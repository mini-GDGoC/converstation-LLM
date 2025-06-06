import json
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from modules.dto import QuestionRequest
from modules.ocr import run_ocr
from modules.test_one_llm import handle_screen_input, get_session_state
from modules.tts import get_tts_file_obj, get_tts
from modules.s3 import upload_obj

import re

# 오탈자 교정 사전
TYPO_CORRECTIONS = {
    "헤이컨버거": "베이컨버거",
    "쿠포교환권": "쿠폰교환권",
    "더불패티버거": "더블패티버거"
}

def clean_text(text: str) -> str:
    text = re.sub(r"[^\w\s가-힣0-9]", "", text)  # 특수문자 제거
    text = re.sub(r"([가-힣])(\d+)", r"\1 \2", text)  # "단어숫자" → "단어 숫자"
    for typo, correct in TYPO_CORRECTIONS.items():
        text = text.replace(typo, correct)
    return text.rstrip()

async def get_question_from_image(file: UploadFile):
    # OCR 실행
    ocr_response = await run_ocr(file)
    ocr_data = ocr_response.body
    ocr_json = json.loads(ocr_data.decode())  # bytes → str → dict

    # 버튼 추출
    visible_buttons = [
    {"text": clean_text(group["text"]), "bbox": group["bbox"]}
    for group in ocr_json.get("groups", [])]

    # 스크롤바 추출
    scrollbar_exists = ocr_json.get("sidebar_exists", False)
    scrollbar_exists_bool = bool(scrollbar_exists)
    if scrollbar_exists_bool:
        # 세션에 스크롤바 좌표 업데이트 해줌
        session = get_session_state("default_session")
        session["side_bar_point"] = scrollbar_exists


    print("Visible buttons:", visible_buttons)
    print("Scrollbar exists:", scrollbar_exists_bool)

    # LLM 요청
    req = QuestionRequest(visible_buttons=visible_buttons, side_bar_exists=scrollbar_exists_bool)
    llm_response = await handle_screen_input(req)
    print("LLM Response:", llm_response)
    # llm_response.body는 bytes이므로 디코딩 후 파싱
    llm_json = json.loads(llm_response.body.decode())
    print("llm_json:", llm_json)
    response_data = llm_json.get("response", llm_json)
    follow_up_question = response_data.get("follow_up_question", "")
    options = response_data.get("choices", [])

    tts_file_url = None
    if follow_up_question:
        tts_path = get_tts("question", follow_up_question)  # "./output/question.mp3"
        tts_file_url = upload_obj("question.mp3", tts_path)  # ← 경로 넘기기!

    if '세트 음료' in response_data.get("choices"):
        first_match = next(
            (d for d in session["visible_buttons"] if d.get("text") == "세트 음료"),
        )

        # 아직 이 패스로 매칭이 안되어서 테스트 불가
        first_match["action"] = "click"
        return first_match

    if response_data.get("matched_button"):
        matched_button = response_data["matched_button"]
        first_match = next(
            (d for d in session["visible_buttons"] if d.get("text") == matched_button),
        )
        first_match["action"] = "click"
        return first_match
            
    return JSONResponse(content={
        "follow_up_question": follow_up_question,
        "choices": options,
        "tts_file": tts_file_url,
        "sidebar": scrollbar_exists
    })