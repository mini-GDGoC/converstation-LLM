import numpy as np
import easyocr
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import cv2
import io

# ocr reader
reader = easyocr.Reader(['ko', 'en'])

# # OCR
# # PaddleOCR 초기화 - 여러 언어 지원
# ocr = None
# try:
#     ocr = PaddleOCR(use_angle_cls=True, lang='korean')
#     print("✅ PaddleOCR korean_english 모델 초기화 성공")
# except Exception as e:
#     print(f"❌ PaddleOCR 초기화 실패: {e}")
#     raise RuntimeError("OCR 초기화 실패")


# @app.post("/get-position")
# async def get_position(file: UploadFile = File(...)):
    # global ocr

    # image_bytes = await file.read()
    # image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # image_np = np.array(image)

    # ocr_result = ocr.ocr(image_np)

    # buttons = []

    # if isinstance(ocr_result, list) and len(ocr_result) > 0 and isinstance(ocr_result[0], dict):
    #     result_dict = ocr_result[0]

    #     polys = result_dict.get("dt_polys", []) or result_dict.get("rec_boxes", [])
    #     texts = result_dict.get("rec_texts", [])
    #     scores = result_dict.get("rec_scores", [])

    #     for i, (box, text, score) in enumerate(zip(polys, texts, scores)):
    #         if score > 0.5:
    #             try:
    #                 if hasattr(box, 'tolist'):
    #                     box = box.tolist()
    #                 x_coords = [int(p[0]) for p in box]
    #                 y_coords = [int(p[1]) for p in box]
    #                 x_min, x_max = min(x_coords), max(x_coords)
    #                 y_min, y_max = min(y_coords), max(y_coords)

    #                 buttons.append({
    #                     "text": text,
    #                     "confidence": float(score),
    #                     "bbox": {
    #                         "x": x_min,
    #                         "y": y_min,
    #                         "width": x_max - x_min,
    #                         "height": y_max - y_min
    #                     }
    #                 })
    #             except Exception as e:
    #                 print(f"[오류] box 처리 실패: {e}")
    #                 continue

    # return JSONResponse(content={
    #     "buttons": buttons,
    #     "count": len(buttons)
    # })

async def ocr(file: UploadFile = File(...)):
    # image load
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)

    # opencv용 bgr로 변환
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 3. CLAHE 객체 생성 (clipLimit 높일수록 대비 강해짐)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)

    # 밝은 영역만 마스킹 (threshold 적용)
    _, bright_mask = cv2.threshold(contrast_enhanced, 100, 255, cv2.THRESH_BINARY)
    bright_only = cv2.bitwise_and(image_bgr, image_bgr, mask=bright_mask)
    bright_rgb = cv2.cvtColor(bright_only, cv2.COLOR_BGR2RGB)

    # detect text and position
    results = reader.readtext(bright_rgb)

    buttons = []
    for (bbox, text, prob) in results:
        if prob > 0.5:  # 신뢰도 기준
            (tl, tr, br, bl) = bbox
            x_min = int(min(tl[0], bl[0]))
            y_min = int(min(tl[1], tr[1]))
            x_max = int(max(tr[0], br[0]))
            y_max = int(max(bl[1], br[1]))
            buttons.append({
                "text": text,
                "bbox": {
                    "x": x_min,
                    "y": y_min,
                    "width": x_max - x_min,
                    "height": y_max - y_min
                }
            })

    visible_button_texts = [b['text'] for b in buttons]

    return JSONResponse(content={
        "buttons": buttons,
    })