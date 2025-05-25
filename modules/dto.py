from pydantic import BaseModel


# 요청 바디 모델
class ChatRequest(BaseModel):
    message: str
    visible_buttons: list[str] = []