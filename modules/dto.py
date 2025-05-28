from pydantic import BaseModel


# 요청 바디 모델
class ChatRequest(BaseModel):
    message: str
    visible_buttons: list[str] = []

class ButtonRequest(BaseModel):
    message: str
    # question: str
    # screen_type: str
    # visible_buttons: list[str] = []

class QuestionRequest(BaseModel):
    visible_buttons: list[str] = []