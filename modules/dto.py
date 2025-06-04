from pydantic import BaseModel
from typing import List, Dict, Any, Optional

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
    visible_buttons: List[Dict[str, Any]] = []
    side_bar_exists: Optional[bool] = False

class ScrollRequest(BaseModel):
    visible_buttons: List[Dict[str, Any]] = []
    side_bar_exists: Optional[bool] = False
    message: str

class TestMessageRequest(BaseModel):
    message: str

class TestScrollRequest(BaseModel):
    audio_message: str