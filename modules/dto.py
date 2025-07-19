from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid

# 요청 바디 모델
class ChatRequest(BaseModel):
    message: str
    visible_buttons: list[str] = []

class ButtonRequest(BaseModel):
    message: str
    # visible_buttons: List[str]
    # question: str = ""
    # screen_type: str = ""
    session_id: str = Field(default_factory=lambda: f"session_{uuid.uuid4()}")

class QuestionRequest(BaseModel):
    visible_buttons: List[dict]
    side_bar_exists: bool = False
    session_id: str = Field(default_factory=lambda: f"session_{uuid.uuid4()}")

class ScrollRequest(BaseModel):
    visible_buttons: List[dict]
    side_bar_exists: bool = False
    message: str = ""
    session_id: str = Field(default_factory=lambda: f"session_{uuid.uuid4()}")

class TestMessageRequest(BaseModel):
    message: str
    session_id: str = Field(default_factory=lambda: f"session_{uuid.uuid4()}")

class TestScrollRequest(BaseModel):
    message: str
    session_id: str = Field(default_factory=lambda: f"session_{uuid.uuid4()}")