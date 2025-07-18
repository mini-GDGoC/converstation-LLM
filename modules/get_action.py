from fastapi import UploadFile, File

from modules.dto import ButtonRequest
from modules.stt import get_stt_from_file_obj
from modules.test_one_llm import handle_user_input, get_session_state
import json

from modules.tts import get_tts, get_tts_file_obj
import modules.s3 as s3



async def get_action_from_audio(file: UploadFile = File(...), session_id: str = "default_session"):
    user_answer = get_stt_from_file_obj(file.file, file.filename, file.content_type)
    print(user_answer, "유저응답 변환")
    result = await handle_user_input(ButtonRequest(
        message=user_answer,
        session_id=session_id,
    ))
    print(json.loads(result.body), 'llm 응답 반환')
    result = json.loads(result.body)["response"]
    session = get_session_state(session_id=session_id)

    """
    {
        "matched_button": "일치하는 버튼 이름 또는 null",
        "follow_up_question": "어르신께 드릴 질문 (없으면 빈 문자열)",
        "choices": ["선택지1", "선택지2", "(없으면 빈 배열)"] ,
        "action": "click | scroll | ask"
    }
    """

    if result["action"] == "ask":
        # 재 질문인 경우
        follow_up_question = result["follow_up_question"]
        options = result["choices"]

        print(follow_up_question, "팔로우업 퀘스쳔")
        follow_up_question_audio = get_tts("follow_up_question", follow_up_question)
        obj_name = "follow_up_question.mp3"
        obj_url = s3.upload_obj(
            obj_name, follow_up_question_audio
        )
        print(obj_url, 's3url')
        return {
            "follow_up_question_url": obj_url,
            "follow_up_question": follow_up_question,
            "choices": options,
            "user_answer": user_answer,
        }
    elif result["action"] == "scroll":
        # 여기서 스크롤 버튼을 어떻게 찾음?
        x, y, w, h = session["side_bar_point"]

        return {
            "action": "scroll",
            "text": '사이드바',
            "bbox":{
                "x": x,
                "y": y,
                "width": w,
                "height": h,
            }
        }
    else:
        # 매치 되는 버튼이 있음
        print("매치되는 버튼이 있음")
        button = result["matched_button"]
        # 버튼 이름으로 버튼을 찾음

        print("버튼이름: ", button)
        print("비저블버튼스", session["visible_buttons"])
        first_match = next(
            (d for d in session["visible_buttons"] if d.get("text") == button),
            None
        )
        if first_match == None:
            return None

        # 아직 이 패스로 매칭이 안되어서 테스트 불가
        first_match["action"] = "click"
        return first_match



async def get_action_from_text(user_message: str):
    result = await handle_user_input(ButtonRequest(
        message=user_message,
    ))
    print(json.loads(result.body), 'llm 응답 반환')
    result = json.loads(result.body)["response"]
    session = get_session_state("default_session")

    print("scrollbar_exists:", session["side_bar_exists"])
    if result["action"] == "ask":
        # 매치 되는 버튼이 없음
        follow_up_question = result["follow_up_question"]
        options = result["choices"]

        print(follow_up_question, "팔로우업 퀘스쳔")
        follow_up_question_audio = get_tts("follow_up_question", follow_up_question)
        obj_name = "follow_up_question.mp3"
        obj_url = s3.upload_obj(
            obj_name, follow_up_question_audio
        )
        print(obj_url, 's3url')
        return {
            "follow_up_question_url": obj_url,
            "follow_up_question": follow_up_question,
            "choices": options,
            "user_answer": user_message,
        }
    elif result["action"] == "scroll":

        # 여기서 스크롤 버튼을 어떻게 찾음?
        x, y, w, h = session["side_bar_point"]


        return {
            "action": "scroll",
            "text": '사이드바',
            "bbox":{
                "x": x,
                "y": y,
                "width": w,
                "height": h,
            }
        }
    else:
        # 매치 되는 버튼이 있음
        print("매치되는 버튼이 있음")
        button = result["matched_button"]
        # 버튼 이름으로 버튼을 찾음

        print("버튼이름: ", button)
        first_match = next(
            (d for d in session["visible_buttons"] if d.get("text") == button),
            None
        )
        if first_match == None:
            return None

        # 아직 이 패스로 매칭이 안되어서 테스트 불가
        first_match["action"] = "click"
        return first_match