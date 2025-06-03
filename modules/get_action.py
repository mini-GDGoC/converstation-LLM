from fastapi import UploadFile, File

from modules.dto import ButtonRequest
from modules.stt import get_stt_from_file_obj
from modules.test_one_llm import handle_user_input, get_session_state
import json

from modules.tts import get_tts, get_tts_file_obj
import modules.s3 as s3



async def get_action_from_audio(file: UploadFile = File(...)):
    user_answer = get_stt_from_file_obj(file.file, file.filename, file.content_type)
    print(user_answer, "유저응답 변환")
    result = await handle_user_input(ButtonRequest(
        message=user_answer,
    ))
    print(json.loads(result.body), 'llm 응답 반환')
    result = json.loads(result.body)["response"]

    if result["matched_button"] is None:
        # 매치 되는 버튼이 없음
        follow_up_question = result["follow_up_question"]
        options = result["choices"]

        print(follow_up_question, "팔로우업 퀘스쳔")
        follow_up_question_audio = get_tts("follow_up_question", follow_up_question)
        obj_name = "follow_up_question.mp3"
        bucket_name = "songil-s3"
        obj_url = s3.upload_obj(
            bucket_name, obj_name, follow_up_question_audio
        )
        print(obj_url, 's3url')
        return {
            "follow_up_question_url": obj_url,
            "choices": options
        }
    else:
        # 매치 되는 버튼이 있음
        print("매치되는 버튼이 있음")
        button = result["matched_button"]
        # 버튼 이름으로 버튼을 찾음

        session = get_session_state("default_session")
        first_match = next(
            (d for d in session["visible_buttons"] if d.get("text") == button),
        )

        # 아직 이 패스로 매칭이 안되어서 테스트 불가
        return first_match