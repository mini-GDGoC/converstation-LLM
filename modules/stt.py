import json
import requests
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from tenacity import sleep

load_dotenv()

class STT_testReq(BaseModel):
    fileName: str


# 인증토큰 가져오기
def get_jwt_token():
    resp = requests.post(
        'https://openapi.vito.ai/v1/authenticate',
        data={'client_id': os.getenv('CLIENT_ID'),
          'client_secret': os.getenv('CLIENT_SECRET'),}
    )
    resp.raise_for_status()
    return resp.json()['access_token']


jwt_token = get_jwt_token()
if not jwt_token:
    raise ValueError("환경변수 'YOUR_JWT_TOKEN'이 설정되지 않았습니다.")

config = {
  "use_diarization": True,
  "diarization": {
    "spk_count": 1
  },
  "use_itn": False,
  "use_disfluency_filter": False,
  "use_profanity_filter": False,
  "use_paragraph_splitter": True,
  "paragraph_splitter": {
    "max": 50
  }
}

def get_stt(file_name) -> str:
    # 응답 실패하면 공스트링 보냄
    resp = requests.post(
        'https://openapi.vito.ai/v1/transcribe',
        headers={'Authorization': f'Bearer {jwt_token}'},
        data={'config': json.dumps(config)},
        files={'file': open(file_name, 'rb')}
    )
    print("전송",resp.json())

    transcribe_id = resp.json()['id']
    msg = ''

    while True:
        resp = requests.get(
            'https://openapi.vito.ai/v1/transcribe/' + transcribe_id,
            headers={'Authorization': 'bearer ' + jwt_token},
        )
        resp.raise_for_status()
        print(resp.json())


        if resp.json()['status'] != 'transcribing':
            if resp.json()['status'] != 'completed':
                return ""
            # {'id': 'R40mABXkSdq3mut7N0XI4Q', 'status': 'completed', 'results': {'utterances': [{'start_at': 635, 'duration': 4310, 'spk': 0, 'spk_type': 'NORMAL', 'msg': '아아아아 가나다라 마바사 가나다라 마바사.'}], 'verified': False}}
            msg = resp.json()['results']['utterances'][0]['msg']
            print(resp.json())
            break

        sleep(4)
    return msg


def get_stt_from_file_obj(file_obj, filename: str, mimetype: str) -> str:
    resp = requests.post(
        'https://openapi.vito.ai/v1/transcribe',
        headers={'Authorization': f'Bearer {jwt_token}'},
        data={'config': json.dumps(config)},
        files={'file': (filename, file_obj, mimetype)}, # 필요시 mimetype 변경
    )
    print("전송", resp.json())

    transcribe_id = resp.json()['id']
    msg = ''

    while True:
        resp = requests.get(
            'https://openapi.vito.ai/v1/transcribe/' + transcribe_id,
            headers={'Authorization': 'bearer ' + jwt_token},
        )
        resp.raise_for_status()
        print(resp.json())

        if resp.json()['status'] != 'transcribing':
            if resp.json()['status'] != 'completed':
                return ""
            msg = resp.json()['results']['utterances'][0]['msg']
            break

        sleep(4)
    return msg
