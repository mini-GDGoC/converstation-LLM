import os
from google.cloud import texttospeech
from pydantic import BaseModel


class TTS_testReq(BaseModel):
    fileName: str
    text: str


os.makedirs("output", exist_ok=True)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = './boreal-diode-459311-e1-f5e8d3dfa0c5.json'
client = texttospeech.TextToSpeechClient()
voice = texttospeech.VoiceSelectionParams(language_code='ko-KR', name='ko-KR-Chirp3-HD-Achernar')
audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)



def get_tts(fileName, text):
    synthesis_input = texttospeech.SynthesisInput(text=text)
    resp = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

    with open(f"./output/{fileName}.mp3", "wb") as out:
        out.write(resp.audio_content)
        print(f"Audio content written to output/{fileName}.mp3")
    print("wefwefwef")
    return f"./output/{fileName}.mp3"


def get_tts_file_obj(text):
    synthesis_input = texttospeech.SynthesisInput(text=text)
    resp = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    return resp.audio_content