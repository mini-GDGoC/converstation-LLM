import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

text = ""
output_path = "output_speech.mp3"

# TTS 요청
response = openai.audio.speech.create(
    model="tts-1",
    voice="nova",  # or alloy, echo, fable, onyx, shimmer
    input=text
)
# 저장
with open(output_path, "wb") as f:
    f.write(response.content)

# 자동 재생 (플랫폼에 따라 명령 다름)
# os.system('start ' + output_path)       # Windows
os.system('afplay ' + output_path)    # MacOS
# os.system('mpg123 ' + output_path)    # Linux (mpg123 필요)
