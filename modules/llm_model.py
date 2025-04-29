from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

model = None


def init_model():
    # gpt-4.1
    # gpt-4.1-mini
    # gpt-4.1-nano
    global model
    model = init_chat_model("gpt-4o-mini", model_provider="openai")


def get_model() -> BaseChatModel:
    global model
    return model