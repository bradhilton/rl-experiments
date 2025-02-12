from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel
from typing import Awaitable, Callable

Grader = Callable[[Choice], float | Awaitable[float]]


class Task(BaseModel):
    messages: list[ChatCompletionMessageParam]
    grader: Grader
