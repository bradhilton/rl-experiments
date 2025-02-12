from dataclasses import dataclass
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import cast


from .tasks import TaskResult


@dataclass
class TokenizedResult:
    conversation: list
    chat_template: str
    chat: str
    tokens: list[str]
    token_ids: list[int]
    assistant_mask: list[int]
    token_logprobs: list[ChatCompletionTokenLogprob] | None


class TaskResultTokenizer:
    def __init__(
        self,
        pretrained_tokenizer_or_model_name_or_path: (
            PreTrainedTokenizer | PreTrainedTokenizerFast | str
        ),
    ) -> None:
        self.tokenizer = (
            AutoTokenizer.from_pretrained(pretrained_tokenizer_or_model_name_or_path)
            if isinstance(pretrained_tokenizer_or_model_name_or_path, str)
            else pretrained_tokenizer_or_model_name_or_path
        )

    def __call__(self, task_result: TaskResult) -> TokenizedResult:
        conversation: list = task_result.task.messages + [
            {
                "role": "assistant",
                "content": task_result.chat_completion.choices[0].message.content,
            }
        ]
        assert isinstance(self.tokenizer.chat_template, str)
        chat_template = (
            self.tokenizer.get_chat_template()
            # Remove template logic that strips reasoning content from the chat messages
            .replace(
                "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}",
                "",
            )
            # Add generation tags for assistant token masking
            .replace(
                "{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}",
                "{{'<｜Assistant｜>'}}{% generation %}{{ content }}{% endgeneration %}{{'<｜end▁of▁sentence｜>'}}",
            )
        )
        chat = cast(
            str,
            self.tokenizer.apply_chat_template(
                conversation, chat_template=chat_template, tokenize=False
            ),
        )
        tokenized_result = cast(
            dict[str, list[int]],
            self.tokenizer.apply_chat_template(
                conversation,
                chat_template=chat_template,
                return_dict=True,
                return_assistant_tokens_mask=True,
            ),
        )
        tokens = [
            self.tokenizer.decode(token_id)
            for token_id in tokenized_result["input_ids"]
        ]
        assistant_tokens = [
            token
            for token, mask in zip(tokens, tokenized_result["assistant_masks"])
            if mask
        ]
        if (
            task_result.chat_completion.choices[0].logprobs
            and task_result.chat_completion.choices[0].logprobs.content
        ):
            result_token_logprobs = task_result.chat_completion.choices[
                0
            ].logprobs.content.copy()
            if "".join(assistant_tokens) != "".join(
                token_logprob.token for token_logprob in result_token_logprobs
            ):
                print("Assistant tokens are not equal, skipping token logprobs")
                token_logprobs = None
            elif assistant_tokens == [
                token_logprob.token for token_logprob in result_token_logprobs
            ]:
                token_logprobs = result_token_logprobs
            else:
                completion = ""
                result_completion = ""
                token_logprobs = []
                try:
                    while True:
                        if completion == result_completion:
                            token = assistant_tokens.pop(0)
                            result_token_logprob = result_token_logprobs.pop(0)
                            result_token = result_token_logprob.token
                            if token == result_token:
                                token_logprobs.append(result_token_logprob)
                            else:
                                token_logprobs.append(
                                    ChatCompletionTokenLogprob(
                                        token=token,
                                        logprob=float("nan"),
                                        top_logprobs=[],
                                    )
                                )
                            completion += token
                            result_completion += result_token
                        elif len(completion) < len(result_completion):
                            token = assistant_tokens.pop(0)
                            token_logprobs.append(
                                ChatCompletionTokenLogprob(
                                    token=token,
                                    logprob=float("nan"),
                                    top_logprobs=[],
                                )
                            )
                            completion += token
                        elif len(completion) > len(result_completion):
                            result_completion += result_token_logprobs.pop(0).token
                        else:
                            print("Warning: Completions are not equal")
                            print(f"Completion: {completion}")
                            print(f"Result completion: {result_completion}")
                            token_logprobs = None
                            break
                except IndexError:
                    pass
        else:
            token_logprobs = None
        return TokenizedResult(
            conversation=conversation,
            chat_template=chat_template,
            chat=chat,
            tokens=[
                self.tokenizer.decode(token_id)
                for token_id in tokenized_result["input_ids"]
            ],
            token_ids=tokenized_result["input_ids"],
            assistant_mask=tokenized_result["assistant_masks"],
            token_logprobs=token_logprobs,
        )
