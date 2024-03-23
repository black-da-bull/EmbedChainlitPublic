import os

import openai
import tiktoken
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored


openai.api_key = os.environ['OPENAI_API_KEY']

# Used only for token count presently
MODEL = "gpt-3.5-turbo"
# Used for retry
ATTEMPTS = 3


class Conversation:
    def __init__(self):
        self.messages = []

    def add_message(self, role, content):
        message = {"role": role, "content": content}
        self.messages.append(message)

    def display_conversation(self, detailed=True):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        for message in self.messages:
            print(
                colored(
                    f"{message['role']}: {message['content']}\n\n",
                    role_to_color[message["role"]],
                )
            )


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(ATTEMPTS))
def chat_completion_request(messages, functions=None, function_call="auto", model="gpt-3.5-turbo-0613", temperature=0):
    try:
        if functions is not None:
            response = openai.ChatCompletion.create(
                model=model,
                temperature=temperature,
                functions=functions,
                function_call=function_call,
                messages=messages
            )
        else:
            response = openai.ChatCompletion.create(
                model=model,
                temperature=temperature,
                messages=messages
            )
        # print(response)
        print(response["usage"])
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


def get_content_from_response(response):
    try:
        content = response.choices[0].message["content"]
        return content
    except AttributeError as e:
        print(f"Error in get_content_from_response: {e}")
        return None

# ---------------- TOKEN RELATED FUNCTIONS ----------------- #
def num_tokens_from_string(text):
    encoding = tiktoken.encoding_for_model(MODEL)
    num_tokens = len(encoding.encode(text))
    return num_tokens