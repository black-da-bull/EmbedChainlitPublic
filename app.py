import os

import chainlit as cl
import openai
from embedchain import App

from database import get_results_list_index, get_url_table, get_results, generate_results_markdown
from openai_api import Conversation


# ------------- SETTINGS --------------- #
# API KEY (imported from secrets)
openai.api_key = os.environ['OPENAI_API_KEY']

# MODEL SETTINGS
model_name = "gpt-3.5-turbo"
settings = {
    "temperature": .7,
    "max_tokens": 1000,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

# CONTEXT PARAMETERS
CONTEXT_CHUNKS = 4
CONTEXT_CHARACTER_LIMIT = 6000
DISPLAY_LONG_RESULT = True
# `NUMBER_RESULTS` must be greater than `CONTEXT_CHUNKS`
NUMBER_RESULTS = CONTEXT_CHUNKS * 2

# QUESTION-ANSWERING PROMPT (REPLACEMENT FOR DEFAULT FROM EMBEDCHAIN) - MUST INCLUDE {context} AND {input_query}
SYSTEM_PROMPT = "You are a helpful assistant"
PROMPT = """Use the following pieces of context to answer the query at the end. If you don't know, just say that you don't know, don't try to make up an answer.
    {context}
Query: {input_query}
Now, provide any correct information that can be concluded from the context information and that is responsive to the user's query. Do not provide information from general knowledge, **only** use the context. Include **direct quotes** from the context supporting any conclusions. Do not cite section numbers. 
    """


# DETAILED SETTINGS
PRINT_DEBUG = False
INCLUDE_SYSTEM_REMINDER = False
SYSTEM_REMINDER = "Provide your answer, relying solely on the context information and indicating where information is missing."


# ----------------- GLOBAL VARIABLES ----------------- #
# `bot` is a global variable - if there were more than one user session, they would have access to the same database
bot = App()


# -------------------- ON START ---------------------- #
@cl.on_chat_start
async def on_start():
    system = SYSTEM_PROMPT

    short_conversation = Conversation()
    context_conversation = Conversation()

    short_conversation.add_message("system", system)
    context_conversation.add_message("system", system)

    set_conversations(short_conversation, context_conversation)

    welcome_message = get_welcome_message()
    await welcome_message.send()

    data_message = get_data_message()
    cl.user_session.set("data_message", data_message)
    await data_message.send()


# ------------------- MESSAGE FUNCTIONS ---------------#
@cl.on_message
async def main(message):
    short_conversation, context_conversation = get_conversations()
    
    formatted_prompt, raw_result, context_string = process_message(message)
    short_conversation.add_message(role="user", content=message)
    context_conversation.add_message(role="user", content=formatted_prompt)

    await display_results(raw_result)

    await cl.Message(content=formatted_prompt, indent=True).send()
    
    messages = generate_messages() 
    messages.append({"role": "user", "content": formatted_prompt})
    messages = await handle_system_reminder(messages)

    await create_and_stream_response(context_string, messages)
    
    handle_debugging()


async def create_and_stream_response(context_string, messages):
    elements = [
        cl.Text(content=context_string,
                name="Context String",
                display="inline")
    ]

    msg = cl.Message(content="", elements=elements)

    response = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        stream=True,
        **settings,
    )

    try:
        for resp in response:
            token = resp.choices[0]["delta"].get("content", "")
            await msg.stream_token(token)
    
        short_conversation, context_conversation = get_conversations()
        context_conversation.add_message("assistant", msg.content)
        short_conversation.add_message("assistant", msg.content)
    
        await msg.send()
    except Exception as e:
        await cl.Message(content=f"Error {e}.\n Try again.").send()
        return


async def display_results(raw_result):
    if DISPLAY_LONG_RESULT:
        result_message = get_result_message(raw_result)
        await result_message.send()


async def handle_system_reminder(messages):
    if INCLUDE_SYSTEM_REMINDER:
        system_reminder = {"role": "system", "content": SYSTEM_REMINDER}
        await cl.Message(content=system_reminder["content"],
                         author="System",
                         indent=True).send()
        messages.append(system_reminder)

    return messages


async def update_data_message():
    old_data_message = cl.user_session.get("data_message")
    await old_data_message.remove()
    data_message = get_data_message()
    cl.user_session.set("data_message", data_message)
    await data_message.send()


# ---------------------- ACTION-CALLBACKS --------------------- #
# LOAD
@cl.action_callback("load_content")
async def load_content_button(action):
    content_type = action.value
    prompt = "Paste the text you want to load to the database." if content_type == "text" else f"Provide your {action.label}"

    response = await cl.AskUserMessage(content=prompt, timeout=10000).send()
    content = response["content"]

    try:
        bot.add(content_type, content)
        await update_data_message()
        success_message = "Content successfully loaded. You may proceed to ask a question."
        await cl.Message(content=success_message, author="Database").send()
    except Exception as e:
        await cl.Message(
            content=f"Load not successful, error: {e}",
            author="Database",
        ).send()
        await update_data_message()


# DELETE BY URL
@cl.action_callback("delete_by_url")
async def delete_by_url_button(action):
    response = await cl.AskUserMessage(
        content=
        "What url do you want to delete? WARNING: This is irreversible!",
        timeout=10000,
        author="Application").send()
    url = response["content"].strip()
    success_message = delete_by_url(url)
    await cl.Message(content=success_message, author="Application").send()
    await update_data_message()


def delete_by_url(url):
    try:
        data = bot.collection.get(
            where={"url": url}
        )
        print(data["ids"])
        if data["ids"]:
            bot.collection.delete(ids=data["ids"])
            return "Deletion successful"
        else:
            return "Deletion unsuccessful, no matches."
    except Exception as e:
        return f"Deletion unsuccessful, error: {e}"


# ------------------- FORMATTING PROMPT ------------------- #
def get_context_string(raw_result):
    index = get_results_list_index(
        raw_result, 
        CONTEXT_CHARACTER_LIMIT,
        CONTEXT_CHUNKS
    )
    
    context_string = ""
    
    for i in range(index):
        metadata = raw_result['metadatas'][0][i]
        metadata_string = ""
        for key in metadata.keys():
            metadata_string += f"{key}: {metadata[key]}\n"
        chunk = f"{metadata_string}...{raw_result['documents'][0][i]}...\n"
        context_string += chunk

    return context_string


def get_formatted_prompt(context_string, message):
    prompt = PROMPT.format(input_query=message, context=context_string)
    return prompt


# ------------------ MESSAGE FORMATTING ------------------ #
# WELCOME MESSAGE
def get_welcome_message():
    """Formats welcome message and returns cl.Message object"""
    welcome_message = f"""
This simple chatbot demonstrates the basic pattern of question-answering chatbots which uses retrieval-augmented generation, using the OpenAI API. It uses open-source Python packages, particularly [EmbedChain](https://github.com/embedchain/embedchain), which is a wrapper around [chromadb](https://github.com/chroma-core/chroma) and [LangChain](https://github.com/hwchase17/langchain), and [Chainlit](https://github.com/Chainlit/chainlit) for the user interface.

Current settings - the context included in the prompt is the lesser of:
- `CONTEXT_CHUNKS`: {CONTEXT_CHUNKS} 
- `CONTEXT_CHARACTER_LIMIT`: {CONTEXT_CHARACTER_LIMIT}

User message history is included (without the context injected in previous prompts). 

See the Readme for more information.
    """
    elements = [
        cl.Text(content=PROMPT, name="Prompt", display="inline"),
    ]

    welcome_message = cl.Message(content=welcome_message,
                                 author="Application",
                                 elements=elements)
    return welcome_message


# DATA MESSAGE
def get_data_message():
    url_table = get_url_table(bot)
    elements = [
        cl.Text(content=url_table,
                name="Existing URLs and chunks",
                display="inline"),
    ]

    actions = [
        cl.Action(name="load_content",
                  value="youtube_video",
                  label="Youtube Video URL"),
        cl.Action(name="load_content", value="pdf_file", label="PDF File URL"),
        cl.Action(name="load_content", value="web_page", label="Webpage URL"),
        cl.Action(name="load_content", value="text", label="Text"),
        cl.Action(name="delete_by_url", value="delete", label="Delete by URL"),
    ]

    return cl.Message(
        content=
        "Ask your question, or use these buttons to enter data to be embedded, or delete existing data.",
        actions=actions,
        elements=elements,
        author="Application")


# RESULTS MESSAGE
def get_result_message(raw_result):
    result_table = generate_results_markdown(raw_result)
    elements = [
        cl.Text(content=result_table, name="Result Table", display="inline")
    ]
    result_message = cl.Message(
        content="Here are the results, in descending order of similarity.",
        author="Database",
        elements=elements,
        indent=True,
    )
    return result_message


# ----------------------- RETRIEVAL --------------------- #
def process_message(message):
    raw_result = get_results(
        bot=bot, 
        message=message, 
        n_results=NUMBER_RESULTS
    )
    context_string = get_context_string(raw_result)
    formatted_prompt = get_formatted_prompt(context_string, message)

    return formatted_prompt, raw_result, context_string


# ---------------------------- MESSAGE HISTORY --------------------------------- #
def generate_messages():
    short_conversation, _ = get_conversations()
    messages = short_conversation.messages.copy()
    return messages


def get_conversations():
    short_conversation = cl.user_session.get("short_conversation")
    context_conversation = cl.user_session.get("context_conversation")
    return short_conversation, context_conversation


def set_conversations(short_conversation, context_conversation):
    cl.user_session.set("short_conversation", short_conversation)
    cl.user_session.set("context_conversation", context_conversation)
    

# -------------------- UTILITY FUNCTIONS ------------------- #
def handle_debugging():
    short_conversation, context_conversation = get_conversations()
    if PRINT_DEBUG:
        print("Short conversation:")
        short_conversation.display_conversation()
        print("")
        print("Context conversation")
        context_conversation.display_conversation()
        print("")

