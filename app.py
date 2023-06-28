import chainlit as cl
import os
from embedchain import App
import pandas as pd

from collections import Counter

from utils import get_list_index_by_characters, generate_results_markdown

CONTEXT_CHUNKS = 3
CONTEXT_CHARACTER_LIMIT = 4000
DISPLAY_LONG_RESULTS = True

# QUESTION-ANSWERING PROMPT (REPLACEMENT FOR DEFAULT FROM EMBEDCHAIN) - MUST INCLUDE {context} AND {input_query}
PROMPT = """Use the following pieces of context to answer the query at the end. If you don't know, just say that you don't know, don't try to make up an answer.
    {context}
Query: {input_query}
Now, provide an accurate and conservative answer drawn only from information in the context material. Where necessary, include quotes from the context material.
    """

# bot is global variable - if there were more than one user session, they would have access to the same database
bot = App()


@cl.on_chat_start
async def start():
    welcome_message = f"""
This simple chatbot demonstrates the basic pattern of question-answering chatbots which use retrieval-augmented generation. It uses [embedchain](https://github.com/embedchain/embedchain/tree/main), a wrapper around [LangChain](https://github.com/hwchase17/langchain) and [chromadb](https://github.com/chroma-core/chroma). Embedchain handles loading data by URL or entering text, splitting it into chunks, generating [embeddings](https://platform.openai.com/docs/guides/embeddings), and saving these to a local vector database. When a user query is submitted, the system retrieves chunks of context information, formats a prompt, submits that prompt to gpt-3.5-turbo and returns the answer.

The prompt will use a maximum of {CONTEXT_CHUNKS} chunks of context information or {CONTEXT_CHARACTER_LIMIT} characters, whichever is less.

No user message history is submitted when asking questions, so be sure to include the entire context for each question in your message.
    """

    welcome_message = cl.Message(content=welcome_message, author="Application")
    await welcome_message.send()
    
    data_message = get_data_message()
    cl.user_session.set("data_message", data_message)
    await data_message.send()


@cl.on_message
async def main(message: str):

    raw_result = bot.collection.query(query_texts=[message,],n_results=CONTEXT_CHUNKS*2)
    try:
        context = raw_result["documents"][0][0]
    except IndexError:
        await cl.Message(content="No results found. Try rephrasing your question to include more detail.").send()
        return
        
    if DISPLAY_LONG_RESULTS:
        results_markdown = generate_results_markdown(raw_result)
        elements = [
            cl.Text(content=results_markdown, name="Results", display="inline"),
        ]
        await cl.Message(content="The database returned these results, in decreasing order of similarity to your question.", elements=elements, author="Database").send()

    context_string = get_context_string(raw_result)
    context_message = f"""
**Chunk(s) retrieved:**
{context_string}
    """
    elements = [
        cl.Text(content=context_message, name="Context", display="inline"),
    ]

    prompt = PROMPT.format(input_query=message, context=context_string)
    await cl.Message(content=prompt, author="System", indent=True).send()
    answer = bot.get_answer_from_llm(prompt)
    
    await cl.Message(content=answer, elements=elements).send()


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
        await cl.Message(content=f"Load not successful, error: {e}").send()


@cl.action_callback("delete_by_url")
async def delete_by_url_button(action):
    response = await cl.AskUserMessage(content="What url do you want to delete? WARNING: This is irreversible!", timeout=10000, author="Application").send()
    url = response["content"]
    success_message = delete_by_url(url)
    await cl.Message(content=success_message, author="Application").send()
    await update_data_message()


async def update_data_message():
    old_data_message = cl.user_session.get("data_message")
    await old_data_message.remove()
    data_message = get_data_message()
    cl.user_session.set("data_message", data_message)
    await data_message.send()


def delete_by_url(url):
    try:
        data = bot.collection.get(where={"url": url})
        ids = data["ids"]
        if ids:
            bot.collection.delete(ids=data["ids"])
            return f"Deleted {url}"
        else:
            return "No matching results found"
    except Exception as e: 
        return f"Exception {e}"


def get_context_string(raw_result):
    character_limit_index = get_list_index_by_characters(raw_result["documents"][0], CONTEXT_CHARACTER_LIMIT)
    print(f"Character limit index: {character_limit_index}")

    if character_limit_index < CONTEXT_CHUNKS:
        index = character_limit_index
    else:
        index = CONTEXT_CHUNKS
    print(f"Index: {index}")
    
    context_string = ""
    for i in range(index):
        chunk = f"...{raw_result['documents'][0][i]}...\n"
        context_string += chunk

    return context_string


def get_data_message():
    elements = [get_url_table_element()]
    
    actions = [
        cl.Action(name="load_content", value="youtube_video", label="Youtube Video URL"),
        cl.Action(name="load_content", value="pdf_file", label="PDF File URL"),
        cl.Action(name="load_content", value="web_page", label="Webpage URL"),
        cl.Action(name="load_content", value="text", label="Text"),
        cl.Action(name="delete_by_url", value="delete", label="Delete by URL")
    ]

    return cl.Message(content="Ask your question, or use these buttons to enter data to be embedded, or delete existing data.", actions=actions, elements=elements, author="Application") 


def get_url_table_element():
    data = bot.collection.get(
        include=["metadatas"]
    )
    metadata = data["metadatas"]

    url_counts = Counter(item['url'] for item in metadata)
    url_counts_list = [{'url': url, 'count': count} for url, count in url_counts.items()]
    df = pd.DataFrame(url_counts_list)
    url_table = df.to_markdown(index=False)
    if url_table:
        content=url_table
    else:
        content="No data found"
    return cl.Text(content=content, name="Existing URLs and chunks", display="inline")



