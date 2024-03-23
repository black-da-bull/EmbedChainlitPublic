from collections import Counter
import pandas as pd


def format_url_table(data):
    metadata = data["metadatas"]
    url_counts = Counter(item['url'] for item in metadata)
    url_counts_list = [{
        'url': url,
        'count': count
    } for url, count in url_counts.items()]
    df = pd.DataFrame(url_counts_list)
    url_table = df.to_markdown(index=False)
    return url_table


def generate_results_markdown(raw_result):
    restructured_results = restructure_results(raw_result)
    df = pd.DataFrame(restructured_results["results"])
    df = df.drop("id", axis=1)
    df['url'] = df['url'].apply(lambda url: '[Link]({})'.format(url))
    return df.to_markdown(index=False)


def get_results(bot, message, n_results):
    """
    Uses `bot` object (EmbedChain App), interfaces directly with chromadb to retrieve matching results.
    See https://docs.trychroma.com/usage-guide#querying-a-collection

    Returns `raw_result` if results found, or None. `raw_result` is a dictionary with embedded lists:
    {
        "ids": [["example"]],
        "documents': [["example"]],
        "embeddings": None,
        "metadatas": [[{"url": "example"}]]
    }
    """
    raw_result = bot.collection.query(query_texts=[message], n_results=n_results)

    try:
        _ = raw_result["documents"][0][0]
    except IndexError:
        return None
    
    return raw_result


def get_results_list_index(raw_result, context_character_limit, context_chunks):
    character_limit_index = get_list_index_by_characters(raw_result["documents"][0], context_character_limit)

    if character_limit_index < context_chunks:
        index = character_limit_index
    else:
        index = context_chunks
    return index


def get_url_table(bot):
    data = bot.collection.get(include=["metadatas"])
    url_table = format_url_table(data)

    if url_table:
        content = url_table
    else:
        content = "No data found"

    return content


def get_list_index_by_characters(strings, length_limit):
    total_length = 0
    n = 0

    for string in strings:
        if total_length + len(string) <= length_limit:
            total_length += len(string)
            n += 1
        else:
            break

    return n

    
def restructure_results(data):
    results = []
    for i in range(len(data['ids'][0])):
        result = {}
        result['id'] = data['ids'][0][i]
        result['document'] = data['documents'][0][i]
        result['url'] = data['metadatas'][0][i]["url"]
        result['distance'] = data['distances'][0][i]
        results.append(result)
    return {'results': results}


