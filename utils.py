

def generate_results_markdown(results):
    restructured_results = restructure_results(results)
    df = pd.DataFrame(restructured_results["results"])
    df = df.drop("id", axis=1)
    return df.to_markdown(index=False)


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