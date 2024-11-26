import json
from dotenv import dotenv_values
from openai import AzureOpenAI

DIMENSIONS = 512
MOVIEJSON = './data/parsed_movies.json'
MOVIEEMBEDDINGS = './data/movies_embeddings_5k.json'

def generate_embeddings(client: AzureOpenAI, text, deployment):
    response = client.embeddings.create(input=text, model=deployment, dimensions=DIMENSIONS)
    embeddings =response.model_dump()
    return embeddings['data'][0]['embedding']

def load_parsed_movies():
    with open(MOVIEJSON, 'r') as jsonfile:
        data = json.load(jsonfile)
    return data

def movies_w_embeddings(data, client:AzureOpenAI, deployment):
    result=[]
    for item in data:
        title_embeddings = generate_embeddings(client, text=item['title'], deployment=deployment)
        overview_embeddings = generate_embeddings(client, text=item['overview'], deployment=deployment)
        item['title_embeddings'] = title_embeddings
        item['overview_embeddings'] = overview_embeddings
        result.append(item)
    return result

def save_movie_embeddings(data):
    with open(MOVIEEMBEDDINGS,'w') as jsonfile:
        json.dump(data, jsonfile)

def main():
    config = dotenv_values(".env")
    openai_client = AzureOpenAI(api_key=config['openai_key'],
                              azure_endpoint=config['openai_endpoint'],
                              api_version=config['openai_api_version'])
    data = load_parsed_movies()
    data_w_embeddings = movies_w_embeddings(data, openai_client, config['openai_embeddings_deployment'])
    save_movie_embeddings(data_w_embeddings)


if __name__ == "__main__":
    main()