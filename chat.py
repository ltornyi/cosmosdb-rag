from dotenv import dotenv_values
from openai import AzureOpenAI
from azure.cosmos import CosmosClient, ContainerProxy

DIMENSIONS = 512
DATABASE = "playground"
CONTAINER = "movies"

def get_setup():
    config = dotenv_values(".env")
    openai_client = AzureOpenAI(api_key=config['openai_key'],
                                azure_endpoint=config['openai_endpoint'],
                                api_version=config['openai_api_version'])
    cosmos_client = CosmosClient(url=config['cosmos_uri'], credential=config['cosmos_readonly_key'])
    db = cosmos_client.get_database_client(DATABASE)
    container = db.get_container_client(CONTAINER)
    return config,openai_client,container

def generate_embeddings(client: AzureOpenAI, text, deployment):
    response = client.embeddings.create(input=text, model=deployment, dimensions=DIMENSIONS)
    embeddings =response.model_dump()
    return embeddings['data'][0]['embedding']

def vector_search(openai_client, deployment, container, query, num_results=5):
    query_embedding = generate_embeddings(openai_client, query, deployment)
    results = container.query_items(
            query='SELECT TOP @num_results c.title, c.overview, VectorDistance(c.overview_embeddings,@embedding) AS SimilarityScore  FROM c ORDER BY VectorDistance(c.overview_embeddings,@embedding)',
            parameters=[
                {"name": "@embedding", "value": query_embedding}, 
                {"name": "@num_results", "value": num_results} 
            ],
            enable_cross_partition_query=True)
    return results

def build_messages(query, results):
    movies = "\n".join([f"{m['title']}: {m['overview']}" for m in results])
    return [
        {"role":"system", "content": "You are a helpful and knowledgeable assistant specialising in movie recommendations. Your goal is to recommend movies based on a user's preferences, ensuring the suggestions are relevant, diverse, and engaging. Use only the vector search results to inform your answers, grounding them in the most relevant retrieved content."},
        {"role":"assistant", "content": f"The user's query has been matched with the following top 5 movies based on plot similarity:{movies}"},
        {"role":"user", "content":query}
    ]

def question_loop(openai_client: AzureOpenAI, config, cosmos_container: ContainerProxy):
    print('Type a question or QUIT for exit')
    done = False
    while not done:
        query = input(">")
        done = query == "QUIT"
        if not done:
            results = list(vector_search(
                openai_client=openai_client,
                deployment=config['openai_embeddings_deployment'],
                container=cosmos_container,
                query=query
            ))
            messages = build_messages(query, results)
            response = openai_client.chat.completions.create(
                model=config['openai_completions_deployment'],
                messages=messages,
                temperature=0.2
            ).model_dump()
            msg0 = response['choices'][0]['message']
            usage = response['usage']
            print(msg0['content'])
            print("***********")
            print(f"total tokens: {usage['total_tokens']}")
            for m in results:
                print(f"Title: {m['title']}, score:{m['SimilarityScore']}")
            print("***********")


def main():
    config,openai_client,cosmos_container = get_setup()
    
    question_loop(openai_client=openai_client, config=config, cosmos_container=cosmos_container)


if __name__ == "__main__":
    main()