from dotenv import dotenv_values
from openai import AzureOpenAI
from azure.cosmos import CosmosClient, ContainerProxy

DIMENSIONS = 512
DATABASE = "playground"
CONTAINER = "movies"

def generate_embeddings(client: AzureOpenAI, text, deployment):
    response = client.embeddings.create(input=text, model=deployment, dimensions=DIMENSIONS)
    embeddings =response.model_dump()
    return embeddings['data'][0]['embedding']

def get_create_database(client:CosmosClient, dbname):
    db= client.create_database_if_not_exists(id=dbname)
    return db

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

def question_loop(openai_client: AzureOpenAI, config, cosmos_container: ContainerProxy):
    print('Type a question or QUIT for exit')
    done = False
    while not done:
        query = input(">")
        done = query == "QUIT"
        if not done:
            results = vector_search(
                openai_client=openai_client,
                deployment=config['openai_embeddings_deployment'],
                container=cosmos_container,
                query=query
            )
            for m in results:
                print(f"Title: {m['title']}, score:{m['SimilarityScore']}")

def main():
    config = dotenv_values(".env")
    openai_client = AzureOpenAI(api_key=config['openai_key'],
                                azure_endpoint=config['openai_endpoint'],
                                api_version=config['openai_api_version'])
    cosmos_client = CosmosClient(url=config['cosmos_uri'], credential=config['cosmos_readonly_key'])
    db = get_create_database(cosmos_client, DATABASE)
    container = db.get_container_client(CONTAINER)
    
    question_loop(openai_client=openai_client, config=config, cosmos_container=container)


if __name__ == "__main__":
    main()