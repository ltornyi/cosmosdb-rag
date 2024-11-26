import json
from azure.cosmos import CosmosClient, PartitionKey, DatabaseProxy, ContainerProxy
from dotenv import dotenv_values

DIMENSIONS = 512
MOVIEEMBEDDINGS = './data/movies_embeddings_5k.json'
DATABASE = "playground"
CONTAINER = "movies"

def get_create_database(client:CosmosClient, dbname):
    db= client.create_database_if_not_exists(id=dbname)
    return db

def get_create_movies_container(db:DatabaseProxy, name):
    vector_embedding_policy = {
        "vectorEmbeddings": [
            {
                "path":"/title_embeddings",
                "dataType":"float32",
                "distanceFunction":"cosine",
                "dimensions":DIMENSIONS
            },
            {
                "path":"/overview_embeddings",
                "dataType":"float32",
                "distanceFunction":"cosine",
                "dimensions":DIMENSIONS
            }
        ]
    }
    indexing_policy = {
        "includedPaths": [
            {
                "path": "/*"
            }
        ],
        "excludedPaths": [
            {
                "path": "/\"_etag\"/?"
            },
            {
                "path": "/title_embeddings/*"
            },
            {
                "path": "/overview_embeddings/*"
            }
        ],
        "vectorIndexes": [
            {"path": "/title_embeddings",
            "type": "quantizedFlat"
            },
            {"path": "/overview_embeddings",
            "type": "quantizedFlat"
            }
        ]
    }
    cont = db.create_container_if_not_exists(
        id=name,
        partition_key=PartitionKey(path='/id', kind='Hash'),
        indexing_policy=indexing_policy,
        vector_embedding_policy=vector_embedding_policy
    )
    return cont

def load_movies_embeddings():
    with open(MOVIEEMBEDDINGS, 'r') as jsonfile:
        data = json.load(jsonfile)
    return data

def upsert_data(container: ContainerProxy, data):
    for item in data:
        container.upsert_item(item)

def main():
    config = dotenv_values(".env")
    cosmos_client = CosmosClient(url=config['cosmos_uri'], credential=config['cosmos_readwrite_key'])
    db = get_create_database(cosmos_client, DATABASE)
    # print(json.dumps(db.read()))
    container = get_create_movies_container(db, CONTAINER)
    movies = load_movies_embeddings()
    upsert_data(container, movies)


if __name__ == "__main__":
    main()
