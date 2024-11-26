# Movie recommender agent

This is a proof-of-concept project that leverages a modern embedding model, Azure CosmosDB vector search (preview feature as of 2024 November) and GPT chat completion.

## Azure setup

1. Create a CosmosDB account
    * enable Full-text & Hybrid search
    * enable Vector Search
    * note URI, read/write key and read-only key
2. Create an Azure OpenAI resource
    * note endpoint and key
3. Deploy embeddings model
    * note deployment name
    * text-embedding-3-small was used, dimensions parameter can be as small as 512
4. Deploy completions model
    * note deployment name
    * gpt-4o-mini was used

## Get the data

Download [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata). Extract and save files into `data` folder.

## Local dev setup

    python3 -m venv venv
    . ./venv/bin/activate
    pip install python-dotenv openai azure-cosmos

Create and populate your `.env` file, see `example.dotenv`.

## Prepare the data

1. Run `preprocessor.py` to create `parsed_movies.json`
2. Run `generate_embeddings.py` to create `movies_embeddings_5k.json`
3. Run `store_in_cosmos.py` to upsert into cosmosDB. The script will create a database and a container.

## Use semantic search or chat experience

* Run `query_cosmos.py` to play with semantic search
* Run `chat.py` to interact with the chat completion model using the semantic search results

## Inspiration and references

[Index and query vectors in Azure Cosmos DB for NoSQL in Python](https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/how-to-python-vector-index-query)

[Building a RAG application with Azure Cosmos DB for NoSQL](https://github.com/microsoft/AzureDataRetrievalAugmentedGenerationSamples/blob/main/Python/CosmosDB-NoSQL_VectorSearch/CosmosDB-NoSQL-Vector_AzureOpenAI_Tutorial.ipynb)