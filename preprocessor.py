import csv
import json

mymovies = []
with open('./data/tmdb_5000_movies.csv') as csvfile:
    moviereader=csv.DictReader(csvfile,delimiter=',',quotechar='"')
    for movie in moviereader:
        genres = [g['name'] for g in json.loads(movie['genres'])]
        id = movie['id']
        keywords = [k['name'] for k in json.loads(movie['keywords'])]
        title = movie['title']
        overview = movie['overview']
        mymovie = {
            "genres": ", ".join(genres),
            "id": id,
            "keywords": ", ".join(keywords),
            "title": title,
            "overview": overview,
            "title_embeddings": [],
            "overview_embeddings": []
        }
        mymovies.append(mymovie)

with open('./data/parsed_movies.json','w') as jsonfile:
  json.dump(mymovies, jsonfile)