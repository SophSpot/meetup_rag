import openai
import pandas as pd
from scipy import spatial


EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

statements = [
    "Madrid is the capital of Spain",
    "Cincinnati is a city in Ohio",
    "Cats are really cute",
    "Poodles are a type of dog",
    "Paris is beautiful this time of year",
]

question = "Tell me something about kittens"

response = openai.Embedding.create(model=EMBEDDING_MODEL, input=statements)
embeddings = [e["embedding"] for e in response["data"]]
df = pd.DataFrame({"text": statements, "embedding": embeddings})


question_embedding_response = openai.Embedding.create(
    model=EMBEDDING_MODEL,
    input=question,
)

question_embedding = question_embedding_response["data"][0]["embedding"]

def relatedness_fn(x, y):
    return 1 - spatial.distance.cosine(x, y)

strings_and_relatednesses = [
    (row["text"], relatedness_fn(question_embedding, row["embedding"]))
    for i, row in df.iterrows()
]

strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
for string, relatedness in strings_and_relatednesses:
    print(string, relatedness)
