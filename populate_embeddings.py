from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter

with open("wikicontent", encoding="utf8") as f:
    wiki_data = f.read()

embedding_models = OpenAIEmbeddings()
text_splitter = TokenTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=1600, chunk_overlap=0
)
split_text = text_splitter.split_text(wiki_data)
Chroma.from_texts(texts=split_text,
    persist_directory="./chroma_db",
    embedding=embedding_models,
    collection_metadata={"hnsw:space": "cosine"},
)