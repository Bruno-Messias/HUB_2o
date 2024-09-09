from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import WebBaseLoader
import os

DATA_PATH = 'data'
CHROMA_PATH = 'db'

db_vector = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=OpenAIEmbeddings()
    )

def remove_languages_from_metadata(document):
    if 'languages' in document.metadata:
        del document.metadata['languages']
    return document

def load_documents(new_files):
    file_paths = [os.path.join(DATA_PATH, file) for file in new_files]
    loader = UnstructuredLoader(file_paths,
                                chunking_strategy="basic",
                                max_characters=999999999999999999999,
                                include_orig_elements=False,
                                )
    
    return loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document], db):
    # Add or Update the documents.
    existing_items = db.get(include=['metadatas'])
    metadatas = existing_items['metadatas']
    filenames = list(set([metadata.get('filename') for metadata in metadatas]))

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks:
        if chunk.metadata["filename"] not in filenames:
            new_chunks.append(chunk)

    if new_chunks:
        print(f"👉 Adding new Documents: {len(list(set([new_chunk.metadata['filename'] for new_chunk in new_chunks])))}")
        db.add_documents(new_chunks)
        
def check_new_files_and_update():
    existing_items = db_vector.get(include=['metadatas'])
    metadatas = existing_items['metadatas']
    db_files = list(set([metadata.get('filename') for metadata in metadatas]))
    folder_files = os.listdir(DATA_PATH)
    
    print(f"Number of documents in Database: {len(db_files)}")
    
    new_files = []
    for file in folder_files:
        if file not in db_files:
            new_files.append(file)
    
    if new_files:
        data = load_documents(new_files)
        data = [remove_languages_from_metadata(doc) for doc in data]
        chunks = split_documents(data)
        add_to_chroma(chunks, db_vector)
    else:
        print("✅ No new documents to add")

def check_delete_file():
    existing_items = db_vector.get(include=['metadatas'])
    metadatas = existing_items['metadatas']
    db_files = list(set([metadata.get('filename') for metadata in metadatas]))
    folder_files = os.listdir(DATA_PATH)
    
    delete_files = []
    for file in db_files:
        if file not in folder_files:
            delete_files.append(file)
    
    if delete_files:
        print(f"🔴 Deleting this files: {delete_files}")
        metadata_filter = {'filename': {'$in': delete_files}}
        results = db_vector.get(where=metadata_filter, include=['metadatas'])
        ids = results['ids']
        db_vector.delete(ids=ids)

def load_web_documents(db):
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    db.add_documents(doc_splits)

def prepare_db_rag():
    # check_new_files_and_update()
    # check_delete_file()
    load_web_documents(db_vector)
    retriever = db_vector.as_retriever(k=4)
    return retriever