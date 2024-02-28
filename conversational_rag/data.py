from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.schema import format_document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from glob import glob
from multiprocessing import Pool, Manager
from tqdm import tqdm

from misc import memory

def load_file(file_path):
    loader = UnstructuredFileLoader(file_path)
    return loader.load()

def worker(input_queue, output_list):
    while True:
        file_path = input_queue.get()
        if file_path is None:
            break
        output_list.append(load_file(file_path))
        input_queue.task_done()

@memory.cache
def load_pdfs(_glob):
    files = glob(_glob)
    manager = Manager()
    input_queue = manager.Queue()
    output_list = manager.list()

    with Pool() as pool:
        for _ in tqdm(range(pool._processes)):
            pool.apply_async(worker, (input_queue, output_list))

        for file in tqdm(files):
            input_queue.put(file)

        for _ in range(pool._processes):
            input_queue.put(None)

        with tqdm(total=len(files)) as progress_bar:
            while len(output_list) < len(files):
                current_length = len(output_list)
                progress_bar.update(current_length - progress_bar.n)
    return [doc[0] for doc in output_list]


@memory.cache
def load_chunks(glob):
    docs = load_pdfs(glob)

    # Chunk text
    text_splitter = CharacterTextSplitter(chunk_size=800,
                                          chunk_overlap=0)
    return text_splitter.split_documents(docs)


@memory.cache
def load_retriever(glob, DEFAULT_DOCUMENT_PROMPT):
    chunked_documents = load_chunks(glob)

    # Load chunked documents into the FAISS index
    huggingface_embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2', multi_process=True, show_progress=True)
    db = FAISS.from_documents(chunked_documents, huggingface_embeddings)

    return db.as_retriever()


def _combine_documents(
    docs, document_prompt, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)
