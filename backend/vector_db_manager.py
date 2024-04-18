from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
import os


class VectorDbManager:
    def __init__(self, embedding_function: Embeddings, embedding_name: str, chunk_size: int, db_directory: Path):
        self.embedding_function = embedding_function
        self.embedding_name = embedding_name
        self.db_directory = db_directory
        self.chunk_size = chunk_size

    def create_vector_store_from_pdf(self, pdf_path):
        """
        create a chroma vector store from a pdf file path
        store the vector store in the db_directory/pdf_name
        where pdf_name is the name of the pdf file

        :param pdf_path:
        :return:
        """
        pdf_path = Path(pdf_path)
        pdf_name = pdf_path.name
        vector_directory = self.db_directory / self.embedding_name / pdf_name

        if os.path.isdir(vector_directory):
            print(f"{vector_directory} found, not recreating a vector store")
            return 0

        print(f"creating vector store for {vector_directory}")
        file = PyPDFLoader(pdf_path)

        docs = []
        pages = file.load_and_split()
        for j, page in enumerate(pages):
            docs.append(page)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=64,
            length_function=len,
            is_separator_regex=False,
        )
        docs = text_splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(docs, self.embedding_function, persist_directory=vector_directory)
        print("pdf vector store created")
        print(vectorstore)

    def create_vector_store_from_latex(self, latex_path: Path):
        """
        create a chroma vector store from a latex file path
        store the vector store in the db_directory/doc_name
        where doc_name is the name of the latex file

        :param latex_path:
        :return:
        """
        doc_name = latex_path.name
        vector_directory = self.db_directory / self.embedding_name / doc_name

        if os.path.isdir(vector_directory):
            print(f"{vector_directory} found, not recreating a vector store")
            return 0

        print(f"creating vector store for {vector_directory}")

        with open(latex_path, mode="r") as file:
            text_splitter = RecursiveCharacterTextSplitter.from_language(Language.MARKDOWN, chunk_size=self.chunk_size, chunk_overlap=64)
            texts = text_splitter.split_text(file.read())

        print(texts)
        vectorstore = Chroma.from_texts(texts, self.embedding_function, persist_directory=vector_directory)
        print("latex vector store created")
        print(vectorstore)

    def get_chroma(self, doc_name):
        """
        get the chroma vector store for a given document name

        :param doc_name:
        :return:
        """
        vector_directory = self.db_directory / self.embedding_name / doc_name
        return Chroma(persist_directory=vector_directory, embedding_function=self.embedding_function)
