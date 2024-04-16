from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from openai import OpenAI
from backend.vector_db_manager import VectorDbManager
from typing import Optional, Iterator, Dict
from pathlib import Path

# point to the local server, I personally use LM Studio to run local LLMs
# You can change this to any other OpenAI API endpoint, local or not
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")


class InferenceInstance:
    def __init__(self, vector_db_manager: VectorDbManager, nb_chunks_retrieved: int = 4):
        self.vector_db_manager = vector_db_manager
        self.history = []
        self.nb_chunks_retrieved = nb_chunks_retrieved

    def get_next_token(self, input_user: str, doc_name: str) -> Iterator[Dict[str, str]]:
        new_assistant_message = {"role": "assistant", "content": ""}
        search_results = self._get_search_results(input_user, doc_name)
        print(f"search results: {search_results}")
        pages = self._update_history(input_user, search_results)
        pages_info = f"pages used : p" + " p".join(pages)
        print(f"history: {self.history}")
        completion = self._get_completion()

        for chunk in completion:
            new_assistant_message["content"] += chunk.choices[0].delta.content
            yield pages_info + " " + new_assistant_message["content"]

    def _get_search_results(self, input_user: str, doc_name: str):
        print(f"input_user: {input_user}")
        vector_db = self.vector_db_manager.get_chroma(doc_name)
        return vector_db.similarity_search(input_user, k=4)

    def _update_history(self, input_user: str, search_results):
        some_context = ""
        pages = []
        for result in search_results:
            pages.append(str(result.metadata['page']))
            some_context += result.page_content + "\n\n"
        self.history.append({"role": "system", "content": f"relevant content for user question {some_context}"})
        self.history.append({"role": "user", "content": input_user})
        return pages

    def _get_completion(self):
        return client.chat.completions.create(
            model="local-model",
            messages=self.history,
            temperature=0.7,
            stream=True,
        )

