import os
import re
import shutil
import urllib.request
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Tuple

import fitz
import numpy as np
import openai
# import tensorflow_hub as hub
from fastapi import UploadFile
from lcserve import serving
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

embedder = None


def download_pdf(url, output_path):
    urllib.request.urlretrieve(url, output_path)


def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text


def pdf_to_text(path, start_page=1, end_page=None):
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []

    for i in range(start_page - 1, end_page):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list


def text_to_chunks(texts, word_length=150, start_page=1):
    text_toks = [t.split(' ') for t in texts]
    chunks = []

    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i : i + word_length]
            if (
                (i + word_length) > len(words)
                and (len(chunk) < word_length)
                and (len(text_toks) != (idx + 1))
            ):
                text_toks[idx + 1] = chunk + text_toks[idx + 1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[Page no. {idx+start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks


class SemanticSearch:
    def __init__(self, model_path = 'all-MiniLM-L6-v2'):
        # self.model = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
        self.model = SentenceTransformer(model_path)
        self.fitted = False

    def fit(self, data, batch_size=1000, n_neighbors=5):
        self.data = data
        self.embeddings = self.get_text_embedding(data, batch_size=batch_size)
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
        self.fitted = True

    def __call__(self, text, return_data=True):
        inp_emb = self.model.encode([text])
        neighbors = self.nn.kneighbors(inp_emb, return_distance=False)[0]

        if return_data:
            return [self.data[i] for i in neighbors]
        else:
            return neighbors

    def get_text_embedding(self, texts, batch_size=1000):
        return self.model.encode(texts, batch_size=batch_size)


def load_recommender(path, start_page=1):
    global embedder
    if embedder is None:
        embedder = SemanticSearch()

    texts = pdf_to_text(path, start_page=start_page)
    chunks = text_to_chunks(texts, start_page=start_page)
    embedder.fit(chunks)
    return 'Corpus Loaded.'


def generate_text(prompt, openAI_key, api_base, engine, api_version="2023-05-15"):
    openai.api_type = "azure"
    openai.api_key = openAI_key
    openai.api_base = api_base
    openai.api_version = api_version
    try:
        completions = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            max_tokens=512,
            n=1,
            stop=["Query"],
            temperature=0.7,
        )
        message = completions.choices[0].text
    except Exception as e:
        message = f"API Error: {str(e)}"
    return message


def generate_answer(question, openAI_key, openAI_base, openAI_deployment):
    topn_chunks = embedder(question)
    prompt = ""
    prompt += 'search results:\n\n'
    for c in topn_chunks:
        prompt += c + '\n\n'

    prompt += (
        "Instructions: Compose a comprehensive reply to the query using the search results given. "
        "Cite each reference using [ Page Number] notation (every result has this number at the beginning). "
        "Citation should be done at the end of each sentence. If the search results mention multiple subjects "
        "with the same name, create separate answers for each. Only include information found in the results and "
        "don't add any additional information. Make sure the answer is correct and don't output false content. "
        "If the text does not relate to the query, simply state 'Text Not Found in PDF'. Ignore outlier "
        "search results which has nothing to do with the question. Only answer what is asked. The "
        "answer should be short and concise. Answer step-by-step. \n\nQuery: {question}\nAnswer: "
    )

    prompt += f"Query: {question}\nAnswer:"
    answer = generate_text(prompt, openAI_key, openAI_base, openAI_deployment)
    return answer


def load_openai_key() -> Tuple[str, str, str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE")
    deployment = os.environ.get("OPENAI_DEPLOYMENT")
    if api_key is None or api_base is None or deployment is None:
        raise Exception("Missing one or more required variables for Azure OpenAI. Get the Azure OpenAI variables here:"
                        " https://learn.microsoft.com/en-us/azure/cognitive-services/openai/quickstart?tabs=command-line&pivots=programming-language-python")

    return api_key, api_base, deployment


@serving
def ask_url(url: str, question: str):
    download_pdf(url, 'corpus.pdf')
    load_recommender('corpus.pdf')
    openAI_key, openAI_base, openAI_deployment = load_openai_key()
    return generate_answer(question, openAI_key, openAI_base, openAI_deployment)


@serving
async def ask_file(file: UploadFile, question: str) -> str:
    suffix = Path(file.filename).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    load_recommender(str(tmp_path))
    openAI_key, openAI_base, openAI_deployment = load_openai_key()
    return generate_answer(question, openAI_key, openAI_base, openAI_deployment)