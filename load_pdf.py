import re
import fitz
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
from tqdm import tqdm  # progress bar
from typing import List, Dict
from os.path import join as ospathjoin
import numpy as np

SAVE_NAME = r"embedd.csv"
SAVE_PATH = ospathjoin("data_set", SAVE_NAME)
DEVICE = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def _text_formatter(text: str) -> str:
    return text.replace("\n", " ").strip()


def _open_read_pdf(path: str) -> List[Dict]:
    doc = fitz.open(path)
    pages_and_texts = []

    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = _text_formatter(text)

        pages_and_texts.append({
            "page_number": page_number + 1,
            "text": text
        })

    return pages_and_texts


def _split_sentence(pages_and_texts: List[Dict]) -> List[Dict]:
    npl = English()
    npl.add_pipe("sentencizer")

    for item in tqdm(pages_and_texts):
        item["sentences"] = list(npl(item["text"]).sents)
        item["sentences"] = [str(sent) for sent in item["sentences"]]

    return pages_and_texts


def __split_list(inp_list: List, slice_size: int = 10) -> List[List]:
    return [inp_list[i:i+slice_size] for i in range(0, len(inp_list), slice_size)]


def _split_chunk(pages_and_texts: List[Dict], chunk_size: int = 10) -> List[Dict]:
    for item in tqdm(pages_and_texts):
        item["sent_chunks"] = __split_list(item["sentences"], chunk_size)

    return pages_and_texts


def _merge_sentence(pages_and_texts: List[Dict]) -> List[Dict]:
    pages_and_chunks = []

    for item in tqdm(pages_and_texts):
        for sents_chunk in item["sent_chunks"]:
            chunk_dict = dict()
            chunk_dict["page_number"] = item["page_number"]

            joined_sentes = str().join(sents_chunk).replace("  ", " ").strip()
            # keep space after dot
            joined_sentes = re.sub(r'\.([A-Z])', r'. \1', joined_sentes)
            chunk_dict["sentence_chunk"] = joined_sentes

            chunk_dict["chunk_word_count"] = len(
                [word for word in joined_sentes.split(" ")])
            pages_and_chunks.append(chunk_dict)

    return pages_and_chunks


def _remove_short_chunk(pages_and_chunks: List[Dict], min_word_length: int = 30) -> List[Dict]:
    return [
        item for item in pages_and_chunks if item["chunk_word_count"] > min_word_length]


def _embedd(pages_and_chunks: List[Dict]) -> List[Dict]:

    embedding_model = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)
    for item in tqdm(pages_and_chunks):
        item["embedd"] = embedding_model.encode(item["sentence_chunk"])

    return pages_and_chunks


def _save_embed(pages_and_chunks: List[Dict], name: str = SAVE_NAME) -> None:
    pd.DataFrame(pages_and_chunks).to_csv(SAVE_PATH, index=False)


def embed_pdf(pdf_name: str) -> None:
    print("Embedding pdf file...")
    pdf_name = ospathjoin("data_set", pdf_name)
    pages_and_texts = _open_read_pdf(pdf_name)

    pages_and_texts = _split_sentence(pages_and_texts)
    pages_and_texts = _split_chunk(pages_and_texts)
    pages_and_chunks = _merge_sentence(pages_and_texts)
    pages_and_chunks = _remove_short_chunk(pages_and_chunks)
    pages_and_chunks = _embedd(pages_and_chunks)
    _save_embed(pages_and_chunks)

    print(f"Embedding file saved! See {SAVE_PATH}")


def load_embed(name: str = SAVE_NAME) -> tuple[dict, torch.Tensor]:
    """ Return pages and chunks dictionary, and embedding tensor"""
    df = pd.read_csv(SAVE_PATH)
    df["embedd"] = df["embedd"].apply(
        lambda x: np.fromstring(x.strip("[]"), sep=" "))

    pages_and_chunks = df.to_dict(orient="records")
    embedding = torch.tensor(np.array(
        df["embedd"].tolist()), dtype=torch.float32).to(DEVICE)

    print(f"Pdf loaded!: {len(pages_and_chunks)} pages and {
          embedding.shape} embedding tensor!")
    return pages_and_chunks, embedding


if __name__ == "__main__":  # test
    embed_pdf(r"Fast_Edge_Based_Stereo_Matching_Algorith.pdf")
    pg_a_chu, embedding = load_embed()
