import spacy
import fitz  # PyMuPDF
from transformers import BertTokenizer, BertModel
import torch

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def parse_document(file_path):
    if file_path.endswith(".pdf"):
        content = extract_text_from_pdf(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

    doc = nlp(content)
    best_practices = []
    keywords = ["should", "recommended", "must", "important", "guideline"]

    for sent in doc.sents:
        if any(keyword in sent.text.lower() for keyword in keywords):
            best_practices.append(sent.text.strip())

    return {"best_practices": best_practices}
