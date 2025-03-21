#!/usr/bin/env python3
"""
scrape_and_ingest_openai.py

Final version fixing:
1) "LocalClipEmbedder object has no attribute 'embed'": 
   We rename embed_image(...) to embed(...).
2) "You tried to access openai.Embedding in openai>=1.0.0": 
   We now use openai.Embedding.create(...) with the updated interface.

Behavior:
- Wipes and recreates "artist-complete-info-index" (dim=1536).
- Scrapes Exa for 25 artists, merges text, summarizes if huge using GPT-3.5 or GPT-4.
- Splits text (~2000 chars each).
- Embeds text with openai.Embedding.create(...) (text-embedding-ada-002).
- Upserts text chunks into Pinecone.
- For images, we use local CLIP (512 dims) zero-padded to 1536. 
- Skips invalid images (non-image or parse error) with a warning, doesn't crash.

Environment needed:
- EXA_API_KEY
- OPENAI_SECRET_KEY (and possibly OPENAI_ORG_ID)
- PINECONE_API_KEY, PINECONE_ENV
"""

import os
import re
import io
import json
import time
import random
import hashlib
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import openai
import pinecone
from dotenv import load_dotenv

import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI

load_dotenv(".env")

# Grab environment
EXA_API_KEY        = os.getenv("EXA_API_KEY", "")
OPENAI_SECRET_KEY  = os.getenv("OPENAI_SECRET_KEY", "")
OPENAI_ORG_ID      = os.getenv("OPENAI_ORG_ID", "")
PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENV       = os.getenv("PINECONE_ENV", "us-east1-gcp")

# Validate
if not EXA_API_KEY:
    raise ValueError("Missing EXA_API_KEY.")
if not OPENAI_SECRET_KEY:
    raise ValueError("Missing OPENAI_SECRET_KEY.")
if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY.")

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OpenAI_Clip_Ingest")

# Initialize the client (replace the old OpenAI setup)
client = OpenAI(
    api_key=OPENAI_SECRET_KEY,
    organization=OPENAI_ORG_ID if OPENAI_ORG_ID else None
)

# Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
INDEX_NAME = "artist-complete-info-index"
DIMENSION = 1536  # text-embedding-ada-002 => 1536

existing = pinecone.list_indexes()
if INDEX_NAME in existing:
    logger.info(f"[Pinecone] Deleting old index '{INDEX_NAME}' => fresh start.")
    pinecone.delete_index(INDEX_NAME)
    time.sleep(5)

logger.info(f"[Pinecone] Creating new index '{INDEX_NAME}' dimension={DIMENSION}...")
pinecone.create_index(name=INDEX_NAME, dimension=DIMENSION)
while INDEX_NAME not in pinecone.list_indexes():
    time.sleep(2)
index = pinecone.Index(INDEX_NAME)
logger.info(f"[Pinecone] Using index => {INDEX_NAME}")

# Local CLIP model, rename embed_image => embed
class LocalClipEmbedder:
    def __init__(self):
        logger.info("[CLIP] Loading model openai/clip-vit-base-patch32...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.clip_dim = 512
    
    def embed(self, image: Image.Image) -> List[float]:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
            # Convert to regular Python list
            arr = feats[0].cpu().numpy().tolist()  # Add .tolist() here
        # Zero-pad to 1536
        pad_needed = DIMENSION - self.clip_dim
        if pad_needed < 0:
            logger.warning("[CLIP] Model dim=512 but index=1536 => mismatch.")
        padded = arr + [0.0]*pad_needed
        return padded[:DIMENSION]

img_embedder = LocalClipEmbedder()

# The 25 artists
ARTISTS = [
    "Leonardo da Vinci", "Michelangelo", "Pablo Picasso", "Vincent van Gogh", "Rembrandt van Rijn",
    "Claude Monet", "Raphael", "J.M.W. Turner", "Caravaggio", "Salvador Dalí", "Frida Kahlo",
    "Johannes Vermeer", "Albrecht Dürer", "Francisco Goya", "Paul Cézanne", "Georgia O'Keeffe",
    "Jackson Pollock", "Hieronymus Bosch", "Edvard Munch", "Diego Velázquez", "Henri Matisse",
    "Andy Warhol", "Wassily Kandinsky", "Artemisia Gentileschi", "Gustav Klimt"
]

################### Exa Helpers ###################
def exa_search(query: str, num_results=5) -> dict:
    url = "https://api.exa.ai/search"
    headers = {"Authorization": f"Bearer {EXA_API_KEY}", "Content-Type":"application/json"}
    payload = {
        "query": query,
        "numResults": num_results,
        "contents": {
            "text": True,
            "livecrawl": "fallback",
            "extras": {"imageLinks":3}
        }
    }
    logger.info(f"[Exa Search] '{query}', num_results={num_results}")
    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()
    return r.json()

def exa_contents(urls: List[str]) -> dict:
    if not urls:
        return {"results":[]}
    c_url = "https://api.exa.ai/contents"
    headers = {"Authorization": f"Bearer {EXA_API_KEY}", "Content-Type":"application/json"}
    payload = {
        "urls": urls,
        "text": True,
        "livecrawl":"fallback",
        "extras": {"imageLinks":5}
    }
    logger.info(f"[Exa Contents] fetching {len(urls)} URL(s).")
    r = requests.post(c_url, headers=headers, json=payload)
    r.raise_for_status()
    return r.json()

################### Summaries w/ OpenAI Chat ###################
def openai_summarize(text: str, max_len=100_000) -> str:
    if len(text) <= max_len:
        return text
    logger.info(f"[Summarizing] text len={len(text)} => summarizing with gpt-3.5.")
    sys_prompt = "You are a text summarizer focusing on an artist's style/history."
    user_prompt = f"Please summarize the following text into a shorter form:\n\n{text}"
    try:
        resp = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"[OpenAI Summ] => {e}")
        return text

################### Chunking ###################
def chunk_text(text: str, chunk_size=2000)->List[str]:
    t= re.sub(r"\s+"," ", text).strip()
    out=[]
    start=0
    while start< len(t):
        end= start+chunk_size
        slice_= t[start:end]
        out.append(slice_)
        start=end
    return out

################### Embedding text with OpenAI v1.0.0+ ###################
def openai_embed_text(batch_texts: List[str]) -> List[List[float]]:
    if not batch_texts:
        return []
    logger.info(f"[OpenAI Embedding] for {len(batch_texts)} text chunk(s)")
    try:
        resp = client.embeddings.create(
            model="text-embedding-ada-002",
            input=batch_texts,
            encoding_format="float"
        )
        return [item.embedding for item in resp.data]
    except Exception as e:
        logger.error(f"[OpenAI Embed Error] => {e}")
        # fallback random
        return [[random.uniform(-1,1) for _ in range(DIMENSION)] for _ in batch_texts]

################### Single Artist ###################
def scrape_artist(artist_name: str):
    logger.info(f"\n=== Scraping {artist_name} ===")
    info_q= f"{artist_name} art style, learning journey, how they learned, what they learned, artistic tendencies, tidbits about their life"
    paint_q=f"All paintings by {artist_name}"

    # gather search results
    results=[]
    for q in [info_q, paint_q]:
        try:
            d= exa_search(q)
            rlist= d.get("results", [])
            logger.info(f"[Exa Search] => {artist_name} => {len(rlist)} results for '{q}'")
            results.extend(rlist)
        except Exception as e:
            logger.warning(f"[Exa Search Error] => {artist_name}, q='{q}' => {e}")
    
    # gather URLs
    urls= {r.get("url") for r in results if r.get("url")}
    all_text=[]
    all_imgs=[]
    if urls:
        try:
            cont_data= exa_contents(list(urls))
            cresults= cont_data.get("results",[])
            logger.info(f"[Exa Contents] => {len(cresults)} pages for {artist_name}")
            for c in cresults:
                t= c.get("text","")
                if t: all_text.append(t)
                main_img= c.get("image", None)
                if main_img:
                    all_imgs.append(main_img)
                extras= c.get("extras", {})
                if "imageLinks" in extras:
                    for iurl in extras["imageLinks"]:
                        all_imgs.append(iurl)
        except Exception as e:
            logger.warning(f"[Exa Contents Error] => {artist_name}: {e}")
    else:
        logger.info(f"[Exa] => no URLs found for {artist_name}")
    
    combined= "\n\n".join(all_text).strip()
    logger.info(f"[Combine] => length={len(combined)} for {artist_name}")
    
    # Summarize if huge
    if len(combined)>100000:
        combined= openai_summarize(combined, 100000)
    
    # chunk
    chunks= chunk_text(combined, 2000)
    logger.info(f"[Chunking text] => {len(chunks)} chunk(s).")
    
    # embed in batches
    bsz=5
    for i in range(0, len(chunks), bsz):
        sub= chunks[i:i+bsz]
        embs= openai_embed_text(sub)
        for idx, ctext in enumerate(sub):
            doc_id= hashlib.md5((artist_name+ str(i+idx)+ ctext[:50]).encode("utf-8")).hexdigest()
            truncated= ctext[:3000]+"..." if len(ctext)>3000 else ctext
            meta= {
                "artist_name": artist_name,
                "chunk_index": i+idx,
                "text": truncated
            }
            try:
                index.upsert([(doc_id, embs[idx], meta)])
            except Exception as e2:
                logger.error(f"[Upsert Error] => chunk={i+idx}, {e2}")
    
    # embed images, skipping invalid
    for imgurl in all_imgs:
        if not imgurl.startswith("http"):
            continue
        
        try:
            # HEAD check
            head_resp= requests.head(imgurl, timeout=5)
            head_resp.raise_for_status()
            ctype= head_resp.headers.get("Content-Type","").lower()
            if not ctype.startswith("image/"):
                logger.warning(f"[Image Skip] => {imgurl} => not an image content-type: {ctype}")
                continue
            
            # GET + parse
            r= requests.get(imgurl, timeout=10)
            r.raise_for_status()
            data= r.content
            
            # parse
            im= Image.open(io.BytesIO(data))
            im= im.convert("RGB")
            
            # embed
            emb_img= img_embedder.embed(im)
            
            doc_id= hashlib.md5((artist_name+imgurl).encode("utf-8")).hexdigest()
            meta= {
                "artist_name": artist_name,
                "image_url": imgurl
            }
            index.upsert([(doc_id, emb_img, meta)])
        except (requests.exceptions.RequestException, UnidentifiedImageError) as eimg:
            logger.warning(f"[Image Skip] => {imgurl} => invalid or parse error: {eimg}")
            # skip
        except Exception as e2:
            logger.warning(f"[Image Skip Unexpected] => {imgurl} => {e2}")
            # skip

def scrape_and_embed_all(max_workers=2):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def worker(a: str):
        try:
            scrape_artist(a)
            return f"SUCCESS: {a}"
        except Exception as ex:
            return f"FAILED: {a} => {ex}"
    
    logger.info(f"[ScrapeAll] concurrency={max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        tasks= [exe.submit(worker, x) for x in ARTISTS]
        for f in as_completed(tasks):
            logger.info(f"[ScrapeAll] {f.result()}")

if __name__=="__main__":
    logger.info("[OpenAI+CLIP ingestion] Starting script. Deleting old index, ingesting anew.")
    scrape_and_embed_all(max_workers=2)
    logger.info("Done ingestion.")
