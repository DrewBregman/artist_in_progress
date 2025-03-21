import os
import json
import random
import re
import hashlib
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import anthropic
import pinecone
from dotenv import load_dotenv

##############################################################################
# 0. Configuration and Setup
##############################################################################

load_dotenv(".env")

EXA_API_KEY = os.environ.get("EXA_API_KEY", "")
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_ENV = os.environ.get("PINECONE_ENV", "us-east1-gcp")

if not EXA_API_KEY:
    raise ValueError("Missing EXA_API_KEY.")
if not CLAUDE_API_KEY:
    raise ValueError("Missing CLAUDE_API_KEY.")
if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY.")

anthropic_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

INDEX_NAME = "artist-complete-info-index"
DIMENSION = 768  # We'll store vectors of length 768 from a Claude 'embedding trick'

if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(name=INDEX_NAME, dimension=DIMENSION)

index = pinecone.Index(INDEX_NAME)

# We handle 25 famous artists:
ARTISTS = [
    "Leonardo da Vinci",
    "Michelangelo",
    "Pablo Picasso",
    "Vincent van Gogh",
    "Rembrandt van Rijn",
    "Claude Monet",
    "Raphael",
    "J.M.W. Turner",
    "Caravaggio",
    "Salvador Dalí",
    "Frida Kahlo",
    "Johannes Vermeer",
    "Albrecht Dürer",
    "Francisco Goya",
    "Paul Cézanne",
    "Georgia O'Keeffe",
    "Jackson Pollock",
    "Hieronymus Bosch",
    "Edvard Munch",
    "Diego Velázquez",
    "Henri Matisse",
    "Andy Warhol",
    "Wassily Kandinsky",
    "Artemisia Gentileschi",
    "Gustav Klimt"
]

##############################################################################
# 1. Exa Search Helpers
##############################################################################

def exa_search(query: str, num_results: int = 5, text: bool = True) -> dict:
    """
    Calls Exa's /search endpoint to retrieve up to `num_results` results.
    If text=True, Exa attempts to return the text content in the search results automatically.

    Returns the response JSON as a Python dict.

    If an error occurs, we raise an exception.
    """
    url = "https://api.exa.ai/search"
    headers = {
        "Authorization": f"Bearer {EXA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "numResults": num_results,
        "contents": {
            "text": text,
            # We might also request highlights, summaries, etc. if needed
            "livecrawl": "fallback",
            "extras": {
                "imageLinks": 3  # We'll attempt to get up to 3 images from each search result
            }
        }
    }
    print(f"[Exa Search] Query: {query} | numResults={num_results} | text={text}")
    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()

def exa_contents(urls: List[str], text: bool = True) -> dict:
    """
    Calls Exa's /contents endpoint to retrieve in-depth data for each URL.
    This includes full text, possibly more images, etc.

    Returns the response JSON as a dict, or raises an exception on error.
    """
    if not urls:
        return {"results": []}

    c_url = "https://api.exa.ai/contents"
    headers = {
        "Authorization": f"Bearer {EXA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "urls": urls,
        "text": text,
        "livecrawl": "fallback",
        "extras": {
            "imageLinks": 5
        }
    }
    print(f"[Exa Contents] Fetching {len(urls)} URLs in detail.")
    resp = requests.post(c_url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()

##############################################################################
# 2. Anthropic Embedding Trick + Summarization
##############################################################################

def anthropic_embedding_trick(text: str) -> List[float]:
    """
    We produce a 768-dimensional embedding from text by instructing Claude
    to output a JSON array of floats in [-1, 1]. This is a hack because 
    Anthropic has no public embeddings endpoint.

    If it fails, we fallback to random vectors.
    """
    system_prompt = f"""
You are an AI that converts text into a 768-dimensional embedding, as a JSON array of floats 
each between -1.0 and 1.0. No extra text, strictly a JSON array.

TEXT:
{text}
""".strip()

    try:
        resp = anthropic_client.messages.create(
            model="claude-3-opus-20240229",
            messages=[{
                "role": "user",
                "content": f"{system_prompt}"
            }],
            temperature=0.0,
            max_tokens=3000
        )
        raw = resp.content[0].text.strip()       # Updated response parsing
        arr = json.loads(raw)
        if isinstance(arr, list) and len(arr) == DIMENSION:
            final_vec = []
            for val in arr:
                if isinstance(val, (int, float)):
                    final_vec.append(float(val))
                else:
                    final_vec.append(random.uniform(-1, 1))
            return final_vec
        else:
            print("[Embedding Trick] Invalid array shape; fallback to random.")
            return [random.uniform(-1, 1) for _ in range(DIMENSION)]
    except Exception as e:
        print(f"[Embedding Trick] error: {e} => fallback to random.")
        return [random.uniform(-1, 1) for _ in range(DIMENSION)]


def anthropic_summarize_text(text: str, max_tokens: int = 1000) -> str:
    """
    If the text is extremely large, we can ask Claude to produce a summary
    to reduce chunking overhead or Pinecone usage.
    """
    summary_prompt = f"""
You are an AI assistant. Summarize the following text into a shorter cohesive form,
capturing important details about the artist. Limit to ~{max_tokens} tokens max.

TEXT:
{text}
"""
    try:
        resp = anthropic_client.messages.create(
            model="claude-3-opus-20240229",
            messages=[{
                "role": "user",
                "content": f"{summary_prompt}"
            }],
            max_tokens=2000,
            temperature=0.7
        )
        return resp.content[0].text.strip()       # Updated response parsing
    except Exception as e:
        print(f"[Summarize Error] {e} => returning original text.")
        return text

##############################################################################
# 3. Adaptive Sentence Chunking
##############################################################################

def sentence_chunk(text: str, chunk_size: int = 2000) -> List[str]:
    """
    Splits text on sentence boundaries up to chunk_size.
    We'll do a naive approach using regex to identify sentence boundaries (.!?).
    This yields more coherent chunks for retrieval.

    Constraints: We only have 're' available, not advanced NLP libs.
    """
    # Replace multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    # Use a simple pattern to split on .!? plus whitespace or EOL
    sentences = re.split(r'([.?!])', text)
    # This returns chunks of text plus the punctuation in separate items, so let's recombine.
    # e.g., ["Hello world", ".", " Another sentence", ".", " Last", ".", ""]
    full_sents = []
    for i in range(0, len(sentences) - 1, 2):
        s = sentences[i].strip()
        punct = sentences[i+1].strip()
        combined = (s + punct).strip()
        if combined:
            full_sents.append(combined)

    # Now chunk them
    chunks = []
    current = ""
    for s in full_sents:
        if len(current) + len(s) <= chunk_size:
            current += " " + s
        else:
            if current.strip():
                chunks.append(current.strip())
            current = s
    # Add remainder
    if current.strip():
        chunks.append(current.strip())

    return chunks

##############################################################################
# 4. Full Scrape + Embed for One Artist
##############################################################################

def scrape_single_artist(artist_name: str):
    """
    For a single artist:
      1) Build a text-based query about their style, learning journey, etc.
      2) Build another query about "All paintings by <artist>"
      3) Merge results
      4) Use exa_contents to get detailed text and images from each result
      5) Possibly summarize if text is huge
      6) Chunk -> embed -> upsert into Pinecone
    """
    print(f"=== Scraping data for {artist_name} ===")

    # Step 1: Queries
    info_query = f"{artist_name} art style, learning journey, how they learned, what they learned, artistic tendencies, tidbits about their life"
    paint_query = f"All paintings by {artist_name}"

    # Step 2: Exa search calls
    results = []
    for q in [info_query, paint_query]:
        try:
            srch = exa_search(q, num_results=5, text=True)
            if "results" in srch:
                results.extend(srch["results"])
        except Exception as e:
            print(f"[Exa Search Error] {e} (query: {q})")

    # Step 3: Extract URLs for deeper contents
    urls = set()
    for r in results:
        url_ = r.get("url")
        if url_:
            urls.add(url_)

    # Step 4: exa_contents
    all_text = []
    all_images = []
    try:
        cont_data = exa_contents(list(urls), text=True)
        res2 = cont_data.get("results", [])
        for rd in res2:
            page_text = rd.get("text", "")
            if page_text:
                all_text.append(page_text)
            main_img = rd.get("image", None)
            if main_img:
                all_images.append(main_img)
            extras = rd.get("extras", {})
            imgs2 = extras.get("imageLinks", [])
            for im2 in imgs2:
                all_images.append(im2)
    except Exception as e:
        print(f"[Exa Contents Error] {e}")

    # Combine everything into one big text block
    big_text = "\n\n".join(all_text).strip()

    # Optional Summarization if text is huge (say > 100k chars)
    if len(big_text) > 100000:
        print(f"[Info] Summarizing huge text for {artist_name}, length={len(big_text)}")
        big_text = anthropic_summarize_text(big_text, max_tokens=5000)

    # Step 5: Chunk
    chunks = sentence_chunk(big_text, chunk_size=2000)

    # Step 6: For each chunk, embed + upsert
    images_uniq = list(set(all_images))
    for i, ctext in enumerate(chunks):
        if not ctext.strip():
            continue
        try:
            emb = anthropic_embedding_trick(ctext)
            doc_id = hashlib.md5((artist_name + str(i) + ctext[:50]).encode("utf-8")).hexdigest()
            metadata = {
                "artist_name": artist_name,
                "chunk_index": i,
                "text": ctext,
                "image_links": images_uniq
            }
            index.upsert([(doc_id, emb, metadata)])
        except Exception as e:
            print(f"[Embedding/Upsert Error] {artist_name} chunk={i} => {e}")


##############################################################################
# 5. Concurrency: Scrape & Embed All Artists
##############################################################################

def scrape_and_embed_all_artists(max_workers: int = 4):
    """
    Scrapes and embeds data for the 25 artists concurrently.
    This can be time-consuming with real Exa + real Anthropic calls, so concurrency helps.

    max_workers sets the thread pool size. Default=4 is a reasonable compromise.
    """
    def worker(artist):
        try:
            scrape_single_artist(artist)
            return f"SUCCESS: {artist}"
        except Exception as ex:
            return f"FAILED: {artist} => {ex}"

    print(f"[ScrapeAll] Starting concurrency with max_workers={max_workers} ...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, a) for a in ARTISTS]
        for f in as_completed(futures):
            result = f.result()
            print(f"[ScrapeAll] {result}")

    print("[ScrapeAll] All artist scraping tasks completed.")
