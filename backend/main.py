#!/usr/bin/env python3
"""
main.py

A significantly enhanced FastAPI server to query the "artist-complete-info-index"
populated by scrape_and_ingest_openai.py. We now focus solely on image queries:

- The user uploads an image of their artwork.
- We embed the image with the local CLIP model from Transformers (dimension=512).
- We zero-pad the result to dimension=1536 to match Pinecone's index (text-embedding-ada-002 + CLIP).
- We query Pinecone (dimension=1536) for top_k chunks or image references.
- We filter out matches below a similarity threshold.
- We then use OpenAI GPT calls to:
    1) Outline the journey of the most similar famous artist(s).
    2) Provide direct feedback and tips to help the user improve their artwork.
    3) Offer recommended next steps or resources to deepen their art practice.
- We also provide a separate function/endpoint to return only the top artist(s) name(s).

Environment Variables:
- OPENAI_SECRET_KEY, (possibly OPENAI_ORG_ID, OPENAI_PROJECT_ID)
- PINECONE_API_KEY, PINECONE_ENV
- Confirm the dimension=1536 in Pinecone for text-embedding-ada-002,
  and CLIP zero-padding approach for images.

Usage:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import re
import io
import json
import random
import logging
from typing import List

import openai
import pinecone
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

load_dotenv(".env")

# Environment
OPENAI_SECRET_KEY  = os.getenv("OPENAI_SECRET_KEY", "")
OPENAI_ORG_ID      = os.getenv("OPENAI_ORG_ID", "")
OPENAI_PROJECT_ID  = os.getenv("OPENAI_PROJECT_ID", "")
PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENV       = os.getenv("PINECONE_ENV", "us-east1-gcp")  # Example

if not OPENAI_SECRET_KEY:
    raise ValueError("Missing OPENAI_SECRET_KEY.")
if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY.")

# OpenAI init
openai.api_key = OPENAI_SECRET_KEY
if OPENAI_ORG_ID:
    openai.organization = OPENAI_ORG_ID

# Pinecone init
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
INDEX_NAME = "artist-complete-info-index"
if INDEX_NAME not in pinecone.list_indexes():
    raise ValueError(f"Index '{INDEX_NAME}' not found. Please run scrape_and_ingest_openai.py first.")
index = pinecone.Index(INDEX_NAME)
DIMENSION = 1536  # text-embedding-ada-002 + zero-padded CLIP

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OpenAI_Clip_RAG")

# Local CLIP for image queries
class LocalClipEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        logger.info(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.clip_output_dim = 512  # openai/clip-vit-base-patch32 => 512 dims

    def embed_image(self, image: Image.Image) -> List[float]:
        """
        Convert an RGB PIL Image into a 1536-dimensional vector:
        512-d CLIP embedding + zero-padding to 1536.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)  # shape: (1, 512)
            emb = outputs[0].cpu().numpy()  # (512,)
        # Zero-pad to 1536
        pad_needed = DIMENSION - self.clip_output_dim
        if pad_needed < 0:
            logger.warning("Index dimension smaller than CLIP dimension, adjusting approach.")
        # Convert numpy float32 to regular Python float
        padded = [float(x) for x in emb] + [0.0] * pad_needed
        return padded[:DIMENSION]

# Instantiate the local clip embedder
clip_embedder = LocalClipEmbedder()

##############################
# GPT Helper Calls
##############################

def outline_artist_journey(chunk_text: str) -> str:
    """
    Provide a creative outline of the most similar artist's journey into becoming a great artist:
    - Early influences and challenges
    - How they practiced and improved
    - Unique attributes, quotes, and perspectives
    - How this can help a new artist
    """
    sys_prompt = "You are an AI art mentor. Provide a creative, inspiring outline of the artist's journey based on the chunks."
    user_prompt = f"""
We found these chunk(s) in the database:
{chunk_text}

Please detail:
1) The background and early influences
2) The challenges and how they overcame them
3) Key techniques and how they practiced
4) Unique attributes or perspectives
5) Notable quotes or personal art philosophies
6) How this can help a new artist on a similar journey

Output a thorough and inspiring narrative or bullet points.
"""
    try:
        resp = openai.chat.completions.create(
            model="o3-mini",  # or gpt-4 if you have access
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"GPT Outline Journey error => {e}")
        return "Error producing artist journey."

def feedback_and_tips_for_artwork(chunk_text: str) -> str:
    """
    Provide direct feedback on the user's artwork based on the style found in chunk_text.
    Include suggestions for improvement and supportive advice.
    """
    sys_prompt = "You are an AI art mentor. Provide constructive feedback based on the chunk text which represents a known artist's style."
    user_prompt = f"""
We found these chunk(s) describing an artist's style or approach:
{chunk_text}

A user has uploaded an artwork that resembles this style. Please:
1) Offer positive feedback regarding what they might be doing well in emulating this style
2) Suggest practical tips, exercises, or techniques to improve
3) Provide supportive, motivating advice to keep them engaged and growing
"""
    try:
        resp = openai.chat.completions.create(
            model="o3-mini",  # or gpt-4 if you have access
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"GPT feedback_and_tips_for_artwork error => {e}")
        return "Error providing feedback and tips."

def recommended_next_steps(chunk_text: str) -> str:
    """
    Provide recommended next steps or resources to deepen the user's art practice,
    based on the style in chunk_text.
    """
    sys_prompt = "You are an AI art mentor. Suggest relevant next steps, references, or resources to help an artist grow."
    user_prompt = f"""
We found these chunk(s) describing an artist's style:
{chunk_text}

For the user who is inspired by this style, recommend:
1) Specific exercises to practice
2) Relevant literature or art resources
3) Mindset or creative approaches
4) Additional learning or classes that could help
5) Inspiration sources (other artists, museums, online platforms)

Output a concise, actionable plan.
"""
    try:
        resp = openai.chat.completions.create(
            model="o3-mini",  # or gpt-4 if you have access
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"GPT recommended_next_steps error => {e}")
        return "Error suggesting next steps."

##############################
# FastAPI App
##############################

app = FastAPI(
    title="OpenAI + CLIP RAG (Image-Only)",
    version="1.0.0",
    description="Query an index with an uploaded image, then summarize with GPT."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    logger.info("OpenAI + CLIP RAG server started. Ready to handle image queries...")

##############################
# Endpoints
##############################

@app.post("/ask-image")
def ask_image(
    file: UploadFile,
    top_k: int = Form(3),
    threshold: float = Form(0.2)
):
    """
    The user uploads a photo of their artwork, and we find which famous artist's style
    it most closely resembles. We then:
    1) Outline that artist's journey
    2) Provide direct feedback and tips for improvement
    3) Offer recommended next steps or resources to deepen the user's skills
    """
    image_data = file.file.read()
    if not image_data:
        raise HTTPException(status_code=400, detail="No image data provided.")
    
    # 1) embed with local CLIP
    img_emb = clip_embedder.embed_image(Image.open(io.BytesIO(image_data)).convert("RGB"))
    
    # 2) query pinecone
    res = index.query(vector=img_emb, top_k=top_k, include_metadata=True)
    if not res.matches:
        return {
            "answer":"No data found in index for your image."
        }
    
    # 3) threshold filtering
    valid = [m for m in res.matches if m.score >= threshold]
    if not valid:
        return {
            "answer":"No relevant data above threshold."
        }
    
    # 4) combine chunks from top matches
    combined_text = ""
    top_artist_name = "Unknown"
    top_score = 0.0

    for i, v in enumerate(valid):
        artist = v.metadata.get("artist_name","Unknown")
        chunk_text = v.metadata.get("text","")
        score_val = v.score
        image_url = v.metadata.get("image_url","")
        
        combined_text += (
            f"\n---[artist:{artist}"
            f"|score:{score_val:.3f}"
            f"|image_url:{image_url}]---\n"
            f"{chunk_text}\n"
        )
        # Track the highest-scoring match as the "top" artist
        if i == 0 or score_val > top_score:
            top_artist_name = artist
            top_score = score_val
    
    # 5) GPT calls: 
    #    a) Outline the journey of the most similar artist(s)
    #    b) Provide direct feedback for the user's artwork
    #    c) Suggest recommended next steps/resources
    journey_outline = outline_artist_journey(combined_text)
    feedback_tips = feedback_and_tips_for_artwork(combined_text)
    next_steps = recommended_next_steps(combined_text)
    
    return {
        "image_filename": file.filename,
        "top_artist_name": top_artist_name,
        "most_similar_artist_journey_outline": journey_outline,
        "artwork_feedback_and_tips": feedback_tips,
        "recommended_next_steps": next_steps,
        "chunks_used": len(valid)
    }

@app.post("/artist-name-only")
def artist_name_only(
    file: UploadFile,
    top_k: int = Form(3),
    threshold: float = Form(0.2)
):
    """
    Return ONLY the name of the top-matching artist.
    This is useful if the frontend wants to display a quick header 
    without needing the full GPT-based details.
    """
    image_data = file.file.read()
    if not image_data:
        raise HTTPException(status_code=400, detail="No image data provided.")
    
    # 1) embed with local CLIP
    img_emb = clip_embedder.embed_image(Image.open(io.BytesIO(image_data)).convert("RGB"))
    
    # 2) query pinecone
    res = index.query(vector=img_emb, top_k=top_k, include_metadata=True)
    if not res.matches:
        return {"artist_name": "No data found in index."}
    
    # 3) threshold filtering
    valid = [m for m in res.matches if m.score >= threshold]
    if not valid:
        return {"artist_name": "No relevant data above threshold."}
    
    # 4) find highest score
    top_match = max(valid, key=lambda x: x.score)
    top_artist_name = top_match.metadata.get("artist_name","Unknown")
    
    return {
        "artist_name": top_artist_name
    }

@app.get("/")
def root():
    return {
        "message":"OpenAI + local CLIP RAG (Image-Only) server. Endpoints: /ask-image and /artist-name-only."
    }
