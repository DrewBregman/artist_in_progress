#!/usr/bin/env python3
"""
Quick test for scrape_and_ingest with a subset of artists.
"""
import scrape_and_ingest

if __name__ == "__main__":
    # Override ARTISTS with a subset
    scrape_and_ingest.ARTISTS = [
        "Leonardo da Vinci",
        "Pablo Picasso",
        "Vincent van Gogh",
        "Claude Monet",
        "Salvador Dalí"
    ]
    print(f"[TEST] Using subset of {len(scrape_and_ingest.ARTISTS)} artists")
    
    # Run with concurrency
    scrape_and_ingest.scrape_and_embed_all_artists(max_workers=2)
    print("[TEST] Done.")