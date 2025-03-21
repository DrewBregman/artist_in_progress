#!/usr/bin/env python3
"""
Test script for the art style comparer API
"""
import requests
import os
from PIL import Image
from io import BytesIO

# Test the example artist setup endpoint
def test_example_setup():
    print("Testing example artist setup endpoint...")
    response = requests.post("http://localhost:8000/example-artist-setup")
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

# Test the Exa API endpoint
def test_exa():
    print("Testing Exa API endpoint...")
    query = "Leonardo da Vinci painting style"
    response = requests.post("http://localhost:8000/test-exa", json={"query": query, "max_results": 2})
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

# Create a simple test image
def create_test_image():
    # Create a simple gradient image
    img = Image.new('RGB', (500, 500), color=(73, 109, 137))
    
    # Draw a gradient
    for x in range(500):
        for y in range(500):
            r = int(x / 2)
            g = int((x + y) / 4)
            b = int(y / 2)
            img.putpixel((x, y), (r, g, b))
    
    # Save to a BytesIO object
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes

# Test the compare-user-art endpoint
def test_compare_art():
    print("Testing compare-user-art endpoint...")
    img_bytes = create_test_image()
    
    # Create the multipart form data
    files = {'file': ('test_image.png', img_bytes, 'image/png')}
    data = {'top_k': 3, 'request_tips': True}
    
    response = requests.post("http://localhost:8000/compare-user-art", files=files, data=data)
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()
    
    # Check if we have matches
    if 'matches' in response.json():
        matches = response.json()['matches']
        print(f"Found {len(matches)} artist matches:")
        for i, match in enumerate(matches):
            print(f"Match #{i+1}: {match['artist_name']} (score: {match['score']})")
            if 'tips' in match and match['tips']:
                print(f"Tips: {match['tips'][:100]}...")
    else:
        print("No matches found. Make sure to run the scrape_and_ingest.py script first.")

if __name__ == "__main__":
    print("Testing the Art Style Comparer API...")
    
    # Run the tests
    test_example_setup()
    test_exa()
    test_compare_art()
    
    print("All tests completed!")