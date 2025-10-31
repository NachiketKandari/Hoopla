import os
import mimetypes
from dotenv import load_dotenv
from google import genai
import logging
from lib.search_utils import PROJECT_ROOT

logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv("gemini_api_key")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"

def read_img(image):
    with open(image, "rb") as f:
        img = f.read()
        return img

def describe_image(query: str, image: str): 
    
    mime, _ = mimetypes.guess_type(image)
    mime = mime or "image/jpeg"

    img = read_img(image)

    system_prompt = f"""
    Given the included image and text query, rewrite the text query to improve search results from a    movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary
    """
    
    parts = [
    system_prompt,
    genai.types.Part.from_bytes(data=img, mime_type=mime),
    query.strip(),
    ]

    response = client.models.generate_content(model=model, contents=parts)

    return response



def describe_image_command(query: str, image: str) -> None:
    image_path = os.path.join(PROJECT_ROOT, image)
    response = describe_image(query, image_path)

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")
