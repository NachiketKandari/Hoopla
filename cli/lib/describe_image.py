import os
import base64
import mimetypes
from dotenv import load_dotenv
import logging
from .search_utils import PROJECT_ROOT, get_llm_client, generate_text

logger = logging.getLogger(__name__)

load_dotenv()
_client, _model, _provider = get_llm_client()


def _generate(prompt: str) -> str:
    return generate_text(prompt, _client, _model, _provider)


def read_img_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def describe_image(query: str, image: str):

    mime, _ = mimetypes.guess_type(image)
    mime = mime or "image/jpeg"

    system_prompt = f"""
    Given the included image and text query, rewrite the text query to improve search results from a    movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary
    """

    if _provider == "gemini":
        from google import genai

        with open(image, "rb") as f:
            img = f.read()

        parts = [
            system_prompt,
            genai.types.Part.from_bytes(data=img, mime_type=mime),
            query.strip(),
        ]

        response = _client.models.generate_content(model=_model, contents=parts)
        return type('Response', (), {
            'text': response.text,
            'usage_metadata': response.usage_metadata
        })()
    else:
        img_b64 = read_img_base64(image)
        data_uri = f"data:{mime};base64,{img_b64}"

        response = _client.chat.completions.create(
            model=_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": query.strip()},
                ]
            }]
        )

        return type('Response', (), {
            'text': response.choices[0].message.content,
            'usage_metadata': response.usage
        })()


def describe_image_command(query: str, image: str) -> None:
    image_path = os.path.join(PROJECT_ROOT, image)
    response = describe_image(query, image_path)

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        token_info = response.usage_metadata
        if hasattr(token_info, 'total_token_count'):
            print(f"Total tokens:    {token_info.total_token_count}")
        elif hasattr(token_info, 'total_tokens'):
            print(f"Total tokens:    {token_info.total_tokens}")
