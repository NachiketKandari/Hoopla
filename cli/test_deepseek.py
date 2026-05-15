import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
api_key = os.environ.get("DEEPSEEK_API_KEY")
print(f"Using key {api_key[:6]}..." if api_key else "No DEEPSEEK_API_KEY found.")

if not api_key:
    print("Set DEEPSEEK_API_KEY in .env or environment.")
    exit(1)

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."}]
)
print(response.choices[0].message.content)
print(f"Prompt Tokens: {response.usage.prompt_tokens}")
print(f"Response Tokens: {response.usage.completion_tokens}")
