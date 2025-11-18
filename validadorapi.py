import os
from google import genai

api_key = os.getenv("GEMINI_API_KEY")
print("API KEY =", api_key[:4] + "..." if api_key else "No detectada")

client = genai.Client(api_key=api_key)

try:
    r = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Hola"
    )
    print("OK =>", r.text)
except Exception as e:
    print("ERROR =>", e)
