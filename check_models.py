# check_models.py

import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure with your API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# List available models
print("🔍 Available models:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"  - {m.name}")