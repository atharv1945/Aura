import os
import google.generativeai as genai
from dotenv import load_dotenv

def test_api_key():
    """
    A simple, standalone script to test if the Google Gemini API key is valid.
    It reads the key from the .env file in the project root.
    """
    print("--- Testing Google Gemini API Key ---")

    # 1. Load the API key from the .env file
    # This looks for d:\Aura\.env
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("\n[FAILURE] ❌ GOOGLE_API_KEY not found in your .env file or environment variables.")
        print("Please make sure you have a .env file in the project root (d:\\Aura\\.env) with the line:")
        print('GOOGLE_API_KEY="YOUR_API_KEY_HERE"')
        return

    print("API Key found. Attempting to connect to Google AI services...")

    try:
        # 2. Configure the generative AI client
        genai.configure(api_key=api_key)

        # 3. Make a simple, low-cost API call to list models. This will fail if the key is invalid.
        genai.list_models()
        print("\n[SUCCESS] ✅ API Key is valid! You can now run the explainability engine.")

    except Exception as e:
        print(f"\n[FAILURE] ❌ An error occurred. This almost always means the API key is invalid, expired, or has a typo.")
        print(f"   Error details: {e}")

if __name__ == "__main__":
    test_api_key()