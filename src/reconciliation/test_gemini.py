import os

from dotenv import dotenv_values
from google import genai


# Load the .env file directly
config = dotenv_values("../../.env")

api_key = config.get("GEMINI_API_KEY")

if not api_key:
    raise EnvironmentError(
        "GEMINI_API_KEY was not found in .env"
    )


def main():

    client = genai.Client(
        api_key=api_key
    )

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=(
            "Explain this financial reconciliation exception "
            "in one short sentence: "
            "expected settlement ₹981.32, actual settlement "
            "₹0.00, difference ₹981.32."
        ),
    )

    print("\n========== GEMINI TEST ==========")
    print(response.text)
    print("=================================")


if __name__ == "__main__":
    main()