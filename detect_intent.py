import os
from langchain_openai import AzureChatOpenAI

# Initialize the LLM (no session needed)
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment="gpt-4o",
    api_version="2023-09-01-preview",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.0
)

def detect_intent(user_input: str) -> str:
    """
    Classify the user's query as 'medical' or 'appointment'.
    No session or conversation memory is used here.
    """

    classification_prompt = f"""
    You are a classification agent. The user said: "{user_input}"

    Classify this query into exactly one of two categories:
    1) "medical" if the user is asking about symptoms, diseases, treatments, medications, or general health.
    2) "appointment" if the user is asking about booking, rescheduling, or canceling appointments, or viewing appointments.

    Return ONLY one word: either "medical" or "appointment".
    """

    try:
        response = llm.invoke(classification_prompt).content.strip().lower()
        if response.startswith("medical"):
            return "medical"
        return "appointment"
    except Exception as e:
        print("LLM Classification Error:", e)
        # Default to 'appointment' on error
        return "appointment"
