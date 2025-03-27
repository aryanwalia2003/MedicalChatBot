import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# Load environment variables
load_dotenv()

# Azure OpenAI Initialization
medical_ai = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment="gpt-4o",
    api_version="2023-09-01-preview",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.7
)

def handle_medical_query(user_input):
    """
    Processes medical-related queries and provides an AI-generated response.
    """

    prompt = f"""
    You are Dr. AI, a licensed medical professional. Provide factual and helpful medical information in clear and concise language. If a question is outside your medical knowledge, politely state that you cannot answer. Use plain text only, without special characters or markdown formatting.

    **User's Question:** {user_input}
    
    **Your Answer:**
    """

    try:
        response = medical_ai.invoke(prompt).content.strip()
        return response

    except Exception as e:
        return f"Error processing medical query: {str(e)}"
