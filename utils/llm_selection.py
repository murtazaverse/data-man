# WILL LATER DO THIS VIA JEV

from langchain.chat_models import init_chat_model

def pick_llm(level: str, temperature: int = 0):
    """
    This function takes in the level and temperature (optional), 
    and returns the relevant model.
    """
    if level == "low":
        model_name = "google_genai:gemini-3.1-flash-lite"
    elif level == "medium":
        model_name = "google_genai:gemini-3.5-flash-lite"
    elif level == "high":
        model_name = "google_genai:gemini-3.8-flash-lite-tts"
    else:
        raise ValueError("Invalid Level!")
    model = init_chat_model(model_name, temperature=temperature)
    return model

if __name__ == "__main__":
    model = pick_llm("low")
    response=model.invoke("Are you working fine?")
    print(response)