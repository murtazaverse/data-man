# WILL LATER DO THIS VIA JEV

from langchain.chat_models import init_chat_model


def pick_llm(level: str, temperature: int = 0, provider: str = "groq"):
    """
    This function takes in the level and temperature (optional),
    and returns the relevant model.
    """
    if provider == "google":
        if level == "low":
            model_name = f"{provider}_genai:gemini-3.1-flash-lite"
        elif level == "medium":
            model_name = f"{provider}_genai:gemini-3.5-flash-lite"
        elif level == "high":
            model_name = f"{provider}_genai:gemini-3.8-flash-lite-tts"
        else:
            raise ValueError("Invalid Level!")
    elif provider == "groq":
        if level == "low":
            model_name = f"{provider}:openai/gpt-oss-20b"
        elif level == "medium":
            model_name = f"{provider}:gemini-3.5-flash-lite"
        elif level == "high":
            model_name = f"{provider}:gemini-3.8-flash-lite-tts"
        else:
            raise ValueError("Invalid Level!")
    elif provider == "openai":
        if level == "low":
            model_name = f"{provider}:gpt-5-nano"
        elif level == "medium":
            model_name = f"{provider}:gpt-5.4-nano"
        elif level == "high":
            model_name = f"{provider}:gpt-5-mini"
        else:
            raise ValueError("Invalid Level!")
    model = init_chat_model(model_name, temperature=temperature)
    return model


if __name__ == "__main__":
    model = pick_llm("low")
    response = model.invoke("Are you working fine?")
    print(response)
