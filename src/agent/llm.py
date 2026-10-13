"""Pick the chat model from LLM_PROVIDER. API keys are read from the
environment by LangChain itself (GROQ_API_KEY, or AWS credentials/profile)."""

from config import settings


def get_llm(temperature: float = 0):
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=settings.groq_model, temperature=temperature, max_retries=6)  # retries on rate limits

    from langchain_aws import ChatBedrockConverse

    # No temperature here: newer Claude models on Bedrock reject sampling params.
    return ChatBedrockConverse(model=settings.bedrock_model_id, region_name=settings.aws_region, max_tokens=4096)
