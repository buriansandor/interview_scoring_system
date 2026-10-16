"""Provider construction with the same bounded, validated scoring loop for all backends."""

from urllib.error import HTTPError, URLError

from local_llm import LocalLLMClient, OUTPUT_SCHEMA


class GeminiScoringClient(LocalLLMClient):
    def __init__(self, config, request_budget=None):
        super().__init__(config, request_budget=request_budget)
        from google import genai
        self.sdk = genai.Client(
            api_key=config.api_key,
            vertexai=False,
            http_options={
                "timeout": int(config.timeout * 1000),
                "retry_options": {"attempts": 1},
            },
        )

    def complete(self, messages):
        from google.genai import errors, types
        self._before_request()
        contents = [
            types.Content(
                role="model" if message["role"] == "assistant" else "user",
                parts=[types.Part.from_text(text=message["content"])],
            )
            for message in messages if message["role"] != "system"
        ]
        parameters = {
            "system_instruction": "\n\n".join(m["content"] for m in messages if m["role"] == "system"),
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
        }
        if self.config.json_mode != "none":
            parameters["response_mime_type"] = "application/json"
        if self.config.json_mode == "json_schema":
            parameters["response_json_schema"] = OUTPUT_SCHEMA
        try:
            response = self.sdk.models.generate_content(
                model=self.config.model, contents=contents,
                config=types.GenerateContentConfig(**parameters),
            )
        except errors.APIError as error:
            raise HTTPError(self.config.base_url, error.code or 500, "Gemini API request failed", {}, None) from None
        except Exception:
            # Do not include provider responses, API keys, or interview text in logs.
            raise URLError("Gemini request failed") from None
        if not isinstance(response.text, str) or not response.text.strip():
            raise ValueError("Gemini returned no text.")
        return response.text


def build_client(config, request_budget=None):
    if config.provider == "gemini":
        return GeminiScoringClient(config, request_budget=request_budget)
    return LocalLLMClient(config, request_budget=request_budget)
