"""Validated JSON configuration; secrets stay in environment variables or .key files."""

from dataclasses import dataclass, field, replace
import json
import math
import os
from pathlib import Path

from local_llm import LocalLLMConfig


PROVIDERS = {
    "local": {
        "base_url": "http://127.0.0.1:8080/v1", "model": "interview-scorer",
        "api_key_env": "LOCAL_LLM_API_KEY", "api_key_file": None,
        "json_mode": "json_object", "requests_per_minute": 0,
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com",
        "model": "gemini-2.5-flash", "api_key_env": "GEMINI_API_KEY",
        "api_key_file": "gemini.key", "json_mode": "json_schema",
        "requests_per_minute": 5,
    },
    "mistral_free": {
        "base_url": "https://api.mistral.ai/v1", "model": "mistral-small-latest",
        "api_key_env": "MISTRAL_API_KEY", "api_key_file": "mistral.key",
        "json_mode": "json_object", "requests_per_minute": 5,
    },
    "groq_free": {
        "base_url": "https://api.groq.com/openai/v1", "model": "openai/gpt-oss-120b",
        "api_key_env": "GROQ_API_KEY", "api_key_file": "groq.key",
        "json_mode": "json_schema", "requests_per_minute": 10,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1", "model": "openrouter/free",
        "api_key_env": "OPENROUTER_API_KEY", "api_key_file": "openrouter.key",
        "json_mode": "json_object", "requests_per_minute": 10,
    },
}


@dataclass(frozen=True)
class ReviewSettings:
    confidence_threshold: float = 0.5
    primary_weight: float = 0.5
    secondary_weight: float = 0.5
    disagreement_threshold: float = 0.25

    def __post_init__(self):
        for name in ("confidence_threshold", "disagreement_threshold"):
            _number(getattr(self, name), name, 0, maximum=1)
        for name in ("primary_weight", "secondary_weight"):
            _number(getattr(self, name), name, 0)
        if max(self.primary_weight, self.secondary_weight) == 0:
            raise ValueError("At least one review weight must be positive.")

    @property
    def normalized_weights(self):
        # Avoid overflow even for finite, very large configured weights.
        scale = max(self.primary_weight, self.secondary_weight)
        primary, secondary = self.primary_weight / scale, self.secondary_weight / scale
        return primary / (primary + secondary), secondary / (primary + secondary)


@dataclass(frozen=True)
class AppConfig:
    llm: LocalLLMConfig
    directory: Path
    config_path: Path | None = None
    input_csv: Path | None = None
    rules_json: Path | None = None
    output_csv: Path | None = None
    api_key_env: str | None = None
    review: ReviewSettings = field(default_factory=ReviewSettings)
    secondary_llm: LocalLLMConfig | None = None
    secondary_api_key_env: str | None = None
    secondary_api_key_file: Path | None = None


def _object(value, label, allowed):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {label} fields: {', '.join(sorted(unknown))}")


def _number(value, name, minimum, integer=False, maximum=None):
    valid_type = type(value) is int if integer else type(value) in (int, float)
    if not valid_type or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be a finite {'integer' if integer else 'number'} >= {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}.")
    return value


def _path(value, directory, label):
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty path string or null.")
    path = Path(value).expanduser()
    return (directory / path).resolve() if not path.is_absolute() else path.resolve()


def _secret(profile, directory, required, secret_getter=None):
    env_name = profile["api_key_env"]
    key_file = profile["api_key_file"]
    if not isinstance(env_name, str) or not env_name:
        raise ValueError("api_key_env must be a nonempty environment-variable name.")
    path = _path(key_file, directory, "api_key_file")
    key = os.getenv(env_name, "").strip()
    if not key and path and path.is_file():
        key = path.read_text(encoding="utf-8-sig").strip()
    if not key and required and secret_getter is not None:
        try:
            value = secret_getter(env_name)
        except Exception:
            # Secret backends may include credentials in error messages.
            raise ValueError(f"Cannot read the selected provider's secret: {env_name}.") from None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"The selected provider's secret is empty: {env_name}.")
        key = value.strip()
    if required and not key:
        raise ValueError(f"Missing API key. Set {env_name} or create the selected provider's .key file.")
    return key


def _role_profile(role, profiles, label):
    if not isinstance(role, dict):
        raise ValueError(f"{label} must be an object with a provider.")
    provider = role.get("provider")
    if not isinstance(provider, str) or provider not in PROVIDERS:
        raise ValueError(f"{label}.provider must name a supported provider.")
    overrides = {key: value for key, value in role.items() if key != "provider"}
    allowed = {"model", "api_key_env", "api_key_file", "json_mode", "requests_per_minute"}
    if provider == "local":
        allowed.add("base_url")
    _object(overrides, label, allowed)
    profile = {**PROVIDERS[provider], **profiles.get(provider, {}), **overrides}
    for key in ("model", "base_url", "api_key_env", "json_mode"):
        if not isinstance(profile[key], str) or not profile[key].strip():
            raise ValueError(f"{label}.{key} must be a nonempty string.")
    _path(profile["api_key_file"], Path("."), "api_key_file")
    _number(profile["requests_per_minute"], "requests_per_minute", 0)
    return provider, profile


def load_config(path="llm_config.json", *, require_file=False, secret_getter=None):
    """JSON wins over legacy LLM_* variables; missing explicitly requested files fail."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        if require_file:
            raise ValueError("The requested LLM configuration file does not exist.")
        return _legacy_config(path.parent)
    with path.open(encoding="utf-8-sig") as file:
        data = json.load(file)
    _object(data, "configuration", {
        "provider", "primary", "secondary", "review", "providers", "settings",
        "input_csv", "rules_json", "output_csv",
    })
    if "provider" in data and "primary" in data:
        raise ValueError("Use primary or the legacy provider field, not both.")
    profiles = data.get("providers", {})
    _object(profiles, "providers", PROVIDERS)
    for name, value in profiles.items():
        allowed = {"model", "api_key_env", "api_key_file", "json_mode", "requests_per_minute"}
        if name == "local":
            allowed.add("base_url")
        _object(value, f"providers.{name}", allowed)
        for field in ("model", "api_key_env", "json_mode", "base_url"):
            if field in value and (not isinstance(value[field], str) or not value[field].strip()):
                raise ValueError(f"providers.{name}.{field} must be a nonempty string.")
        if "api_key_file" in value:
            _path(value["api_key_file"], path.parent, "api_key_file")
        if "requests_per_minute" in value:
            _number(value["requests_per_minute"], "requests_per_minute", 0)
        if "json_mode" in value and value["json_mode"] not in {"none", "json_object", "json_schema"}:
            raise ValueError("json_mode must be none, json_object, or json_schema.")
    provider, profile = _role_profile(
        data.get("primary", {"provider": data.get("provider", "local")}), profiles, "primary",
    )
    secondary_role = data.get("secondary")
    secondary_provider, secondary_profile = (None, None)
    if secondary_role is not None:
        secondary_provider, secondary_profile = _role_profile(secondary_role, profiles, "secondary")
        if (
            secondary_profile["api_key_env"] == profile["api_key_env"]
            and _path(secondary_profile["api_key_file"], path.parent, "api_key_file")
            != _path(profile["api_key_file"], path.parent, "api_key_file")
        ):
            raise ValueError("Different primary/secondary key files require distinct api_key_env names.")
    review_data = data.get("review", {})
    _object(review_data, "review", {
        "confidence_threshold", "primary_weight", "secondary_weight", "disagreement_threshold",
    })
    review = ReviewSettings(**review_data)
    settings = data.get("settings", {})
    _object(settings, "settings", {
        "timeout", "max_tokens", "max_attempts", "max_reflections",
        "temperature", "max_requests_per_run",
    })
    timeout = _number(settings.get("timeout", 120), "timeout", 0.001)
    max_tokens = _number(settings.get("max_tokens", 1024), "max_tokens", 1, integer=True)
    max_attempts = _number(settings.get("max_attempts", 3), "max_attempts", 1, integer=True)
    max_reflections = _number(settings.get("max_reflections", 1), "max_reflections", 0, integer=True)
    temperature = _number(settings.get("temperature", 0.1), "temperature", 0, maximum=2)
    max_requests = _number(settings.get("max_requests_per_run", 50), "max_requests_per_run", 1, integer=True)
    key = _secret(profile, path.parent, required=provider != "local", secret_getter=secret_getter)
    llm = LocalLLMConfig(
        base_url=profile["base_url"].strip(), model=profile["model"].strip(),
        api_key=key, provider=provider, json_mode=profile["json_mode"],
        timeout=timeout, max_tokens=max_tokens, max_attempts=max_attempts,
        max_reflections=max_reflections, temperature=temperature,
        requests_per_minute=profile["requests_per_minute"],
        max_requests_per_run=max_requests,
    )
    secondary_llm = None
    if secondary_profile is not None:
        secondary_llm = replace(
            llm, provider=secondary_provider, model=secondary_profile["model"].strip(),
            base_url=secondary_profile["base_url"].strip(), api_key="",
            json_mode=secondary_profile["json_mode"],
            requests_per_minute=secondary_profile["requests_per_minute"],
        )
    input_csv = _path(data.get("input_csv"), path.parent, "input_csv")
    rules_json = _path(data.get("rules_json"), path.parent, "rules_json")
    output_csv = _path(data.get("output_csv", "scored_answers.csv"), path.parent, "output_csv")
    if output_csv is None:
        raise ValueError("output_csv must not be null.")
    if output_csv in {input_csv, rules_json, path} or output_csv.suffix.lower() != ".csv":
        raise ValueError("output_csv must be a CSV path distinct from input and configuration files.")
    secret_paths = {
        _path(p.get("api_key_file", PROVIDERS[n]["api_key_file"]), path.parent, "api_key_file")
        for n, p in profiles.items()
    }
    secret_paths.add(_path(profile["api_key_file"], path.parent, "api_key_file"))
    if secondary_profile:
        secret_paths.add(_path(secondary_profile["api_key_file"], path.parent, "api_key_file"))
    if output_csv in secret_paths:
        raise ValueError("output_csv must not overwrite an API-key file.")
    return AppConfig(
        llm, path.parent, path, input_csv, rules_json, output_csv, profile["api_key_env"],
        review=review, secondary_llm=secondary_llm,
        secondary_api_key_env=secondary_profile["api_key_env"] if secondary_profile else None,
        secondary_api_key_file=(
            _path(secondary_profile["api_key_file"], path.parent, "api_key_file")
            if secondary_profile else None
        ),
    )


def resolve_secondary_config(config, *, secret_getter=None):
    """Resolve an explicitly configured secondary only when it is needed."""
    if config.secondary_llm is None:
        return config.llm
    profile = {
        "api_key_env": config.secondary_api_key_env,
        "api_key_file": str(config.secondary_api_key_file) if config.secondary_api_key_file else None,
    }
    key = _secret(
        profile, config.directory, required=config.secondary_llm.provider != "local",
        secret_getter=secret_getter,
    )
    return replace(config.secondary_llm, api_key=key)


def _legacy_config(directory):
    """Preserve no-config local environment usage; never guess a cloud key's owner."""
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider == "local":
        llm = LocalLLMConfig.from_env()
    elif provider == "gemini":
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            file = directory / "gemini.key"
            if file.is_file():
                key = file.read_text(encoding="utf-8-sig").strip()
        if not key:
            raise ValueError("Missing GEMINI_API_KEY. Create llm_config.json to select a different provider.")
        llm = LocalLLMConfig(
            provider="gemini", api_key=key, model=PROVIDERS["gemini"]["model"],
            base_url=PROVIDERS["gemini"]["base_url"], json_mode="json_schema",
        )
    else:
        raise ValueError("Create llm_config.json to select groq_free, mistral_free, or openrouter.")
    return AppConfig(
        replace(llm, max_requests_per_run=50), directory,
        output_csv=directory / "scored_answers.csv",
    )
