import logging
from threading import Lock
from typing import TYPE_CHECKING, ClassVar, Dict, Union, cast

if TYPE_CHECKING:
    from src.core.types import ModelType


class UnifiedModelType(str):
    r"""Class used for support both :obj:`ModelType` and :obj:`str` to be used
    to represent a model type in a unified way. This class is a subclass of
    :obj:`str` so that it can be used as string seamlessly.

    Args:
        value (Union[ModelType, str]): The value of the model type.
    """

    _cache: ClassVar[Dict[str, "UnifiedModelType"]] = {}
    _lock: ClassVar[Lock] = Lock()

    def __new__(cls, value: Union["ModelType", str]) -> "UnifiedModelType":
        with cls._lock:
            if value not in cls._cache:
                instance = super().__new__(cls, value)
                cls._cache[value] = cast(UnifiedModelType, instance)
            else:
                instance = cls._cache[value]
        return instance

    def __init__(self, value: Union["ModelType", str]) -> None:
        pass

    @property
    def value_for_tiktoken(self) -> str:
        r"""Returns the model name for TikToken."""
        return "gpt-4o-mini"

    _token_limit_warned = False

    @property
    def token_limit(self) -> int:
        r"""Returns the token limit for the model. Here we set the default
        value as `999_999_999` if it's not provided from `model_config_dict`"""
        if not UnifiedModelType._token_limit_warned:
            logging.warning(
                "Invalid or missing `max_tokens` in `model_config_dict`. "
                "Defaulting to 999_999_999 tokens."
            )
            UnifiedModelType._token_limit_warned = True
        return 999_999_999

    # For ad-hoc string model types, we cannot determine the platform.
    # All is_* return False; platform is determined by ModelPlatformType.

    @property
    def is_openai(self) -> bool:
        return False

    @property
    def is_anthropic(self) -> bool:
        return False

    @property
    def is_azure_openai(self) -> bool:
        return False

    @property
    def is_groq(self) -> bool:
        return False

    @property
    def is_openrouter(self) -> bool:
        return False

    @property
    def is_zhipuai(self) -> bool:
        return False

    @property
    def is_gemini(self) -> bool:
        return False

    @property
    def is_mistral(self) -> bool:
        return False

    @property
    def is_reka(self) -> bool:
        return False

    @property
    def is_cohere(self) -> bool:
        return False

    @property
    def is_yi(self) -> bool:
        return False

    @property
    def is_qwen(self) -> bool:
        return False

    @property
    def is_internlm(self) -> bool:
        return False

    @property
    def is_moonshot(self) -> bool:
        return False

    @property
    def support_native_structured_output(self) -> bool:
        return False

    @property
    def support_native_tool_calling(self) -> bool:
        return False
