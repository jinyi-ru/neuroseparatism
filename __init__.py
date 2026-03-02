from .ns_qwen_cond import (
    TextEncodeQwenImageEditAdvanced,
    QwenProcessingParams,
    QwenMultiProcessingParams,
)

# Маппинг типов для ComfyUI
NODE_CLASS_MAPPINGS = {
    "TextEncodeQwenImageEditAdvanced": TextEncodeQwenImageEditAdvanced,
    "QwenProcessingParams": QwenProcessingParams,
    "QwenMultiProcessingParams": QwenMultiProcessingParams,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeQwenImageEditAdvanced": "🔧 Qwen Image Edit Advanced",
    "QwenProcessingParams": "⚙️ Qwen Processing Parameters",
    "QwenMultiProcessingParams": "⚙️ Qwen Multi-Image Parameters",
}

# НЕ РЕГИСТРИРУЕМ QWEN_PARAMS и QWEN_MULTI_PARAMS как ноды!
# Это просто типы данных, они не должны быть в NODE_CLASS_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
