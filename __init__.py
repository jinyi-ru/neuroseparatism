from .ns_qwen_cond import (
    TextEncodeQwenImageEditAdvanced,
    QwenProcessingParams,
    QwenMultiProcessingParams
)

# Регистрация типов данных для ComfyUI
class QwenParamsType:
    """Кастомный тип данных для параметров Qwen"""
    pass

class QwenMultiParamsType:
    """Кастомный тип данных для множественных параметров Qwen"""
    pass

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

# Регистрируем кастомные типы данных
NODE_CLASS_MAPPINGS["QWEN_PARAMS"] = QwenParamsType
NODE_CLASS_MAPPINGS["QWEN_MULTI_PARAMS"] = QwenMultiParamsType

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']