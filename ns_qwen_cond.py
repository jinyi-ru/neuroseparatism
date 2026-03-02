"""Нода предобработки изображения для подачи в VL"""

# pylint: disable=invalid-name
# Контракт API ComfyUI на использование специальных методов
# в UPPER CASE

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import comfy.utils
import torch

from .strings.strings import text, prompts

# ============================================================================
# КОНСТАНТЫ
# ============================================================================

# Размеры и множественность
DEFAULT_VL_TARGET_SIZE: int = 384
DEFAULT_LATENT_TARGET_SIZE: int = 1024
DEFAULT_MAX_PIXELS: int = 4194304  # 2048x2048

# Границы значений
MIN_SIZE: int = 8
MAX_SIZE_VL: int = 2048
MAX_SIZE_LATENT: int = 4096
STEP_SIZE: int = 8
STEP_SIZE_LATENT: int = 32

# Границы для max_pixels
MIN_PIXELS: int = 131072  # 256x512
MAX_PIXELS_LIMIT: int = 16777216  # 4096x4096
PIXELS_STEP: int = 1024

# Требования к кратности
MULTIPLE_VL: int = 8
MULTIPLE_LATENT: int = 32

# Минимальный размер для VAE
MIN_VAE_SIZE: int = 32

# Индексы для работы с тензорами
CHANNELS_RGB: int = 3
DIM_BATCH: int = 0
DIM_HEIGHT: int = 1
DIM_WIDTH: int = 2
DIM_CHANNELS: int = 3

# ============================================================================
# МЕТОДЫ ОБРАБОТКИ (константы для выбора)
# ============================================================================

CROP_METHODS: List[str] = ["crop", "pad", "square_pad"]
RESIZE_METHODS: List[str] = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

DEFAULT_CROP_METHOD: str = "crop"
DEFAULT_RESIZE_METHOD: str = "lanczos"

# ============================================================================
# КАТЕГОРИЯ
# ============================================================================

QWEN_CATEGORY: str = "neuroseparatism"


class QwenProcessingParams:
    """Нода для создания параметров обработки изображений"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "vl_target_size": (
                    "INT",
                    {
                        "default": DEFAULT_VL_TARGET_SIZE,
                        "min": MIN_SIZE,
                        "max": MAX_SIZE_VL,
                        "step": STEP_SIZE,
                    },
                ),
                "latent_target_size": (
                    "INT",
                    {
                        "default": DEFAULT_LATENT_TARGET_SIZE,
                        "min": MIN_SIZE * 4,  # 32
                        "max": MAX_SIZE_LATENT,
                        "step": STEP_SIZE_LATENT,
                    },
                ),
                "crop_method": (CROP_METHODS, {"default": DEFAULT_CROP_METHOD}),
                "raw_mode": ("BOOLEAN", {"default": False}),
                "resize_method": (RESIZE_METHODS, {"default": DEFAULT_RESIZE_METHOD}),
                "foolproof_protection": ("BOOLEAN", {"default": False}),
                "max_pixels": (
                    "INT",
                    {
                        "default": DEFAULT_MAX_PIXELS,
                        "min": MIN_PIXELS,
                        "max": MAX_PIXELS_LIMIT,
                        "step": PIXELS_STEP,
                    },
                ),
            },
            "optional": {
                "params_name": ("STRING", {"default": "default", "multiline": False}),
            },
        }

    RETURN_TYPES = ("QWEN_PARAMS", "STRING")
    RETURN_NAMES = ("params", "info")
    FUNCTION = "create_params"
    CATEGORY = QWEN_CATEGORY
    DESCRIPTION = "Create processing parameters for Qwen nodes"

    def create_params(
        self,
        vl_target_size: int,
        latent_target_size: int,
        crop_method: str,
        raw_mode: bool,
        resize_method: str,
        foolproof_protection: bool,
        max_pixels: int,
        params_name: str = "default",
    ) -> Tuple[Dict[str, Any], str]:
        """Создает словарь параметров обработки"""
        params = {
            "name": params_name,
            "vl_target_size": vl_target_size,
            "latent_target_size": latent_target_size,
            "crop_method": crop_method,
            "raw_mode": raw_mode,
            "resize_method": resize_method,
            "foolproof_protection": foolproof_protection,
            "max_pixels": max_pixels,
        }

        info = (
            f"Params '{params_name}': VL={vl_target_size}, "
            f"Latent={latent_target_size}, Crop={crop_method}, "
            f"Raw={raw_mode}, Resize={resize_method}, "
            f"Foolproof={foolproof_protection}, "
            f"MaxPixels={max_pixels}"
        )

        return (params, info)


class QwenMultiProcessingParams:
    """Нода для создания индивидуальных параметров для каждого изображения"""

    MAX_IMAGES: int = 4

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        input_dict: Dict[str, Any] = {
            "required": {
                "image1_params": ("QWEN_PARAMS",),
            },
            "optional": {},
        }

        # Добавляем опциональные параметры для остальных изображений
        for i in range(2, cls.MAX_IMAGES + 1):
            input_dict["optional"][f"image{i}_params"] = (
                "QWEN_PARAMS",
                {"default": None},
            )

        return input_dict

    RETURN_TYPES = ("QWEN_MULTI_PARAMS", "STRING")
    RETURN_NAMES = ("multi_params", "info")
    FUNCTION = "combine_params"
    CATEGORY = QWEN_CATEGORY
    DESCRIPTION = "Combine multiple parameter sets for different images"

    def combine_params(
        self,
        image1_params: Dict[str, Any],
        image2_params: Optional[Dict[str, Any]] = None,
        image3_params: Optional[Dict[str, Any]] = None,
        image4_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """Объединяет параметры для нескольких изображений"""
        multi_params: Dict[str, Any] = {
            "image1": image1_params,
        }

        info_lines = ["Multi-processing parameters:"]
        info_lines.append(f"  Image1: {image1_params.get('name', 'default')}")

        # Словарь для удобного обхода
        params_map = {
            "image2": image2_params,
            "image3": image3_params,
            "image4": image4_params,
        }

        for img_key, params in params_map.items():
            if params is not None:
                multi_params[img_key] = params
                info_lines.append(
                    f"  {img_key.capitalize()}: {params.get('name', 'default')}"
                )

        return (multi_params, "\n".join(info_lines))


class TextEncodeQwenImageEditAdvanced:
    """Основная нода для кодирования изображений с Qwen"""

    # Ключи для параметров
    PARAM_VL_SIZE = "vl_target_size"
    PARAM_LATENT_SIZE = "latent_target_size"
    PARAM_CROP_METHOD = "crop_method"
    PARAM_RAW_MODE = "raw_mode"
    PARAM_RESIZE_METHOD = "resize_method"
    PARAM_FOOLPROOF = "foolproof_protection"
    PARAM_MAX_PIXELS = "max_pixels"
    PARAM_NAME = "name"

    # Токены для промпта
    VISION_START = "<|vision_start|>"
    VISION_END = "<|vision_end|>"
    IMAGE_PAD = "<|image_pad|>"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "image1": ("IMAGE",),
            },
            "optional": {
                "vae": ("VAE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "multi_params": ("QWEN_MULTI_PARAMS", {"default": None}),
                # Глобальные параметры (используются если multi_params не задан)
                "global_vl_size": (
                    "INT",
                    {
                        "default": DEFAULT_VL_TARGET_SIZE,
                        "min": MIN_SIZE,
                        "max": MAX_SIZE_VL,
                        "step": STEP_SIZE,
                    },
                ),
                "global_latent_size": (
                    "INT",
                    {
                        "default": DEFAULT_LATENT_TARGET_SIZE,
                        "min": MIN_SIZE * 4,
                        "max": MAX_SIZE_LATENT,
                        "step": STEP_SIZE_LATENT,
                    },
                ),
                "global_crop_method": (CROP_METHODS, {"default": DEFAULT_CROP_METHOD}),
                "global_resize_method": (
                    RESIZE_METHODS,
                    {"default": DEFAULT_RESIZE_METHOD},
                ),
                "global_raw_mode": ("BOOLEAN", {"default": False}),
                "global_foolproof_protection": ("BOOLEAN", {"default": False}),
                "global_max_pixels": (
                    "INT",
                    {
                        "default": DEFAULT_MAX_PIXELS,
                        "min": MIN_PIXELS,
                        "max": MAX_PIXELS_LIMIT,
                        "step": PIXELS_STEP,
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": text["default_sysprompt"],
                    },
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "INT", "INT", "STRING")
    RETURN_NAMES = ("conditioning", "orig_width", "orig_height", "info")
    FUNCTION = "encode"
    CATEGORY = QWEN_CATEGORY
    DESCRIPTION = text["node_description"]

    def encode(
        self,
        clip: Any,
        prompt: str,
        image1: torch.Tensor,
        vae: Optional[Any] = None,
        image2: Optional[torch.Tensor] = None,
        image3: Optional[torch.Tensor] = None,
        image4: Optional[torch.Tensor] = None,
        multi_params: Optional[Dict[str, Any]] = None,
        global_vl_size: int = DEFAULT_VL_TARGET_SIZE,
        global_latent_size: int = DEFAULT_LATENT_TARGET_SIZE,
        global_crop_method: str = DEFAULT_CROP_METHOD,
        global_resize_method: str = DEFAULT_RESIZE_METHOD,
        global_raw_mode: bool = False,
        global_foolproof_protection: bool = False,
        global_max_pixels: int = DEFAULT_MAX_PIXELS,
        system_prompt: str = "",
    ) -> Tuple[Union[List, Any], int, int, str]:
        """Основной метод кодирования"""

        # Сохраняем оригинальные размеры первого изображения
        batch_size, height, width, channels = image1.shape
        orig_width = width
        orig_height = height

        # Формируем список всех изображений
        images = [img for img in [image1, image2, image3, image4] if img is not None]

        info_lines = [
            f"Processing {len(images)} image(s)",
            f"Original size of image1: {orig_width}x{orig_height}",
        ]

        # Подготавливаем системный промпт
        if not system_prompt:
            system_prompt = text["default_sysprompt"]

        # Форматируем шаблон с системным промптом
        llama_template = prompts["system"].format(system_prompt)

        ref_latents = []
        images_vl = []
        image_prompt_parts = []

        for i, image in enumerate(images, start=1):
            # Получаем параметры для текущего изображения
            params = self._get_image_params(
                i,
                multi_params,
                {
                    self.PARAM_VL_SIZE: global_vl_size,
                    self.PARAM_LATENT_SIZE: global_latent_size,
                    self.PARAM_CROP_METHOD: global_crop_method,
                    self.PARAM_RAW_MODE: global_raw_mode,
                    self.PARAM_RESIZE_METHOD: global_resize_method,
                    self.PARAM_FOOLPROOF: global_foolproof_protection,
                    self.PARAM_MAX_PIXELS: global_max_pixels,
                },
            )

            info_lines.append(f"\nImage {i} ({params['source']} params):")
            info_lines.append(
                f"  Original: {image.shape[DIM_WIDTH]}x{image.shape[DIM_HEIGHT]}"
            )

            # Форматируем строку параметров
            params_str = text["params_vl"].format(
                params[self.PARAM_VL_SIZE],
                params[self.PARAM_LATENT_SIZE],
                params[self.PARAM_CROP_METHOD],
                params[self.PARAM_RAW_MODE],
                params[self.PARAM_FOOLPROOF],
            )
            info_lines.append(params_str)

            # 1. Обработка для VL
            vl_info = self._process_image(
                image=image,
                target_size=params[self.PARAM_VL_SIZE],
                resize_method=params[self.PARAM_RESIZE_METHOD],
                crop_method=params[self.PARAM_CROP_METHOD],
                target_multiple=MULTIPLE_VL,
                raw_mode=False,  # Для VL всегда обрабатываем
                purpose="VL",
                foolproof_protection=params[self.PARAM_FOOLPROOF],
                max_pixels=params[self.PARAM_MAX_PIXELS],
            )
            images_vl.append(vl_info["image"])
            info_lines.append(f"  VL result: {vl_info['info']}")

            # 2. Обработка для VAE (если есть)
            if vae is not None:
                latent_info = self._process_image(
                    image=image,
                    target_size=params[self.PARAM_LATENT_SIZE],
                    resize_method=params[self.PARAM_RESIZE_METHOD],
                    crop_method=params[self.PARAM_CROP_METHOD],
                    target_multiple=MULTIPLE_LATENT,
                    raw_mode=params[self.PARAM_RAW_MODE],
                    purpose="VAE",
                    foolproof_protection=params[self.PARAM_FOOLPROOF],
                    max_pixels=params[self.PARAM_MAX_PIXELS],
                )

                # Кодируем в латент
                with torch.no_grad():
                    latent = vae.encode(latent_info["image"][:, :, :, :CHANNELS_RGB])
                ref_latents.append(latent)
                info_lines.append(f"  VAE result: {latent_info['info']}")

            # Добавляем в промпт
            image_prompt_parts.append(
                f"Picture {i}: {self.VISION_START}{self.IMAGE_PAD}{self.VISION_END}"
            )

        # Формируем полный промпт
        full_prompt = "".join(image_prompt_parts) + prompt

        # Токенизация и кодирование
        try:
            tokens = clip.tokenize(
                full_prompt, images=images_vl, llama_template=llama_template
            )
        except Exception as e:
            info_lines.append(f"\nWarning: Could not use llama_template: {e}")
            tokens = clip.tokenize(full_prompt, images=images_vl)

        # Кодирование
        if hasattr(clip, "encode_from_tokens_scheduled"):
            conditioning = clip.encode_from_tokens_scheduled(tokens)
        else:
            conditioning = clip.encode_from_tokens(tokens)

        # Добавляем reference_latents если есть
        if ref_latents:
            conditioning = self._add_reference_latents(conditioning, ref_latents)
            info_lines.append(f"\nAdded {len(ref_latents)} reference latent(s)")

        # Формируем информационную строку
        info_text = "\n".join(info_lines)

        return (conditioning, orig_width, orig_height, info_text)

    def _get_image_params(
        self,
        image_index: int,
        multi_params: Optional[Dict[str, Any]],
        global_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Получает параметры для изображения по индексу.
        Возвращает словарь с параметрами и источником.
        """
        if multi_params is not None and f"image{image_index}" in multi_params:
            params = multi_params[f"image{image_index}"].copy()
            params["source"] = "individual"
        else:
            params = global_params.copy()
            params["source"] = "global"

        return params

    def _add_reference_latents(
        self, conditioning: Union[List, Any], ref_latents: List[torch.Tensor]
    ) -> Union[List, Any]:
        """Добавляет reference latents в conditioning"""
        if isinstance(conditioning, list):
            for c in conditioning:
                if len(c) > 1 and isinstance(c[1], dict):
                    c[1]["reference_latents"] = ref_latents
        else:
            conditioning = [conditioning]
            if len(conditioning[0]) > 1 and isinstance(conditioning[0][1], dict):
                conditioning[0][1]["reference_latents"] = ref_latents
            else:
                # Если структура неожиданная, создаем новую
                conditioning = [(conditioning[0], {"reference_latents": ref_latents})]

        return conditioning

    def _process_image(
        self,
        image: torch.Tensor,
        target_size: int,
        resize_method: str,
        crop_method: str,
        target_multiple: int,
        raw_mode: bool,
        purpose: str = "VL",
        foolproof_protection: bool = False,
        max_pixels: int = DEFAULT_MAX_PIXELS,
    ) -> Dict[str, Any]:
        """Универсальная обработка изображения"""
        batch_size, height, width, channels = image.shape

        # Создаем копию для обработки [B, C, H, W]
        image_tensor = image.movedim(-1, 1)

        # Флаг, указывающий на то, что было применено уменьшение из-за защиты
        foolproof_applied = False
        foolproof_scale = 1.0

        # ПРИМЕНЕНИЕ ЗАЩИТЫ ОТ ДУРАКА
        if self._should_apply_foolproof(purpose, raw_mode, foolproof_protection):
            current_pixels = width * height
            if current_pixels > max_pixels:
                # Вычисляем коэффициент уменьшения
                scale_factor = math.sqrt(max_pixels / current_pixels)

                # Вычисляем новые размеры
                new_width = max(MIN_VAE_SIZE, int(width * scale_factor))
                new_height = max(MIN_VAE_SIZE, int(height * scale_factor))

                # Применяем уменьшение
                image_tensor = comfy.utils.common_upscale(
                    image_tensor, new_width, new_height, resize_method, "disabled"
                )

                # Обновляем размеры
                width, height = new_width, new_height
                foolproof_applied = True
                foolproof_scale = scale_factor

        # Определяем целевые размеры
        if raw_mode and purpose == "VAE":
            new_width, new_height = self._make_multiple(
                width, height, target_multiple, crop_method
            )
            info = self._format_raw_info(
                width, height, new_width, new_height, foolproof_applied, foolproof_scale
            )
        else:
            new_width, new_height, info = self._resize_to_target(
                width, height, target_size, target_multiple, crop_method
            )

        # Применяем изменение размера если нужно
        if new_width != width or new_height != height:
            image_tensor = comfy.utils.common_upscale(
                image_tensor, new_width, new_height, resize_method, "disabled"
            )

        # Возвращаем в исходный формат [B, H, W, C]
        result_image = image_tensor.movedim(1, -1)

        return {"image": result_image, "info": info}

    def _should_apply_foolproof(
        self, purpose: str, raw_mode: bool, foolproof_protection: bool
    ) -> bool:
        """Проверяет, нужно ли применять защиту от дурака"""
        return purpose == "VAE" and raw_mode and foolproof_protection

    def _format_raw_info(
        self,
        orig_width: int,
        orig_height: int,
        new_width: int,
        new_height: int,
        foolproof_applied: bool,
        foolproof_scale: float,
    ) -> str:
        """Форматирует информационное сообщение для raw режима"""
        info_parts = []
        if foolproof_applied:
            info_parts.append(f"foolproof(x{foolproof_scale:.2f})")

        info_parts.append(f"{orig_width}x{orig_height} -> {new_width}x{new_height}")

        if not foolproof_applied:
            scale_w = new_width / orig_width
            scale_h = new_height / orig_height
            info_parts.append(f"(raw, x{scale_w:.2f}, x{scale_h:.2f})")

        return " ".join(info_parts)

    def _resize_to_target(
        self,
        width: int,
        height: int,
        target_size: int,
        target_multiple: int,
        crop_method: str,
    ) -> Tuple[int, int, str]:
        """Масштабирует изображение к целевому размеру с сохранением пропорций"""
        if width >= height:
            # Ширина больше или равна высоте
            scale = target_size / width
            new_width = target_size
            new_height = round(height * scale)
        else:
            # Высота больше ширины
            scale = target_size / height
            new_height = target_size
            new_width = round(width * scale)

        # Корректируем до кратности
        new_width, new_height = self._make_multiple(
            new_width, new_height, target_multiple, crop_method
        )

        scale_w = new_width / width
        scale_h = new_height / height
        info = f"{width}x{height} -> {new_width}x{new_height} (x{scale_w:.2f}, x{scale_h:.2f})"

        return new_width, new_height, info

    def _make_multiple(
        self, width: int, height: int, multiple: int, method: str = "crop"
    ) -> Tuple[int, int]:
        """Приведение размеров к кратности multiple выбранным методом"""
        if method == "crop":
            # Обрезка до ближайшего меньшего кратного значения
            new_width = max(multiple, (width // multiple) * multiple)
            new_height = max(multiple, (height // multiple) * multiple)

        elif method == "pad":
            # Дополнение до ближайшего большего кратного значения
            new_width = ((width + multiple - 1) // multiple) * multiple
            new_height = ((height + multiple - 1) // multiple) * multiple

        elif method == "square_pad":
            # Делаем квадрат с дополнением по большей стороне
            max_side = max(width, height)
            new_side = ((max_side + multiple - 1) // multiple) * multiple
            new_width = new_side
            new_height = new_side

        else:
            new_width = width
            new_height = height

        return new_width, new_height
