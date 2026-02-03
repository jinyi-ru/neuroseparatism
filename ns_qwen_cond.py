"""Нода предобработки изображения для подачи в VL"""

# pylint: disable=invalid-name
# Контракт API ComfyUI на использование специальных методов
# в UPPER CASE

# import json
import math

from typing import Any, Dict, List, Optional, LiteralString

import comfy.utils
import torch

from strings.strings import text, prompts

MAX_PIXELS = 4194304

# Определяем категорию для всех Qwen нод
QWEN_CATEGORY = "neuroseparatism"


class QwenProcessingParams:
    """Нода для создания параметров обработки изображений"""

    @classmethod
    def INPUT_TYPES(s) -> dict[str, Any]:
        return {
            "required": {
                "vl_target_size": (
                    "INT",
                    {"default": 384, "min": 8, "max": 2048, "step": 8},
                ),
                "latent_target_size": (
                    "INT",
                    {"default": 1024, "min": 32, "max": 4096, "step": 32},
                ),
                "crop_method": (["crop", "pad", "square_pad"], {"default": "crop"}),
                "raw_mode": ("BOOLEAN", {"default": False}),
                "resize_method": (
                    ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],
                    {"default": "lanczos"},
                ),
                "foolproof_protection": (
                    "BOOLEAN",
                    {"default": False},
                ),  # Защита от дурака
                "max_pixels": (
                    "INT",
                    {
                        "default": MAX_PIXELS,
                        "min": 131072,
                        "max": 16777216,
                        "step": 1024,
                    },
                ),  # 2048x2048 по умолчанию
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
        vl_target_size,
        latent_target_size,
        crop_method,
        raw_mode,
        resize_method,
        foolproof_protection,
        max_pixels,
        params_name="default",
    ) -> tuple[dict[str, Any], str]:
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

    @classmethod
    def INPUT_TYPES(s) -> dict[str, Any]:
        return {
            "required": {
                "image1_params": ("QWEN_PARAMS",),
            },
            "optional": {
                "image2_params": ("QWEN_PARAMS", {"default": None}),
                "image3_params": ("QWEN_PARAMS", {"default": None}),
                "image4_params": ("QWEN_PARAMS", {"default": None}),
            },
        }

    RETURN_TYPES = ("QWEN_MULTI_PARAMS", "STRING")
    RETURN_NAMES = ("multi_params", "info")
    FUNCTION = "combine_params"
    CATEGORY = QWEN_CATEGORY
    DESCRIPTION = "Combine multiple parameter sets for different images"

    def combine_params(
        self, image1_params, image2_params=None, image3_params=None, image4_params=None
    ) -> tuple[dict[str, Any], str]:
        multi_params = {
            "image1": image1_params,
        }

        info_lines = ["Multi-processing parameters:"]
        info_lines.append(f"  Image1: {image1_params.get('name', 'default')}")

        if image2_params is not None:
            multi_params["image2"] = image2_params
            info_lines.append(f"  Image2: {image2_params.get('name', 'default')}")

        if image3_params is not None:
            multi_params["image3"] = image3_params
            info_lines.append(f"  Image3: {image3_params.get('name', 'default')}")

        if image4_params is not None:
            multi_params["image4"] = image4_params
            info_lines.append(f"  Image4: {image4_params.get('name', 'default')}")

        return (multi_params, "\n".join(info_lines))


class TextEncodeQwenImageEditAdvanced:
    @classmethod
    def INPUT_TYPES(s) -> dict[str, dict[str, Any]]:
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
                    {"default": 384, "min": 8, "max": 2048, "step": 8},
                ),
                "global_latent_size": (
                    "INT",
                    {"default": 1024, "min": 32, "max": 4096, "step": 32},
                ),
                "global_crop_method": (
                    ["crop", "pad", "square_pad"],
                    {"default": "crop"},
                ),
                "global_resize_method": (
                    ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],
                    {"default": "lanczos"},
                ),
                "global_raw_mode": ("BOOLEAN", {"default": False}),
                "global_foolproof_protection": (
                    "BOOLEAN",
                    {"default": False},
                ),  # Защита от дурака
                "global_max_pixels": (
                    "INT",
                    {
                        "default": MAX_PIXELS,
                        "min": 131072,
                        "max": 16777216,
                        "step": 1024,
                    },
                ),  # 2048x2048 по умолчанию
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": text.get("default_sysprompt"),
                    },
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "INT", "INT", "STRING")
    RETURN_NAMES = ("conditioning", "orig_width", "orig_height", "info")
    FUNCTION = "encode"
    CATEGORY = QWEN_CATEGORY
    DESCRIPTION = text.get("node_description")

    def encode(
        self,
        clip,
        prompt,
        image1,
        vae=None,
        image2=None,
        image3=None,
        image4=None,
        multi_params=None,
        global_vl_size=384,
        global_latent_size=1024,
        global_crop_method="crop",
        global_resize_method="lanczos",
        global_raw_mode=False,
        global_foolproof_protection=False,
        global_max_pixels=MAX_PIXELS,
        system_prompt="",
    ) -> tuple[list | Any, Any, Any, LiteralString]:

        # Сохраняем оригинальные размеры первого изображения
        batch_size, height, width, channels = image1.shape
        orig_width = width
        orig_height = height

        # Формируем список всех изображений
        images = [image1, image2, image3, image4]
        images = [img for img in images if img is not None]

        info_lines = []
        info_lines.append(f"Processing {len(images)} image(s)")
        info_lines.append(f"Original size of image1: {orig_width}x{orig_height}")

        # Подготавливаем системный промпт
        if not system_prompt:
            system_prompt = text.get("default_sysprompt")

        llama_template = prompts.get("system")

        ref_latents = []
        images_vl = []
        image_prompt = ""

        for i, image in enumerate(images):
            # Получаем параметры для текущего изображения
            if multi_params is not None and f"image{i+1}" in multi_params:
                # Используем индивидуальные параметры из multi_params
                params = multi_params[f"image{i+1}"]
                img_vl_size = params["vl_target_size"]
                img_latent_size = params["latent_target_size"]
                img_crop_method = params["crop_method"]
                img_raw_mode = params["raw_mode"]
                img_resize_method = params["resize_method"]
                img_foolproof_protection = params.get("foolproof_protection", False)
                img_max_pixels = params.get("max_pixels", MAX_PIXELS)
                params_source = "individual"
            else:
                # Используем глобальные параметры
                img_vl_size = global_vl_size
                img_latent_size = global_latent_size
                img_crop_method = global_crop_method
                img_raw_mode = global_raw_mode
                img_resize_method = global_resize_method
                img_foolproof_protection = global_foolproof_protection
                img_max_pixels = global_max_pixels
                params_source = "global"

            # Получаем размеры текущего изображения
            batch_size, height, width, channels = image.shape
            info_lines.append(f"\nImage {i+1} ({params_source} params):")
            info_lines.append(f"  Original: {width}x{height}")
            info_lines.append(text.get("params_vl"))

            # 1. Обработка для VL (всегда масштабируем для избежания OOM)
            vl_info = self._process_image(
                image,
                target_size=img_vl_size,
                resize_method=img_resize_method,
                crop_method=img_crop_method,
                target_multiple=8,
                raw_mode=False,  # Для VL всегда обрабатываем
                purpose="VL",
                foolproof_protection=img_foolproof_protection,
                max_pixels=img_max_pixels,
            )
            images_vl.append(vl_info["image"])
            info_lines.append(f"  VL result: {vl_info['info']}")

            # 2. Обработка для VAE (если есть)
            if vae is not None:
                latent_info = self._process_image(
                    image,
                    target_size=img_latent_size,
                    resize_method=img_resize_method,
                    crop_method=img_crop_method,
                    target_multiple=32,
                    raw_mode=img_raw_mode,
                    purpose="VAE",
                    foolproof_protection=img_foolproof_protection,
                    max_pixels=img_max_pixels,
                )

                # Кодируем в латент
                with torch.no_grad():
                    latent = vae.encode(latent_info["image"][:, :, :, :3])
                ref_latents.append(latent)
                info_lines.append(f"  VAE result: {latent_info['info']}")

            # Добавляем в промпт
            image_prompt += (
                f"Picture {i+1}: <|vision_start|><|image_pad|><|vision_end|>"
            )

        # Токенизация и кодирование
        try:
            tokens = clip.tokenize(
                image_prompt + prompt, images=images_vl, llama_template=llama_template
            )
        except Exception as e:
            info_lines.append(f"\nWarning: Could not use llama_template: {e}")
            tokens = clip.tokenize(image_prompt + prompt, images=images_vl)

        # Кодирование
        if hasattr(clip, "encode_from_tokens_scheduled"):
            conditioning = clip.encode_from_tokens_scheduled(tokens)
        else:
            conditioning = clip.encode_from_tokens(tokens)

        # Добавляем reference_latents если есть
        if ref_latents:
            if isinstance(conditioning, list):
                for c in conditioning:
                    c[1]["reference_latents"] = ref_latents
            else:
                conditioning = [conditioning]
                conditioning[0][1]["reference_latents"] = ref_latents
            info_lines.append(f"\nAdded {len(ref_latents)} reference latent(s)")

        # Формируем информационную строку
        info_text = "\n".join(info_lines)

        return (conditioning, orig_width, orig_height, info_text)

    def _process_image(
        self,
        image,
        target_size,
        resize_method,
        crop_method,
        target_multiple,
        raw_mode,
        purpose="VL",
        foolproof_protection=False,
        max_pixels=MAX_PIXELS,
    ):
        """Универсальная обработка изображения"""
        batch_size, height, width, channels = image.shape

        # Создаем копию для обработки
        image_tensor = image.movedim(-1, 1)  # [B, C, H, W]

        # Флаг, указывающий на то, что было применено уменьшение из-за защиты
        foolproof_applied = False
        foolproof_scale = 1.0

        # ПРИМЕНЕНИЕ ЗАЩИТЫ ОТ ДУРАКА
        # Защита работает только если:
        # 1. Включена галочка foolproof_protection
        # 2. Режим RAW включен (только в RAW мы пропускаем большие изображения)
        # 3. Назначение - VAE (для VL мы всегда уменьшаем)
        if purpose == "VAE" and raw_mode and foolproof_protection:
            current_pixels = width * height
            if current_pixels > max_pixels:
                # Вычисляем коэффициент уменьшения
                scale_factor = math.sqrt(max_pixels / current_pixels)
                # Округляем до ближайшего целого
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)

                # Убедимся, что размеры не стали меньше 32 (минимальный размер для VAE)
                new_width = max(32, new_width)
                new_height = max(32, new_height)

                # Применяем уменьшение
                image_tensor = comfy.utils.common_upscale(
                    image_tensor, new_width, new_height, resize_method, "disabled"
                )

                # Обновляем размеры
                width, height = new_width, new_height
                foolproof_applied = True
                foolproof_scale = scale_factor

                info_lines = []
                info_lines.append(
                    f"Foolproof protection applied: {width}x{height} -> {new_width}x{new_height}"
                )
                info_lines.append(text.get("scale_pixels"))

        if raw_mode and purpose == "VAE":
            # В raw_mode для VAE только приводим к кратности
            new_width, new_height = self._make_multiple(
                width, height, target_multiple, crop_method
            )
            scale_w = new_width / width
            scale_h = new_height / height

            # Формируем информационное сообщение
            info_parts = []
            if foolproof_applied:
                info_parts.append(f"foolproof(x{foolproof_scale:.2f})")
            info_parts.append(f"{width}x{height} -> {new_width}x{new_height}")
            if not foolproof_applied:
                info_parts.append(f"(raw, x{scale_w:.2f}, x{scale_h:.2f})")

            info = " ".join(info_parts)
        else:
            # Масштабируем к целевому размеру с сохранением пропорций
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

        # Применяем изменение размера если нужно
        if new_width != width or new_height != height:
            image_tensor = comfy.utils.common_upscale(
                image_tensor, new_width, new_height, resize_method, "disabled"
            )

        # Возвращаем в исходный формат
        result_image = image_tensor.movedim(1, -1)

        return {"image": result_image, "info": info}

    def _make_multiple(self, width, height, multiple, method="crop"):
        """Приведение размеров к кратности multiple выбранным методом"""
        if method == "crop":
            # Обрезка до ближайшего меньшего кратного значения
            new_width = (width // multiple) * multiple
            new_height = (height // multiple) * multiple
            if new_width <= 0:
                new_width = multiple
            if new_height <= 0:
                new_height = multiple

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
