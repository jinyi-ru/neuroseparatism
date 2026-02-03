"""Строковые текстовки"""

# pylint: disable=E0602:undefined-variable
# Траверсинг переменной в f-строке

text = {
    "default_sysprompt": "Describe the key features of the input image "
    "(color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter "
    "or modify the image. Generate a new image that meets "
    "the user's requirements while maintaining consistency "
    "with the original input where appropriate.",
    "node_description": "Qwen Image Edit Advanced - Flexible processing "
    "with per-image parameters",
    "params_vl": f"  Params: VL={img_vl_size}, "  # type: ignore
    f"Latent={img_latent_size}, "  # type: ignore
    f"Crop={img_crop_method}, "  # type: ignore
    f"Raw={img_raw_mode}, "  # type: ignore
    f"Foolproof={img_foolproof_protection}",  # type: ignore
    "scale_pixels": f"  Scale: {scale_factor:.4f}, "  # type: ignore
    f"Pixels: {current_pixels} -> {new_width*new_height}",  # type: ignore
}

prompts = {
    "system": f"<|im_start|>system\n{system_prompt}<|im_end|>\n"  # type: ignore
    f"<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n"
}
