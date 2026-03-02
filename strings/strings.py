"""Строковые текстовки"""

text = {
    "default_sysprompt": "Describe the key features of the input image "
    "(color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter "
    "or modify the image. Generate a new image that meets "
    "the user's requirements while maintaining consistency "
    "with the original input where appropriate.",
    "node_description": "Qwen Image Edit Advanced - Flexible processing "
    "with per-image parameters",
    "params_vl": "  Params: VL={}, Latent={}, Crop={}, Raw={}, Foolproof={}",
    "scale_pixels": "  Scale: {:.4f}, Pixels: {} -> {}",
}

prompts = {
    "system": "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n"
}
