"""
Image and video captioning with multiple backends:
- BLIP / BLIP-2
- InternVL
- Qwen2-VL
- Florence-2

Plugin interface: Captioner.predict(frame_path)
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Literal, Optional, Union

import torch
from PIL import Image

from .logger import get_logger

logger = get_logger()


class Captioner(ABC):
    """
    Abstract base class for image/video captioners.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize captioner.

        Args:
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        logger.info(f"Captioner initialized on device: {device}")

    @abstractmethod
    def predict(self, image_path: Union[str, Path]) -> str:
        """
        Generate caption for an image.

        Args:
            image_path: Path to image file

        Returns:
            Generated caption string
        """
        pass

    def predict_batch(self, image_paths: List[Union[str, Path]]) -> List[str]:
        """
        Generate captions for multiple images.

        Default implementation calls predict() sequentially.
        Subclasses can override for batch processing.

        Args:
            image_paths: List of paths to image files

        Returns:
            List of generated captions
        """
        return [self.predict(path) for path in image_paths]


class BLIPCaptioner(Captioner):
    """BLIP captioner (Salesforce/blip-image-captioning-base)."""

    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base", device: Optional[str] = None):
        super().__init__(device)
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError:
            raise ImportError("transformers package required. Install with: pip install transformers")

        logger.info(f"Loading BLIP model: {model_name}")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def predict(self, image_path: Union[str, Path]) -> str:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=50)

        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption

    def predict_batch(self, image_paths: List[Union[str, Path]]) -> List[str]:
        images = [Image.open(path).convert("RGB") for path in image_paths]
        inputs = self.processor(images, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=50)

        captions = [self.processor.decode(output, skip_special_tokens=True) for output in outputs]
        return captions


class BLIP2Captioner(Captioner):
    """BLIP-2 captioner (Salesforce/blip2-opt-2.7b or blip2-flan-t5-xl)."""

    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b", device: Optional[str] = None):
        super().__init__(device)
        try:
            from transformers import Blip2ForConditionalGeneration, Blip2Processor
        except ImportError:
            raise ImportError("transformers package required. Install with: pip install transformers")

        logger.info(f"Loading BLIP-2 model: {model_name}")
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval()

    def predict(self, image_path: Union[str, Path]) -> str:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(image, return_tensors="pt").to(self.device, dtype=torch.float16 if self.device == "cuda" else torch.float32)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=50)

        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption

    def predict_batch(self, image_paths: List[Union[str, Path]]) -> List[str]:
        images = [Image.open(path).convert("RGB") for path in image_paths]
        inputs = self.processor(images, return_tensors="pt", padding=True).to(self.device, dtype=torch.float16 if self.device == "cuda" else torch.float32)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=50)

        captions = [self.processor.decode(output, skip_special_tokens=True) for output in outputs]
        return captions


class InternVLCaptioner(Captioner):
    """InternVL captioner (OpenGVLab/InternVL-Chat-V1-5)."""

    def __init__(self, model_name: str = "OpenGVLab/InternVL-Chat-V1-5", device: Optional[str] = None):
        super().__init__(device)
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError("transformers package required. Install with: pip install transformers")

        logger.info(f"Loading InternVL model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def predict(self, image_path: Union[str, Path]) -> str:
        image = Image.open(image_path).convert("RGB")
        prompt = "Describe this image in detail."

        # InternVL uses a chat interface
        with torch.no_grad():
            response = self.model.chat(self.tokenizer, image, prompt, generation_config={"max_new_tokens": 100})

        return response


class Qwen2VLCaptioner(Captioner):
    """Qwen2-VL captioner (Qwen/Qwen2-VL-7B-Instruct)."""

    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct", device: Optional[str] = None):
        super().__init__(device)
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError:
            raise ImportError("transformers package required. Install with: pip install transformers")

        logger.info(f"Loading Qwen2-VL model: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def predict(self, image_path: Union[str, Path]) -> str:
        image = Image.open(image_path).convert("RGB")

        # Qwen2-VL uses a specific prompt format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image in detail."},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=[text_prompt], images=[image], return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=100)

        # Decode only the generated part
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return caption


class Florence2Captioner(Captioner):
    """Florence-2 captioner (microsoft/Florence-2-large)."""

    def __init__(self, model_name: str = "microsoft/Florence-2-large", device: Optional[str] = None):
        super().__init__(device)
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError:
            raise ImportError("transformers package required. Install with: pip install transformers")

        logger.info(f"Loading Florence-2 model: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def predict(self, image_path: Union[str, Path]) -> str:
        image = Image.open(image_path).convert("RGB")

        # Florence-2 uses task prompts
        task_prompt = "<CAPTION>"
        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=100)

        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Remove task prompt from output
        caption = caption.replace(task_prompt, "").strip()

        return caption


def create_captioner(
    backend: Literal["blip", "blip2", "internvl", "qwen2-vl", "florence2"],
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> Captioner:
    """
    Factory function to create a captioner.

    Args:
        backend: Captioner backend to use
        model_name: Optional specific model name (uses default if None)
        device: Device to run on ('cuda', 'cpu', or None for auto)

    Returns:
        Captioner instance

    Raises:
        ValueError: If backend is not supported
    """
    backend_map = {
        "blip": BLIPCaptioner,
        "blip2": BLIP2Captioner,
        "internvl": InternVLCaptioner,
        "qwen2-vl": Qwen2VLCaptioner,
        "florence2": Florence2Captioner,
    }

    if backend not in backend_map:
        raise ValueError(f"Unsupported backend: {backend}. Choose from: {list(backend_map.keys())}")

    captioner_class = backend_map[backend]

    if model_name is not None:
        return captioner_class(model_name=model_name, device=device)
    else:
        return captioner_class(device=device)
