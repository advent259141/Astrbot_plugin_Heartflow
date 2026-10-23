"""Minimal asynchronous client for TypeSafe's native System One API."""

import asyncio
import base64
import io
import math
import os
import uuid
from urllib.parse import urlsplit


def image_components(event) -> list:
    """Read direct image components without treating filenames as message text."""
    message = getattr(getattr(event, "message_obj", None), "message", None) or []
    return [part for part in message if getattr(part, "type", None) == "Image"]


def image_source(image) -> str:
    """Return a source usable by AstrBot's image resolver and reply provider."""
    if isinstance(image, str):
        return image
    return getattr(image, "url", None) or getattr(image, "file", None) or ""


def _encode_image(encoded: str) -> str:
    """Normalize EXIF, alpha, animation and size to the native vision contract."""
    from PIL import Image, ImageOps

    raw = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(raw)) as original:
        original.seek(0)
        upright = ImageOps.exif_transpose(original)
        rgba = upright.convert("RGBA")
        rgb = Image.new("RGB", rgba.size, "white")
        rgb.paste(rgba, mask=rgba.getchannel("A"))
        rgb.thumbnail((1568, 1568), Image.Resampling.LANCZOS)
        # Padding preserves narrow images without exceeding either dimension.
        width, height = rgb.size
        padded = Image.new("RGB", (max(280, width), max(280, height)), "white")
        padded.paste(rgb, ((padded.width - width) // 2, (padded.height - height) // 2))
        output = io.BytesIO()
        padded.save(output, "JPEG", quality=85, progressive=False)
    return base64.b64encode(output.getvalue()).decode("ascii")


def number(value, minimum: float, maximum: float) -> float:
    """Reject malformed API values, including booleans and NaN."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("JEV returned a non-numeric value")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError("JEV returned a value outside the expected range")
    return result


class JevClient:
    def __init__(self, config):
        self.api_key = (
            str(config.get("jev_api_key", "") or "").strip()
            or os.getenv("TYPESAFE_API_KEY", "").strip()
        )
        base_url = (
            str(config.get("jev_base_url", "") or "https://api.typesafe.ai/v1")
            .strip()
            .rstrip("/")
        )
        self.endpoint = (
            base_url if base_url.endswith("/systemone") else base_url + "/systemone"
        )
        self.inline_images = urlsplit(self.endpoint).hostname == "openrouter.ai"
        self.model = str(config.get("jev_model", "") or "jev-latest").strip()
        self.enable_images = bool(config.get("jev_enable_images", True))
        try:
            self.max_images = max(1, min(8, int(config.get("jev_max_images", 4))))
        except (TypeError, ValueError, OverflowError):
            self.max_images = 4
        try:
            timeout = float(config.get("jev_timeout_seconds", 10))
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError
        except (TypeError, ValueError):
            timeout = 10.0
        self.timeout = min(timeout, 120.0)

    async def evaluate(
        self, state, questions: dict, *, images: list | None = None
    ) -> dict:
        if not self.api_key:
            raise ValueError("Configure jev_api_key or TYPESAFE_API_KEY")
        return await asyncio.wait_for(
            self._evaluate(state, questions, images), timeout=self.timeout
        )

    async def _evaluate(self, state, questions: dict, images: list | None) -> dict:
        # AstrBot already uses aiohttp; keep ONNX/LLM-only imports independent.
        import aiohttp

        payload = {"model": self.model, "state": state, "questions": questions}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if images and self.enable_images:
            unique = {}
            for picture in images:
                source = image_source(picture)
                unique.setdefault(source or str(id(picture)), picture)
            selected = list(unique.values())[-self.max_images :]

            async def asset(picture):
                if isinstance(picture, str):
                    from astrbot.api.message_components import Image

                    picture = Image(file=picture)
                encoded = await picture.convert_to_base64()
                return {
                    "type": "image",
                    "format": "jpg",
                    "data_base64": await asyncio.to_thread(_encode_image, encoded),
                }

            prepared = await asyncio.gather(*(asset(picture) for picture in selected))
            multimodal_state = (
                dict(state) if isinstance(state, dict) else {"message": state}
            )
            if self.inline_images:
                # OpenRouter ignores native assets; embed images in the state.
                multimodal_state["images"] = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64," + value["data_base64"]
                        },
                    }
                    for value in prepared
                ]
            else:
                assets = {str(uuid.uuid4()): value for value in prepared}
                multimodal_state["images"] = list(assets)
                payload["assets"] = assets
                headers["TypeSafe-Danger-Zone-Images"] = "true"
            if len(unique) > len(selected):
                multimodal_state["omitted_images"] = len(unique) - len(selected)
            payload["state"] = multimodal_state

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    allow_redirects=False,
                ) as response:
                    if response.status != 200:
                        # Error bodies/URLs can contain credentials or user messages.
                        raise ValueError(f"JEV HTTP {response.status}")
                    try:
                        data = await response.json()
                    except (ValueError, aiohttp.ContentTypeError) as exc:
                        raise ValueError("JEV returned invalid JSON") from exc
        except aiohttp.ClientError as exc:
            raise ValueError("JEV connection failed") from exc

        if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
            raise ValueError("JEV response is missing answers")
        answers = data["answers"]
        for name, question in questions.items():
            if (
                not isinstance(answers.get(name), dict)
                or answers[name].get("type") != question["type"]
            ):
                raise ValueError("JEV response has a missing or mismatched answer")
        return answers
