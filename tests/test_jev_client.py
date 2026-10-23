import asyncio
import base64
import io
import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch
from unittest.mock import AsyncMock
from types import SimpleNamespace

from aiohttp import web
from PIL import Image


root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("jev_client_test", root / "jev_client.py")
jev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jev)


class JevHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.image_headers = []
        self.status = 200
        self.data = {"answers": {"complete": {"type": "noul", "noul": 0.9}}}
        self.delay = 0
        self.invalid_json = False

        async def handler(request):
            self.image_headers.append(
                request.headers.get("TypeSafe-Danger-Zone-Images")
            )
            self.requests.append(
                (request.headers.get("Authorization"), await request.json())
            )
            await asyncio.sleep(self.delay)
            if self.invalid_json:
                return web.Response(text="not json", content_type="application/json")
            return web.json_response(self.data, status=self.status)

        app = web.Application()
        app.router.add_post("/v1/systemone", handler)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}/v1"
        self.client = jev.JevClient(
            {"jev_base_url": self.base_url, "jev_api_key": "test-only-key"}
        )
        self.questions = {
            "complete": {"type": "noul", "instructions": "Is it complete?"}
        }

    async def asyncTearDown(self):
        await self.runner.cleanup()

    async def test_native_endpoint_auth_and_body(self):
        answers = await self.client.evaluate({"message": "你好"}, self.questions)
        self.assertEqual(answers["complete"]["noul"], 0.9)
        auth, body = self.requests[0]
        self.assertEqual(auth, "Bearer test-only-key")
        self.assertEqual(
            body,
            {
                "model": "jev-latest",
                "state": {"message": "你好"},
                "questions": self.questions,
            },
        )

    async def test_http_failures_and_invalid_responses(self):
        for status in [401, 429, 500, 302]:
            self.status = status
            with self.assertRaisesRegex(ValueError, f"HTTP {status}"):
                await self.client.evaluate("x", self.questions)
        self.status = 200
        for data in [
            [],
            {},
            {"answers": []},
            {"answers": {}},
            {"answers": {"complete": {"type": "score"}}},
        ]:
            self.data = data
            with self.assertRaises(ValueError):
                await self.client.evaluate("x", self.questions)
        self.invalid_json = True
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            await self.client.evaluate("x", self.questions)

    async def test_timeout(self):
        self.client.timeout = 0.01
        self.delay = 0.03
        with self.assertRaises(asyncio.TimeoutError):
            await self.client.evaluate("x", self.questions)

    async def test_missing_key_does_not_send_request(self):
        self.client.api_key = ""
        with self.assertRaisesRegex(ValueError, "jev_api_key"):
            await self.client.evaluate("x", self.questions)
        self.assertEqual(self.requests, [])

    def image(self, size=(320, 320), color="red", source="test-image", fmt="PNG"):
        output = io.BytesIO()
        Image.new("RGB", size, color).save(output, fmt)
        return SimpleNamespace(
            type="Image",
            file=source,
            convert_to_base64=AsyncMock(
                return_value=base64.b64encode(output.getvalue()).decode()
            ),
        )

    async def test_images_have_references_native_assets_and_header(self):
        picture = self.image(size=(20, 3000))
        state = {"message": "What is in this picture?"}
        await self.client.evaluate(state, self.questions, images=[picture])
        body = self.requests[-1][1]
        self.assertEqual(self.image_headers, ["true"])
        self.assertNotIn("images", state)
        self.assertEqual(body["state"]["images"], list(body["assets"]))
        asset = next(iter(body["assets"].values()))
        self.assertEqual(asset["format"], "jpg")
        with Image.open(io.BytesIO(base64.b64decode(asset["data_base64"]))) as decoded:
            self.assertTrue(280 <= decoded.width <= 1568)
            self.assertTrue(280 <= decoded.height <= 1568)
            self.assertEqual(decoded.mode, "RGB")

    async def test_image_limit_uses_recent_unique_images(self):
        first, second, third = (self.image(source=str(index)) for index in range(3))
        self.client.max_images = 2
        await self.client.evaluate(
            "message", self.questions, images=[first, second, second, third]
        )
        body = self.requests[-1][1]
        self.assertEqual(len(body["assets"]), 2)
        self.assertEqual(body["state"]["omitted_images"], 1)
        first.convert_to_base64.assert_not_awaited()
        second.convert_to_base64.assert_awaited_once()

    async def test_disabled_vision_sends_text_without_assets(self):
        picture = self.image()
        self.client.enable_images = False
        await self.client.evaluate("hello", self.questions, images=[picture])
        self.assertNotIn("assets", self.requests[-1][1])
        self.assertEqual(self.image_headers, [None])
        picture.convert_to_base64.assert_not_awaited()

    async def test_image_conversion_shares_request_timeout(self):
        picture = self.image()

        async def slow():
            await asyncio.sleep(10)

        picture.convert_to_base64 = slow
        self.client.timeout = 0.01
        with self.assertRaises(asyncio.TimeoutError):
            await self.client.evaluate("hello", self.questions, images=[picture])
        self.assertEqual(self.requests, [])

    async def test_invalid_image_does_not_send_a_partial_request(self):
        picture = self.image()
        picture.convert_to_base64.return_value = "not valid base64"
        with self.assertRaises(ValueError):
            await self.client.evaluate("hello", self.questions, images=[picture])
        self.assertEqual(self.requests, [])

    def test_environment_and_full_endpoint(self):
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "env-test-key"}):
            client = jev.JevClient({"jev_base_url": self.base_url + "/systemone/"})
            self.assertEqual(client.api_key, "env-test-key")
            self.assertEqual(client.endpoint, self.base_url + "/systemone")
            explicit = jev.JevClient({"jev_api_key": "configured-key"})
            self.assertEqual(explicit.api_key, "configured-key")
