from __future__ import annotations
import asyncio
import hmac
import json

import httpx
from fastapi import HTTPException

from .config import load_configuration
from .illustrations import LOCALIZATION_PROMPT, apply_illustrations, missing_illustrations, wants_illustrations
from .prompts import parse_choices
from .providers import ProviderFailure, stream_provider
from .schemas import Budgets, Message


class Runtime:
    def __init__(self):
        self.settings, self.digest, self.keys, self.token, self.state = load_configuration()
        self.client = None
        self.attempts = {}  # Process-lifetime replay guard; Core owns durable recovery.
        self.active = 0
        self.retrieval_active = 0

    async def start(self):
        if self.state == "READY":
            limits = self.settings.limits
            self.client = httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(retries=0),
                timeout=httpx.Timeout(limits.idleMs / 1000, connect=limits.connectMs / 1000),
                limits=httpx.Limits(max_connections=limits.concurrency * 2, max_keepalive_connections=limits.concurrency),
                follow_redirects=False, trust_env=False)

    async def close(self):
        for attempt in self.attempts.values():
            attempt["cancelled"].set()
        if self.client is not None:
            await self.client.aclose()

    def authenticate(self, authorization):
        if self.state != "READY":
            raise HTTPException(503, detail="ASSISTANT_" + self.state)
        if not authorization or not hmac.compare_digest(authorization, "Bearer " + self.token):
            raise HTTPException(401, detail="ASSISTANT_UNAUTHORIZED")

    def bind(self, digest):
        if not hmac.compare_digest(self.digest, digest):
            raise HTTPException(409, detail="ASSISTANT_CONFIGURATION_CHANGED")

    def model(self, key):
        for item in self.settings.models:
            if item.id == key and item.enabled:
                return item
        raise HTTPException(422, detail="MODEL_UNAVAILABLE")

    def models(self):
        return {"contractVersion": 1, "configSha256": self.digest,
            "models": [{"id": m.id, "providerId": m.provider, "modelId": m.model,
                "label": m.label or m.model, "configured": m.id in self.keys, "enabled": m.enabled,
                "capabilities": {"images": m.capabilities.images, "streaming": m.capabilities.streaming,
                    "imageMimeTypes": ["image/jpeg", "image/png"] if m.capabilities.images else []},
                "evidence": m.evidence} for m in self.settings.models],
            "retrieval": {"state": "AVAILABLE" if self.settings.ragflow.enabled else "NOT_CONFIGURED"},
            "limits": {"concurrency": self.settings.limits.concurrency}}

    def admit(self, request):
        key = str(request.attemptId)
        if key in self.attempts:
            raise HTTPException(409, detail="ATTEMPT_ALREADY_SEEN")
        if len(self.attempts) >= self.settings.limits.attemptCapacity:
            raise HTTPException(503, detail="ASSISTANT_ATTEMPT_CAPACITY")
        if self.active >= self.settings.limits.concurrency:
            raise HTTPException(429, detail="ASSISTANT_EXECUTION_LIMIT")
        self.attempts[key] = {"runId": str(request.runId), "cancelled": asyncio.Event(), "state": "ACTIVE"}
        self.active += 1
        return self.attempts[key]

    async def stream(self, request, model, attempt, illustration_ms=0):
        answer = []
        limit = self.settings.limits
        total_ms = min(limit.chatMs, request.budgets.totalMs or limit.chatMs)
        deadline = asyncio.get_running_loop().time() + total_ms / 1000
        try:
            async for event, data in stream_provider(self.client, request, model, self.keys[model.id], self.settings, attempt["cancelled"]):
                if event == "delta":
                    answer.append(data["text"])
                if event == "done":
                    original = "".join(answer)
                    remaining = int((deadline - asyncio.get_running_loop().time()) * 1000) - 1000
                    budget = min(illustration_ms, remaining)
                    if budget >= 1000 and data["finishReason"] == "stop" and wants_illustrations(request):
                        revised, usage = await self.localize(request, model, attempt["cancelled"], original, budget)
                        if usage is not None:
                            yield self.event("usage", {"purpose": "CHAT", "illustration": usage})
                        if revised is not None:
                            data["illustratedAnswer"] = revised
                    cards = parse_choices(original) if request.purpose == "CHAT" else []
                    if cards:
                        yield self.event("choices", {"cards": cards})
                    attempt["state"] = "COMPLETED"
                yield self.event(event, data)
        except ProviderFailure as exc:
            attempt["state"] = "CANCELLED" if exc.code == "CANCELLED" else "FAILED"
            yield self.event("error", {"code": exc.code})
        except asyncio.CancelledError:
            attempt["state"] = "INTERRUPTED"
            raise
        except Exception:
            attempt["state"] = "FAILED"
            yield self.event("error", {"code": "EXECUTION_FAILED"})
        finally:
            self.active -= 1

    async def localize(self, request, model, cancelled, original, budget):
        current = request.messages[-1]
        if len(original.encode("utf-8")) > 23000 or not model.capabilities.images:
            return None, None
        answer, headings, targets = missing_illustrations(original, len(current.images))
        if not targets:
            return None, None
        vision = next((candidate for candidate in self.settings.models
            if candidate.model == "qwen3-vl-plus" and candidate.enabled
            and candidate.capabilities.images and candidate.capabilities.streaming
            and candidate.provider == model.provider and candidate.baseUrl == model.baseUrl
            and self.keys.get(candidate.id) == self.keys[model.id]), model)
        localization = request.model_copy(update={
            "messages": [Message(role="user", content=json.dumps(targets, ensure_ascii=False), images=current.images)],
            "budgets": Budgets(totalMs=budget, maxOutputTokens=600),
        })
        chunks = []
        usage = {"status": "UNCONFIRMED"}
        try:
            async for event, data in stream_provider(self.client, localization, vision, self.keys[vision.id],
                    self.settings, cancelled, instruction=LOCALIZATION_PROMPT):
                if event == "delta":
                    chunks.append(data["text"])
                elif event == "usage":
                    usage["providerReported"] = data["providerReported"]
                elif event == "done" and data["finishReason"] == "stop":
                    usage["status"] = "COMPLETED"
        except ProviderFailure as exc:
            if exc.code == "CANCELLED":
                raise
            return None, usage
        if cancelled.is_set():
            raise ProviderFailure("CANCELLED")
        if usage["status"] != "COMPLETED":
            return None, usage
        revised = apply_illustrations(answer, headings, targets, "".join(chunks), len(current.images))
        if revised is not None and len(json.dumps(revised, ensure_ascii=False).encode("utf-8")) > 60000:
            revised = None
        return revised, usage

    @staticmethod
    def event(name, data):
        return "event: " + name + "\ndata: " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n\n"
