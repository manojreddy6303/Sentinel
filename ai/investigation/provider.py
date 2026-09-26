"""
LLM Provider Abstraction Layer for Sentinel Phase 7

Provides modular integration for LLM providers:
- LLMProvider: Abstract base interface
- GeminiProvider: REST integration with Google Gemini Models (e.g. gemini-1.5-flash)
- MockLLMProvider: Test provider supporting deterministic mocking without live API calls
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

import requests

logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """Raised when an LLM provider request fails or encounters an unrecoverable error."""
    pass


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate_structured_intent(
        self, user_query: str, history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Translate user natural language query into structured retrieval parameters.
        Must return a dictionary conforming to the structured query schema.
        """
        pass

    @abstractmethod
    def generate_grounded_response(
        self,
        user_query: str,
        retrieved_data: Dict[str, Any],
        history: Optional[List[Dict[str, str]]] = None,
        context_notes: Optional[str] = None,
    ) -> str:
        """
        Synthesize a natural language investigation answer strictly grounded
        in the provided retrieved Sentinel database records and evidence.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is configured and available."""
        pass


class GeminiProvider(LLMProvider):
    """
    Google Gemini REST API implementation.
    Uses standard HTTP calls (compatible with Python 3.7+ without extra heavy SDKs).
    """

    DEFAULT_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-flash-latest",
        temperature: float = 0.1,
        timeout: float = 30.0,
    ):
        self.api_key = (api_key or "").strip()
        self.model = model or "gemini-flash-latest"
        self.temperature = temperature
        self.timeout = timeout

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _call_gemini(self, prompt: str, system_instruction: Optional[str] = None, json_mode: bool = False) -> str:
        if not self.api_key:
            raise LLMProviderError("Gemini API key is not configured.")

        url = f"{self.DEFAULT_API_BASE}/models/{self.model}:generateContent?key={self.api_key}"

        contents = [{"parts": [{"text": prompt}]}]
        body: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": 1024,
            },
        }

        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"

        if system_instruction:
            body["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        try:
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=body,
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                raise LLMProviderError(f"Gemini API error ({resp.status_code}): {resp.text}")

            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise LLMProviderError("No response candidates returned by Gemini.")

            content_parts = candidates[0].get("content", {}).get("parts", [])
            if not content_parts:
                raise LLMProviderError("Empty content returned by Gemini.")

            return content_parts[0].get("text", "")
        except requests.RequestException as e:
            logger.error(f"Gemini HTTP request failed: {e}")
            raise LLMProviderError(f"Gemini connection failure: {str(e)}")

    def generate_structured_intent(
        self, user_query: str, history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        system_instruction = (
            "You are Sentinel's query translation engine. Your job is to convert investigator questions "
            "into a structured JSON filter object for querying the CCTV detections database.\n"
            "DO NOT answer the question. Only output JSON matching this schema:\n"
            "{\n"
            '  "intent": "investigate" | "summarize" | "activity_analysis" | "guardrail",\n'
            '  "object_classes": ["car", "person", ...] or null,\n'
            '  "start_time": float or null,\n'
            '  "end_time": float or null,\n'
            '  "min_confidence": float or null,\n'
            '  "result_type": "detections" | "events" | "count",\n'
            '  "is_summary_request": bool,\n'
            '  "is_activity_request": bool,\n'
            '  "guardrail_category": "identity" | "criminal_attribution" | null\n'
            "}\n"
            "Rules:\n"
            "- Only use recognized classes: person, car, bicycle, motorcycle, bus, truck, backpack, handbag, suitcase, bottle, cell phone, chair, traffic light, stop sign.\n"
            "- Map 'vehicle' to object_classes=['car','bus','truck','motorcycle','bicycle'].\n"
            "- Do not infer colors.\n"
            "- If user asks who someone is, mark intent='guardrail', guardrail_category='identity'.\n"
            "- If user asks about summary/overview, set is_summary_request=true.\n"
            "- If user asks about activity/movement density/suspicious patterns, set is_activity_request=true."
        )

        history_context = ""
        if history:
            history_context = "Recent conversation context:\n" + "\n".join(
                f"{item.get('role', 'user')}: {item.get('content', '')}" for item in history[-4:]
            ) + "\n\n"

        prompt = f"{history_context}Investigator query: {user_query}"

        raw_text = self._call_gemini(prompt, system_instruction=system_instruction, json_mode=True)
        try:
            # Clean possible markdown wrapping if returned despite json mode
            cleaned = raw_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            return json.loads(cleaned.strip())
        except Exception as e:
            logger.warning(f"Failed to parse Gemini JSON output: {raw_text}. Error: {e}")
            raise LLMProviderError(f"Malformed structured output from provider: {e}")

    def generate_grounded_response(
        self,
        user_query: str,
        retrieved_data: Dict[str, Any],
        history: Optional[List[Dict[str, str]]] = None,
        context_notes: Optional[str] = None,
    ) -> str:
        system_instruction = (
            "You are Sentinel AI, an evidence-grounded video investigation assistant.\n"
            "CORE DIRECTIVE: The retrieved database records and evidence are the ABSOLUTE SOURCE OF TRUTH.\n"
            "NEVER invent timestamps, object counts, confidence scores, evidence, or events.\n"
            "NEVER perform facial recognition, claim identities, or label individuals as criminals or thieves.\n"
            "Distinguish weak/isolated raw model detections from persistent visual intelligence: "
            "If only an isolated low-confidence (<0.50) detection is present for an object, describe it as an isolated low-confidence model prediction that did not persist across frames, rather than claiming the object was present. "
            "Only describe an object as confirmed or active when supported by persistent, validated detections.\n"
            "For potential theft behavioral patterns, refer to 'Potential Theft Pattern (Human verification required)' and never claim legal confirmation or call a person a thief.\n"
            "Always cite evidence and events using tags like [08.01s], [DET-XXXX], [EVENT-XXXX], or [EV-XXXX] where available.\n"
            "If no records were found, clearly state that no matching data was detected.\n"
            "Be professional, concise, and structured."
        )

        data_summary = json.dumps(retrieved_data, indent=2, default=str)
        prompt = (
            f"User Question: {user_query}\n\n"
            f"Retrieved Sentinel Ground Truth Records:\n{data_summary}\n\n"
        )
        if context_notes:
            prompt += f"Investigator Notes:\n{context_notes}\n\n"
        prompt += "Synthesize a clear, grounded response citing relevant timestamps and records."

        return self._call_gemini(prompt, system_instruction=system_instruction, json_mode=False)


class MockLLMProvider(LLMProvider):
    """
    Mock LLM Provider for unit testing and offline development.
    Allows injecting custom intent maps, failure triggers, and predictable responses.
    """

    def __init__(
        self,
        is_available_flag: bool = True,
        intent_override: Optional[Dict[str, Any]] = None,
        response_override: Optional[str] = None,
        should_fail: bool = False,
        failure_message: str = "Simulated provider failure",
    ):
        self.is_available_flag = is_available_flag
        self.intent_override = intent_override
        self.response_override = response_override
        self.should_fail = should_fail
        self.failure_message = failure_message
        self.last_query = None
        self.last_retrieved_data = None

    def is_available(self) -> bool:
        return self.is_available_flag

    def generate_structured_intent(
        self, user_query: str, history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        self.last_query = user_query
        if self.should_fail:
            raise LLMProviderError(self.failure_message)

        if self.intent_override is not None:
            return self.intent_override

        q = (user_query or "").lower().strip()

        # Guardrail checks
        if any(term in q for term in ["who is", "what is his name", "identify this person", "identity"]):
            return {
                "intent": "guardrail",
                "guardrail_category": "identity",
                "object_classes": ["person"],
            }

        if any(term in q for term in ["criminal", "thief", "steal", "terrorist", "guilty"]):
            return {
                "intent": "guardrail",
                "guardrail_category": "criminal_attribution",
            }

        # Summary check
        if any(term in q for term in ["summarize", "summary", "overview", "activity in this video"]):
            return {
                "intent": "summarize",
                "is_summary_request": True,
                "is_activity_request": False,
                "result_type": "events",
            }

        # Activity check
        if any(term in q for term in ["suspicious", "unusual", "activity", "noteworthy"]):
            return {
                "intent": "activity_analysis",
                "is_activity_request": True,
                "is_summary_request": False,
                "result_type": "events",
            }

        # Phase 8: Potential theft / takeaway behavior check
        if any(term in q for term in ["theft", "takeaway", "burglary"]):
            return {
                "intent": "investigate",
                "object_classes": None,
                "start_time": None,
                "end_time": None,
                "min_confidence": None,
                "event_type": "POTENTIAL_THEFT",
                "result_type": "security_events",
                "is_summary_request": False,
                "is_activity_request": False,
                "guardrail_category": None,
            }

        # Phase 8: Security events review
        if any(term in q for term in ["which events", "review", "security events"]):
            return {
                "intent": "investigate",
                "object_classes": None,
                "start_time": None,
                "end_time": None,
                "min_confidence": None,
                "event_type": None,
                "result_type": "security_events",
                "is_summary_request": False,
                "is_activity_request": False,
                "guardrail_category": None,
            }

        # Parameter extraction mock
        obj_class = None
        for cls in ["car", "person", "bicycle", "truck", "bus", "dog"]:
            if cls in q:
                obj_class = cls
                break

        # Time extraction mock (e.g. "around 8 seconds", "between 8 and 12 seconds")
        start_time = None
        end_time = None
        around_match = re.search(r"around\s+(\d+(?:\.\d+)?)\s*s", q)
        if around_match:
            val = float(around_match.group(1))
            start_time = max(0.0, val - 1.0)
            end_time = val + 1.0

        between_match = re.search(r"between\s+(\d+(?:\.\d+)?)\s*and\s+(\d+(?:\.\d+)?)\s*s", q)
        if between_match:
            start_time = float(between_match.group(1))
            end_time = float(between_match.group(2))

        return {
            "intent": "investigate",
            "object_classes": [obj_class] if obj_class else None,
            "start_time": start_time,
            "end_time": end_time,
            "min_confidence": None,
            "result_type": "detections" if obj_class else "events",
            "is_summary_request": False,
            "is_activity_request": False,
            "guardrail_category": None,
        }

    def generate_grounded_response(
        self,
        user_query: str,
        retrieved_data: Dict[str, Any],
        history: Optional[List[Dict[str, str]]] = None,
        context_notes: Optional[str] = None,
    ) -> str:
        self.last_retrieved_data = retrieved_data
        if self.should_fail:
            raise LLMProviderError(self.failure_message)

        if self.response_override:
            return self.response_override

        results = retrieved_data.get("results", [])
        count = retrieved_data.get("count", len(results))
        evidence = retrieved_data.get("evidence", [])
        filters = retrieved_data.get("filters", {})

        if filters.get("event_type") == "POTENTIAL_THEFT":
            if count > 0 and results:
                first = results[0]
                ts = first.get("timestamp") if first.get("timestamp") is not None else first.get("timestamp_seconds", 0.0)
                desc = first.get("description", "")
                return (
                    f"Sentinel identified a potential object-takeaway pattern around {ts:.1f}s "
                    f"(Human verification required). {desc} Review the linked evidence."
                )
            return "No potential theft pattern was detected in the available visual evidence."

        if count == 0 and not retrieved_data.get("is_summary"):
            return "No matching Sentinel data was found for this question."

        lines = [f"Sentinel ground truth analysis found {count} relevant records."]
        if results:
            first = results[0]
            ts = first.get("timestamp") or first.get("start_time")
            objs = first.get("objects")
            if objs and isinstance(objs, list):
                obj_names = ", ".join(o.get("class", "") for o in objs if isinstance(o, dict))
                cls = f"{obj_names} ({first.get('event_type', 'event')})"
            else:
                cls = first.get("object_class") or first.get("event_type", "activity")
            conf = first.get("confidence") or first.get("max_confidence")
            conf_str = f" with {round(conf * 100)}% confidence" if conf is not None else ""
            lines.append(f"At [{ts:.2f}s], Sentinel detected {cls}{conf_str}.")

        if evidence:
            ev = evidence[0]
            lines.append(f"Preserved evidence is available: snapshot ({'yes' if ev.get('has_snapshot') else 'no'}), clip ({'yes' if ev.get('has_clip') else 'no'}).")
        else:
            lines.append("No preserved evidence currently exists for this event.")

        return " ".join(lines)


def get_llm_provider(
    provider_name: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> LLMProvider:
    """Factory function to instantiate the configured LLM provider."""
    from backend.app.core.config import settings

    prov = (provider_name or settings.LLM_PROVIDER or "gemini").lower()
    key = api_key if api_key is not None else settings.LLM_API_KEY
    mdl = model or settings.LLM_MODEL or "gemini-flash-latest"

    if prov == "mock":
        return MockLLMProvider()
    elif prov == "gemini":
        return GeminiProvider(api_key=key, model=mdl)
    else:
        logger.warning(f"Unknown LLM provider '{prov}', defaulting to Gemini provider.")
        return GeminiProvider(api_key=key, model=mdl)
