"""
Investigation Agent Module for Sentinel

Exposes InvestigationAgent as a facade over InvestigationOrchestrator.
"""

from typing import List, Dict, Any, Optional
from ai.investigation.orchestrator import InvestigationOrchestrator
from ai.investigation.provider import get_llm_provider


class InvestigationAgent:
    """Agent interface for natural-language investigation inquiry."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        provider = get_llm_provider(api_key=api_key, model=model)
        self.orchestrator = InvestigationOrchestrator(provider=provider)

    def query(
        self,
        video_id: str,
        user_prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Process natural language questions against the indexed video event timeline."""
        return self.orchestrator.process_investigation(video_id=video_id, user_query=user_prompt, history=history)
