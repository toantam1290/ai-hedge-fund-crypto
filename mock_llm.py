"""
Mock LLM for testing without real API keys
"""
import json
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Optional

class MockLLM(BaseLanguageModel):
    """Mock LLM that returns predetermined responses"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        # Mock trading decision based on simple heuristics
        mock_response = {
            "decisions": {
                "BTCUSDT": {
                    "action": "buy",
                    "quantity": 0.1,
                    "confidence": 75,
                    "reasoning": "Mock decision: MACD signals show bullish trend, recommend small buy position"
                }
            }
        }
        
        from langchain_core.outputs import LLMResult, Generation
        return LLMResult(generations=[[Generation(text=json.dumps(mock_response))]])
    
    @property
    def _llm_type(self) -> str:
        return "mock"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        mock_response = {
            "decisions": {
                "BTCUSDT": {
                    "action": "hold",
                    "quantity": 0,
                    "confidence": 60,
                    "reasoning": "Mock decision: Mixed signals, maintaining current position"
                }
            }
        }
        return json.dumps(mock_response)

# Create instance for easy import
mock_llm = MockLLM()
