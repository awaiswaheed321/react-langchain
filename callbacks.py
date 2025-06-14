from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> Any:
        """Run when LLM starts running"""
        print(f"***Prompt to LLM is: ***\n{prompts[0]}")
        print("******************")

    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM ends running"""
        print(f"***** LLM Response: **** \n{response.generations[0][0].text}")
        print("******************")
