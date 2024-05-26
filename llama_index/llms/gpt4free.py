from typing import Any

from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback

import g4f


class GPT4Free(CustomLLM):
    context_window: int = 3900
    num_output: int = 256
    model_name: str = "g4f"
    g4f_model: str = "gpt-4"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
            g4f_model=self.g4f_model,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        print("prompt", prompt)
        print("kwargs", kwargs)
        response = g4f.ChatCompletion.create(
            model=self.g4f_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return CompletionResponse(text=response)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        streaming_tokens = g4f.ChatCompletion.create(
            model=self.g4f_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        response = ""
        for token in streaming_tokens:
            response += token
            yield CompletionResponse(text=response, delta=token)

    def get_provider(self):
        return g4f.get_last_provider().__name__
