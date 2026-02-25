"""
LLM Client — Ollama interface for local inference.

Wraps the Ollama Python client with:
- Financial-domain system prompt
- Context injection
- Streaming support
- Fallback error handling
"""

import logging
from typing import Iterator, Optional

import ollama

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a financial tax expert assistant with access to a knowledge base containing:
1. Tax records data (CSV): anonymized financial records with income, deductions, tax rates, and tax owed across different taxpayer types, states, and years
2. Financial reports (PDF): IRS form instructions, tax code statutes, and financial regulations
3. Financial presentations (PDF/PPT): educational slides on microeconomics, tax theory, and tax policy

Your role:
- Answer questions accurately using ONLY the provided context
- For numerical questions, cite specific figures from the data
- For policy questions, explain the concept clearly and cite the relevant source
- If the context doesn't contain enough information, say so honestly — do not hallucinate
- Format currency as $X,XXX.XX and percentages as X.XX%
- Keep answers concise but complete

Always distinguish between:
- Factual data answers (from CSV/graph): "According to the financial records..."
- Conceptual answers (from PDF/PPT): "According to the financial presentation/IRS guidelines..."
"""


class LLMClient:
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        # Test connectivity
        try:
            ollama.list()
            logger.info(f"Ollama connected. Model: {model}")
        except Exception as e:
            logger.warning(f"Ollama not reachable: {e}. Will fail at query time.")

    def chat(self, user_message: str, context: str) -> str:
        """
        Single-turn chat with injected retrieval context.
        Returns the full response string.
        """
        prompt = self._build_prompt(user_message, context)

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": 0.1,  # low temp for factual financial answers
                    "num_ctx": 4096,
                    "top_p": 0.9,
                },
            )
            return response["message"]["content"]

        except Exception as e:
            logger.error(f"Ollama inference error: {e}")
            return (
                f"I encountered an error generating a response. "
                f"Please ensure Ollama is running with: `ollama serve` "
                f"and the model is available: `ollama pull {self.model}`"
            )

    def chat_stream(self, user_message: str, context: str) -> Iterator[str]:
        """
        Streaming version — yields tokens as they're generated.
        Use with Server-Sent Events for real-time UI updates.
        """
        prompt = self._build_prompt(user_message, context)

        try:
            stream = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
                options={"temperature": 0.1, "num_ctx": 4096},
            )
            for chunk in stream:
                token = chunk["message"]["content"]
                if token:
                    yield token

        except Exception as e:
            logger.error(f"Ollama stream error: {e}")
            yield f"[Error: {e}]"

    @staticmethod
    def _build_prompt(user_message: str, context: str) -> str:
        return (
            f"CONTEXT FROM KNOWLEDGE BASE:\n"
            f"{'='*60}\n"
            f"{context}\n"
            f"{'='*60}\n\n"
            f"USER QUESTION: {user_message}\n\n"
            f"Please answer the question using the context above."
        )
