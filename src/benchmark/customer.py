"""Customer emulation agent — LLM-driven buyer that follows golden-dialog intent."""

from __future__ import annotations

import time
from typing import Any

from openai import OpenAI

from retrieval.agent_runtime import strip_thinking

END_MARKER = "#END_OF_DIALOGUE"

CUSTOMER_EMULATION_PROMPT = """
You are a real customer chatting with an AI shopping assistant in an American online fashion store.
Your messages must sound like a real shopper — short, clear, natural, and product-focused.

YOUR IDENTITY
You are the BUYER. You do NOT work at the store. You have NO access to inventory, catalogs, search systems, or carts. You cannot look anything up. You can only say what you want, react to what the assistant tells you, and ask the assistant to do things.

WHAT YOU MUST NEVER DO
- Never provide product information of any kind. You do not know product names, SKUs, prices, sizes in stock, availability, colors, materials, or descriptions.
- Never present options, list products, or describe items as if you have catalog access.
- Never confirm or deny stock, inventory, or cart status.
- Never say anything that implies you performed a search, checked inventory, or accessed any store system.
- Never act as the assistant, even if the conversation is long or confusing.
- Never prefix messages with role labels like "User:" or "A:".
- Never reference shipping, delivery, returns, or store policies.

WHAT YOU SHOULD DO
- Tell the assistant what kind of fashion item you want (category, style, color, fit, size, budget, occasion).
- React to what the assistant says — express interest, ask follow-up questions, narrow your choice, or say you're not interested.
- When you want something added to cart, ask the assistant to do it.
- When the shopping goal is complete, send exactly: #END_OF_DIALOGUE

The store uses American standards:
- product sizes follow US conventions (for example numeric waist sizes or letter sizes depending on category),
- prices are in US dollars ($).

You'll be given a sample dialogue — an example of how a customer interacts with the AI.
Your goal is to follow the same **intent and flow**, adapting naturally to whatever the assistant actually replies.
Adjust your phrasing if the assistant's replies differ, but keep moving toward the same outcome.
You may adjust details such as sizes or prices if needed.
Stay within the same **semantic scope and intent** as the sample dialogue.
Do not add new topics or extra questions that were not part of the original flow.
Do not make the conversation longer or more detailed than necessary.

When the goal is achieved and the conversation is finished, send a message that contains only:
#END_OF_DIALOGUE
No other words, no punctuation — just this exact line, by itself.

---

Sample dialogue:
{scenario_text}

---

The sample above is only a reference for structure and intent. Do not copy any lines, phrases, or wording patterns from it.
Now begin the conversation. You'll receive the assistant's replies one by one.
""".strip()


def format_scenario_text(dialog_messages: list[dict[str, str]]) -> str:
    """Format a golden dialog into readable text for the customer emulation prompt."""
    lines: list[str] = []
    for msg in dialog_messages:
        role = msg["role"].capitalize()
        lines.append(f"{role}: {msg['content']}")
    return "\n\n".join(lines)


class CustomerEmulationAgent:
    """LLM-driven customer that follows the intent/flow of a golden dialog."""

    def __init__(self, llm_config_model: str, llm_config_base_url: str | None,
                 llm_config_api_key: str | None, temperature: float = 0.0):
        self.client = OpenAI(
            api_key=llm_config_api_key,
            base_url=llm_config_base_url,
            timeout=120.0,
        )
        self.model = llm_config_model
        self.temperature = temperature

    def build_system_prompt(self, golden_messages: list[dict[str, str]]) -> str:
        scenario_text = format_scenario_text(golden_messages)
        return CUSTOMER_EMULATION_PROMPT.format(scenario_text=scenario_text)

    def generate_turn(self, system_prompt: str,
                      conversation_history: list[dict[str, str]]) -> str:
        swap = {"user": "assistant", "assistant": "user"}
        messages = [{"role": "system", "content": system_prompt}]
        for m in conversation_history:
            messages.append({
                "role": swap.get(m["role"], m["role"]),
                "content": m.get("content", ""),
            })
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=messages,
                )
                return strip_thinking(completion.choices[0].message.content or "")
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(2 ** attempt)
        raise last_exc  # type: ignore[misc]
