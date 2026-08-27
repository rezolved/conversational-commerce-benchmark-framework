from __future__ import annotations

import unittest

from benchmark.conversation import run_conversation


class FakeCustomer:
    def build_system_prompt(self, golden_messages: list[dict[str, str]]) -> str:
        return "customer prompt"

    def generate_turn(
        self,
        system_prompt: str,
        conversation: list[dict[str, str]],
    ) -> str:
        return "#END_OF_DIALOGUE"


class FakeAssistant:
    def run(
        self,
        user_query: str,
        conversation_history: list[dict],
    ) -> dict:
        return {
            "assistant_message": "I found one option.",
            "tool_messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "catalog_search",
                                "arguments": '{"query":"black jeans"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "catalog_search",
                    "tool_call_id": "call-1",
                    "content": '[{"sku":"J-1","title":"Black Jeans"}]',
                },
            ],
            "top_results": [{"sku": "J-1"}],
            "refinement_summary": "",
        }


class ConversationToolMessageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.golden_messages = [{"role": "user", "content": "Find black jeans"}]

    def test_tool_messages_are_preserved_by_default(self) -> None:
        result = run_conversation(
            FakeCustomer(),
            FakeAssistant(),
            self.golden_messages,
            max_turns=1,
        )

        self.assertEqual(
            [message["role"] for message in result["result_dialog"]],
            ["user", "assistant", "tool", "assistant"],
        )
        self.assertEqual(
            result["result_dialog"][1]["tool_calls"][0]["function"]["name"],
            "catalog_search",
        )
        self.assertEqual(result["result_dialog"][2]["tool_call_id"], "call-1")

    def test_tool_messages_can_be_explicitly_omitted(self) -> None:
        result = run_conversation(
            FakeCustomer(),
            FakeAssistant(),
            self.golden_messages,
            max_turns=1,
            include_tool_messages=False,
        )

        self.assertEqual(
            [message["role"] for message in result["result_dialog"]],
            ["user", "assistant"],
        )


if __name__ == "__main__":
    unittest.main()
