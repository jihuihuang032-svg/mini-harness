from __future__ import annotations

import unittest

from harness.actions import action_from_openai_message, parse_action


class ParseActionTests(unittest.TestCase):
    def test_parse_action_plain_json(self) -> None:
        action = parse_action('{"type":"final","content":"done"}')
        self.assertEqual(action.type, "final")
        self.assertEqual(action.content, "done")

    def test_parse_action_code_fence(self) -> None:
        action = parse_action('```json\n{"type":"final","content":"done"}\n```')
        self.assertEqual(action.type, "final")
        self.assertEqual(action.content, "done")

    def test_parse_action_rejects_missing_tool_name(self) -> None:
        with self.assertRaises(ValueError):
            parse_action('{"type":"tool_call","args":{}}')

    def test_parse_action_tool_call(self) -> None:
        action = parse_action('{"type":"tool_call","tool":"read_file","args":{"path":"README.md"}}')
        self.assertEqual(action.type, "tool_call")
        self.assertEqual(action.tool, "read_file")
        self.assertEqual(action.args, {"path": "README.md"})

    def test_parse_action_plan(self) -> None:
        action = parse_action('{"type":"plan","items":[{"id":"1","content":"Inspect","status":"pending"}]}')
        self.assertEqual(action.type, "plan")
        self.assertEqual(action.items, [{"id": "1", "content": "Inspect", "status": "pending"}])

    def test_parse_action_todo_update(self) -> None:
        action = parse_action('{"type":"todo_update","items":[{"id":"1","status":"completed"}]}')
        self.assertEqual(action.type, "todo_update")
        self.assertEqual(action.items, [{"id": "1", "status": "completed"}])

    def test_openai_tool_call_message_normalizes_to_action_json(self) -> None:
        text = action_from_openai_message(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{\"path\":\"README.md\"}",
                        },
                    }
                ],
            }
        )

        self.assertIsNotNone(text)
        action = parse_action(text or "")
        self.assertEqual(action.type, "tool_call")
        self.assertEqual(action.tool, "read_file")
        self.assertEqual(action.args, {"path": "README.md"})

    def test_openai_tool_call_rejects_non_object_arguments(self) -> None:
        with self.assertRaises(ValueError):
            action_from_openai_message(
                {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": "[\"README.md\"]",
                            }
                        }
                    ]
                }
            )


if __name__ == "__main__":
    unittest.main()
