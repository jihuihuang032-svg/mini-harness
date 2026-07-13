from __future__ import annotations

import unittest

from harness.schema import SchemaValidationError, validate_args


class SchemaValidationTests(unittest.TestCase):
    def test_validate_args_applies_defaults(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "."},
                "limit": {"type": "integer", "default": 200},
            },
        }

        self.assertEqual(validate_args(schema, {}), {"path": ".", "limit": 200})

    def test_validate_args_requires_fields(self) -> None:
        schema = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
        with self.assertRaises(SchemaValidationError):
            validate_args(schema, {})

    def test_validate_args_rejects_wrong_type(self) -> None:
        schema = {"type": "object", "properties": {"limit": {"type": "integer"}}}
        with self.assertRaises(SchemaValidationError):
            validate_args(schema, {"limit": "10"})

    def test_validate_args_allows_unknown_args(self) -> None:
        schema = {"type": "object", "properties": {"path": {"type": "string"}}}
        self.assertEqual(validate_args(schema, {"path": ".", "extra": True}), {"path": ".", "extra": True})

    def test_bool_is_not_integer(self) -> None:
        schema = {"type": "object", "properties": {"limit": {"type": "integer"}}}
        with self.assertRaises(SchemaValidationError):
            validate_args(schema, {"limit": True})


if __name__ == "__main__":
    unittest.main()
