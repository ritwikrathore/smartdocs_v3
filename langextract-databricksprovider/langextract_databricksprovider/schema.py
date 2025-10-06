"""Schema implementation for DatabricksProvider provider."""

import langextract as lx
from langextract import schema


class DatabricksProviderSchema(lx.schema.BaseSchema):
    """Schema implementation for DatabricksProvider structured output."""

    def __init__(self, schema_dict: dict):
        """Initialize the schema with a dictionary."""
        self._schema_dict = schema_dict

    @property
    def schema_dict(self) -> dict:
        """Return the schema dictionary."""
        return self._schema_dict

    @classmethod
    def from_examples(cls, examples_data, attribute_suffix="_attributes"):
        """Build schema from example extractions.

        Args:
            examples_data: Sequence of ExampleData objects.
            attribute_suffix: Suffix for attribute fields.

        Returns:
            A configured DatabricksProviderSchema instance.
        """
        extraction_types = {}
        for example in examples_data:
            for extraction in example.extractions:
                class_name = extraction.extraction_class
                if class_name not in extraction_types:
                    extraction_types[class_name] = set()
                if extraction.attributes:
                    extraction_types[class_name].update(extraction.attributes.keys())

        schema_dict = {
            "type": "object",
            "properties": {
                "extractions": {
                    "type": "array",
                    "items": {"type": "object"}
                }
            },
            "required": ["extractions"]
        }

        return cls(schema_dict)

    def to_provider_config(self) -> dict:
        """Convert to provider-specific configuration.

        Returns:
            Dictionary of provider-specific configuration.
        """
        return {
            "response_schema": self._schema_dict,
            "structured_output": True
        }

    @property
    def supports_strict_mode(self) -> bool:
        """Whether this schema guarantees valid structured output.

        Returns:
            True if the provider enforces valid JSON output.
        """
        return False  # Set to True only if your provider guarantees valid JSON
