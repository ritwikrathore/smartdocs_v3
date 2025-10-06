"""
Custom schema for Databricks provider in LangExtract.
"""

import langextract as lx
from langextract import schema
from typing import Dict, Any, List


class DatabricksProviderSchema(lx.schema.BaseSchema):
    """Schema class for Databricks provider."""
    
    def __init__(self, schema_dict: dict):
        self._schema_dict = schema_dict

    @property
    def schema_dict(self) -> dict:
        return self._schema_dict

    @classmethod
    def from_examples(cls, examples_data, attribute_suffix="_attributes"):
        """Build schema from example extractions."""
        extraction_types = {}
        
        for example in examples_data:
            for extraction in example.extractions:
                class_name = extraction.extraction_class
                if class_name not in extraction_types:
                    extraction_types[class_name] = set()
                if extraction.attributes:
                    extraction_types[class_name].update(extraction.attributes.keys())

        # Build a JSON schema for the extractions
        schema_dict = {
            "type": "object",
            "properties": {
                "extractions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "extraction_class": {"type": "string"},
                            "extraction_text": {"type": "string"},
                            "attributes": {
                                "type": "object",
                                "additionalProperties": {"type": "string"}
                            }
                        },
                        "required": ["extraction_class", "extraction_text"]
                    }
                }
            },
            "required": ["extractions"]
        }
        
        return cls(schema_dict)

    def to_provider_config(self) -> dict:
        """Convert to provider-specific configuration."""
        return {
            "response_schema": self._schema_dict,
            "structured_output": True
        }

    @property
    def supports_strict_mode(self) -> bool:
        """Return True if provider enforces valid JSON output."""
        return True
