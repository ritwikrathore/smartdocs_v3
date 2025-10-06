# Custom Provider Guide for langextract

## Introduction

This guide provides a step-by-step process for creating a custom model provider for the `langextract` library. A custom provider is necessary when you want to use a model or endpoint that is not supported by the built-in providers (e.g., Gemini, OpenAI, Ollama). By creating a custom provider, you can extend `langextract` to work with any model that has a Python client library.

This guide will walk you through the process of creating a custom provider for a Databricks-hosted Llama model, but the same principles can be applied to any other model or endpoint.

## Prerequisites

Before you begin, ensure you have the following installed:

*   Python 3.10 or higher
*   `pip` (the Python package installer)
*   `git` (the version control system)
*   A virtual environment tool (e.g., `venv`)

## Step 1: Generate the Boilerplate

The `langextract` library provides a script to generate the boilerplate code for a new custom provider. This script is part of the `langextract` source code, so you will need to clone the repository to use it.

1.  **Clone the `langextract` repository:**

    ```bash
    git clone https://github.com/google/langextract.git /tmp/langextract_source
    ```

2.  **Generate the provider boilerplate:**

    Run the `create_provider_plugin.py` script to generate the boilerplate for your new provider. Replace `MyProvider` with the name of your provider (e.g., `DatabricksProvider`).

    ```bash
    python3 /tmp/langextract_source/scripts/create_provider_plugin.py MyProvider --with-schema
    ```

    This will create a new directory named `langextract-myprovider` in your current working directory. This directory contains all the necessary files to get you started.

## Step 2: Implement the Provider

Now that you have the boilerplate, you need to modify the `provider.py` file to implement the logic for your custom provider. This file is located in the `langextract-myprovider/langextract_myprovider/` directory.

Here's an example of how to implement a custom provider for a Databricks-hosted Llama model:

```python
import os
import langextract as lx
from openai import OpenAI
from langextract_databricksprovider.schema import DatabricksProviderSchema


@lx.providers.registry.register(r'^databricks-', priority=10)
class DatabricksProviderLanguageModel(lx.inference.BaseLanguageModel):
    """LangExtract provider for DatabricksProvider."""

    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        """Initialize the DatabricksProvider provider."""
        super().__init__()
        self.model_id = model_id
        self.api_key = api_key or os.environ.get('DATABRICKS_TOKEN')

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://adb-3858882779799477.17.azuredatabricks.net/serving-endpoints"
        )

    @classmethod
    def get_schema_class(cls):
        """Tell LangExtract about our schema support."""
        return DatabricksProviderSchema

    def infer(self, batch_prompts, **kwargs):
        """Run inference on a batch of prompts."""
        for prompt in batch_prompts:
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an AI assistant that extracts information from documents."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model=self.model_id,
                    max_tokens=5000
                )
                result = chat_completion.choices[0].message.content
                yield [lx.inference.ScoredOutput(score=1.0, output=result)]
            except Exception as e:
                raise lx.core.exceptions.InferenceRuntimeError(f"Databricks API error: {e}") from e
```

**Key changes in this implementation:**

*   **Import `OpenAI`:** The `openai` library is imported to interact with the Databricks endpoint.
*   **Initialize `OpenAI` client:** The `__init__` method initializes the `OpenAI` client with the Databricks API key and endpoint URL.
*   **Implement `infer` method:** The `infer` method makes the API call to the Databricks endpoint using the `self.client.chat.completions.create` method.
*   **Update `@lx.providers.registry.register` decorator:** The regular expression in the decorator is updated to match the naming convention of your Databricks models (e.g., `r'^databricks-'`).

## Step 3: Install the Custom Provider

Once you have implemented your custom provider, you need to install it in the same virtual environment where `langextract` is installed. You can do this using `pip` in editable mode.

1.  **Navigate to your custom provider directory:**

    ```bash
    cd langextract-myprovider
    ```

2.  **Install the provider in editable mode:**

    ```bash
    venv/bin/python -m pip install -e .
    ```

    This will install the provider in a way that allows you to make changes to the code without having to reinstall it every time.

## Step 4: Use the Custom Provider

After you have installed your custom provider, you can use it in your scripts just like any other provider. `langextract` will automatically discover your provider based on the `model_id` you provide.

Here's an example of how to use the custom Databricks provider:

```python
import langextract as lx
import os
from dotenv import load_dotenv

load_dotenv()

# It is recommended to set the API key as an environment variable.
if 'DATABRICKS_TOKEN' not in os.environ:
    print('Please set the DATABRICKS_TOKEN environment variable.')
    exit()

# 1. Define the prompt and extraction rules
prompt = textwrap.dedent("""
    Extract the following information from the financial agreement:
    - Final Maturity Date
    - Relevant Spread
    - Total Loan Amount
    - A (Refinancing) Loan Amount
    - A (Greenfield) Loan Amount
    """)

# 2. Provide a high-quality example to guide the model
examples = [
    lx.data.ExampleData(
        text=textwrap.dedent("""
            "Final Maturity Date" means 31 December 2045;

            "Relevant Spread" means one point five zero percent (1.50%) per annum;

            Section 2.01 The Loan. (a) Subject to the provisions of this Agreement, IFC agrees to lend,
            and the Borrower agrees to borrow, the Loan in the aggregate principal amount of up to US Dollar One
            billion (USD 1,000,000,000) consisting of:

            (i) the A (Refinancing) Loan, being up to US Dollar five hundred
            million (USD 500,000,000); and
            (ii) the A (Greenfield) Loan, being up to US Dollar five hundred million
            (USD 500,000,000).
        """),
        extractions=[
            lx.data.Extraction(
                extraction_class="Final Maturity Date",
                extraction_text="31 December 2045",
            ),
            lx.data.Extraction(
                extraction_class="Relevant Spread",
                extraction_text="one point five zero percent (1.50%) per annum",
            ),
            lx.data.Extraction(
                extraction_class="Total Loan Amount",
                extraction_text="US Dollar One billion (USD 1,000,000,000)",
            ),
            lx.data.Extraction(
                extraction_class="A (Refinancing) Loan Amount",
                extraction_text="US Dollar five hundred million (USD 500,000,000)",
            ),
            lx.data.Extraction(
                extraction_class="A (Greenfield) Loan Amount",
                extraction_text="US Dollar five hundred million (USD 500,000,000)",
            ),
        ]
    )
]

# 3. The input text to be processed
input_text = textwrap.dedent("""
    "Final Maturity Date" means 15 June 2030;

    "Relevant Spread" means two point nine zero percent (3.90%) per annum;

    Section 2.01 The Loan. (a) Subject to the provisions of this Agreement, IFC agrees to lend,
    and the Borrower agrees to borrow, the Loan in the aggregate principal amount of up to US Dollar One
    thousand six hundred and seventy six million (USD 1,676,000,000) consisting of:

    (i) the A (Refinancing) Loan, being up to US Dollar eight hundred and eighty eight
    million (USD 888,000,000); and
    (ii) the A (Greenfield) Loan, being up to US Dollar seven hundred and eighty eight million
    (USD 788,000,000).
    """)

# 4. Run the extraction
result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="databricks-llama-4-maverick",
)

# 5. Save the results and create a visualization
lx.io.save_annotated_documents([result], output_name="extraction_results.jsonl", output_dir=".")

# Generate the interactive visualization from the file
html_content = lx.visualize("extraction_results.jsonl")
with open("visualization.html", "w") as f:
    if hasattr(html_content, 'data'):
        f.write(html_content.data)  # For Jupyter/Colab
    else:
        f.write(html_content)

print("Extraction complete. Open visualization.html to see the results.")
```

## Troubleshooting

*   **`InferenceConfigError: No provider registered for model_id`:** This error occurs if `langextract` cannot find a provider that matches the `model_id` you have provided. Ensure that the regular expression in your `@lx.providers.registry.register` decorator matches the `model_id` you are using in your script.
*   **`ModuleNotFoundError: No module named 'langextract_myprovider'`:** This error occurs if your custom provider is not installed correctly. Make sure you have run `pip install -e .` from within your custom provider directory.
*   **API errors:** If you are getting errors from your model's API, make sure that your API key and endpoint URL are correct in your provider's `__init__` method.
