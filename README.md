# PDF QA

PDF QA is a command-line tool that allows you to ask questions about the content of PDF documents using various Language Models (LLMs).

## Features

- Supports multiple LLMs: OpenAI, Anthropic, and Google
- Handles both local PDF files and online PDF URLs
- Maintains conversation history for each PDF session
- Provides concise answers based on the PDF content

## Installation

You can install PDF QA using pip:

bash
pip install pdf_qa

## Usage

To use PDF QA, run the following command:

bash
pdf_qa <pdf_path_or_url> [options]


### Options

- `-l, --llm`: Choose the LLM to use (openai, anthropic, or google). Default is openai.
- `-q, --question`: Ask a single question and exit.

### Examples

1. Interactive mode with a local PDF:
   ```bash
   pdf_qa /path/to/your/document.pdf
   ```

2. Using a specific LLM with an online PDF:
   ```bash
   pdf_qa https://example.com/sample.pdf -l anthropic
   ```

3. Asking a single question:
   ```bash
   pdf_qa /path/to/your/document.pdf -q "What is the main topic of this document?"
   ```

## API Keys

On first use, you'll be prompted to enter your API key for the chosen LLM. The key will be saved for future use.

## Chat History

Chat histories are saved in the `chat_histories` directory, allowing you to resume conversations in future sessions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.