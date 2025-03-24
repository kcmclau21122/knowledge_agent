# Knowledge Agent - Open Source Botpress Alternative

An open-source chatbot system similar to Botpress for creating AI knowledge agents. This agent can be integrated into your website to answer customer questions using your company's knowledge base and an open source LLM.
knowldge
## Features

- **Document Processing**: Ingest various document formats (PDF, DOCX, TXT, MD, CSV, HTML)
- **Vector Search**: Efficient retrieval using ChromaDB vector database
- **Open Source LLMs**: Integration with Llama-2/3, Mistral, and other open source models
- **Web Interface**: Real-time chat interface for testing
- **API & WebSockets**: Integration options for your website
- **RAG Architecture**: Retrieval Augmented Generation for knowledge-grounded responses

## Project Structure

```
knowledge_agent/
├── config.py                # Configuration settings
├── document_processor.py    # Document loading and processing
├── vector_database.py       # Vector database operations
├── llm_manager.py           # LLM integration
├── knowledge_agent.py       # Core agent functionality
├── web_server.py            # Web server and API endpoints
├── templates.py             # HTML templates and static files
├── utils.py                 # Utility functions
├── main.py                  # Entry point script
```

## Requirements

- Python 3.9+ 
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/knowledge-agent.git
   cd knowledge-agent
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `data/knowledge` directory and add your documents:
   ```bash
   mkdir -p data/knowledge
   # Copy your PDFs, DOCXs, etc. to data/knowledge/
   ```

## Usage

### Adding Knowledge Documents

Place your company documents (PDFs, DOCXs, etc.) in the `data/knowledge` directory, then run:

```bash
python main.py --ingest
```

### Starting the Server

```bash
python main.py --company "Your Company Name"
```

Available options:
- `--company`: Your company name (used in prompts)
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 8000)
- `--knowledge-dir`: Custom knowledge directory
- `--ingest`: Ingest knowledge base
- `--model`: Custom LLM model name
- `--embedding-model`: Custom embedding model name

### Web Interface

Once running, you can access:
- Chat interface: http://localhost:8000/
- Admin interface: http://localhost:8000/admin

### Integrating with Your Website

Generate the embedding code:

```bash
python generate_embed.py --server "http://your-server-url:8000" --button "Chat with Us" --header "Customer Support"
```

Then include the generated code in your website.

## API Reference

### WebSocket

Connect to `ws://your-server:8000/ws` for real-time chat.

### REST API

- `POST /api/chat`: Chat endpoint
  ```json
  {
    "message": "What are your business hours?",
    "history": [
      {"role": "user", "content": "Hi there"},
      {"role": "assistant", "content": "Hello! How can I help you?"}
    ]
  }
  ```

- `POST /api/admin/upload`: Upload a document
  ```
  Form data with 'file' field
  ```

- `POST /api/admin/ingest`: Ingest the knowledge base

## Customization

### Using Different LLMs

Edit the model name in `config.py` or use the `--model` argument:

```bash
python main.py --model "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
```

Options include:
- `TheBloke/Llama-2-7B-Chat-GGUF`
- `TheBloke/Mistral-7B-Instruct-v0.1-GGUF`
- `TheBloke/Falcon-7B-Instruct-GGUF`
- Any other compatible model from Hugging Face

### Modifying the System Prompt

Edit the `SYSTEM_PROMPT` variable in `config.py` to customize the AI assistant's behavior and tone.

## Adding New Features

The modular design makes it easy to extend:

1. Document Processors: Add new format handlers in `document_processor.py`
2. LLM Integration: Support new models in `llm_manager.py`
3. API Endpoints: Add new routes in `web_server.py`

## License

MIT License
