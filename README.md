# Document-Based Conversational AI

This project implements a conversational AI system for interacting with documents. The system uses **Retrieval-Augmented Generation (RAG)** and various techniques for scaling, caching, and maintaining conversation history.

## Features
- **Document chunking and embedding** for efficient search.
- **Multi-GPU support** for scalability.
- **Chat history management** to maintain context.
- **Caching** for repeated queries.

## Run the Application
To run the application locally:

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Start the FastAPI app:

    ```bash
    python run.py
    ```

The API will be available at `http://localhost:8000`.

## API Endpoints
### POST `/get_response/`
Send a user query and receive a response based on the document interactions.

Example request:

```json
{
  "query": "Tell me about AI"
}
