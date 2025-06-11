<!-- Usage Instructions &  Example  -->

# Maverick_AI Usage Guide

## Overview

Maverick is a sophisticated AI chatbot built with LangChain, featuring multi-source web search capabilities and real-time chat functionality. It provides both a command-line interface and a web server with WebSocket support.

## Features

- 🤖 Professional AI assistant with natural conversational abilities
- 🔍 Intelligent web search across multiple providers
- 🌐 Real-time WebSocket-based chat interface
- 🚀 REST API endpoints for integration
- 💾 Session-based conversation memory
- 📊 Usage statistics and health monitoring

## Quick Start

### Environment Setup

1. Create a `.env` file in the project root:
```bash
GROQ_API_KEY=your_groq_api_key
SERPER_API_KEY=your_serper_api_key  # Optional but recomended
SCRAPINGDOG_API_KEY=your_scrapingdog_api_key  # Optional
GROQ_MODEL = "mistral-saba-24b" #by default you can change 

```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Maverick

#### Command Line Interface
```bash
python maverick.py
```

#### Web Server
```bash
python main.py
```

## API Documentation

### WebSocket Endpoint

Connect to real-time chat:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/{session_id}');
```

### REST Endpoints

- `POST /api/chat` - Send chat messages
- `GET /api/stats` - Get system statistics
- `POST /api/sessions/{session_id}/clear` - Clear conversation history
- `GET /api/sessions` - List active sessions
- `DELETE /api/sessions/{session_id}` - Delete a session
- `GET /api/health` - Health check

## Docker Deployment

1. Build the image:
```bash
docker build -t maverick-ai .
```

2. Run the container:
```bash
docker run -it -p 8000:8000 maverick-ai
```

## Configuration

Key settings in `Config` class:
```python
MAX_SEARCH_RESULTS = 5
MAX_CONVERSATION_HISTORY = 50
DEFAULT_MODEL = "mistral-saba-24b"
REQUEST_TIMEOUT = 10
```

## Search Providers

Maverick supports multiple search providers:
- DuckDuckGo (default)
- Serper Google Search API (optional but recomended)
- ScrapingDog Google Search API (optional)

## Error Handling

Maverick includes robust error handling:
- Automatic retry for API failures
- Fallback search providers
- Graceful degradation of features
- Comprehensive logging

## Security Notes

- Uses non-root user in Docker
- Environment variable based configuration
- CORS middleware for web security
- Rate limiting on API endpoints
- Input validation and sanitization

