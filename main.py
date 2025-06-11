"""
FastAPI Main Application for Maverick AI
Robust web server with WebSocket support, static file serving, and chat functionality

Author: AI Assistant
Version: 2.1.0
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Import your Maverick AI class
try:
    from maverick import Maverick, create_maverick, Config
except ImportError:
    # Fallback if maverick module not available
    logging.warning("Maverick module not found. Using mock implementation.")
    
    class MockMaverick:
        def __init__(self, model: str = "mock-model"):
            self.model = model
            
        async def chat_async(self, message: str) -> str:
            return f"Mock response to: {message}"
            
        def get_stats(self) -> Dict:
            return {
                "total_messages": 0,
                "available_search_providers": 1,
                "model": self.model,
                "memory_window": 50
            }
            
        def clear_conversation(self) -> str:
            return "Conversation cleared (mock)"
    
    def create_maverick(model: str = "mock-model"):
        return MockMaverick(model)
    
    Maverick = MockMaverick

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fastapi_maverick.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Pydantic models for API
class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: datetime
    search_used: bool = False

class SessionInfo(BaseModel):
    session_id: str
    created_at: datetime
    message_count: int
    last_message: Optional[str] = None

class StatsResponse(BaseModel):
    total_messages: int
    available_search_providers: int
    model: str
    memory_window: int
    active_sessions: int
    uptime_seconds: float

# WebSocket Connection Manager
class ConnectionManager:
    """Manages WebSocket connections for real-time chat"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_mavericks: Dict[str, "Maverick"] = {}
        
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept WebSocket connection and create Maverick instance"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.session_mavericks[session_id] = create_maverick()
        logger.info(f"WebSocket connected for session: {session_id}")
        
    def disconnect(self, session_id: str):
        """Remove connection and cleanup"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.session_mavericks:
            del self.session_mavericks[session_id]
        logger.info(f"WebSocket disconnected for session: {session_id}")
        
    async def send_message(self, message: str, session_id: str):
        """Send message to specific session"""
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to {session_id}: {e}")
                self.disconnect(session_id)
                
    def get_maverick(self, session_id: str) -> Optional["Maverick"]:
        """Get Maverick instance for session"""
        return self.session_mavericks.get(session_id)

# Initialize FastAPI app
app = FastAPI(
    title="Maverick AI Server",
    description="Intelligent AI Assistant with Real-time Web Search",
    version="2.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Connection manager
manager = ConnectionManager()

# Application state
app_start_time = datetime.now()
request_count = 0

# Static files
static_path = Path("static")
if not static_path.exists():
    static_path.mkdir()
    logger.warning("Static directory created. Please add your static files.")

app.mount("/static", StaticFiles(directory="static"), name="static")

# Middleware to count requests
@app.middleware("http")
async def count_requests(request: Request, call_next):
    global request_count
    request_count += 1
    start_time = datetime.now()
    
    response = await call_next(request)
    
    process_time = (datetime.now() - start_time).total_seconds()
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# Routes

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the main HTML page"""
    try:
        html_path = static_path / "index.html"
        if html_path.exists():
            return HTMLResponse(content=html_path.read_text(), status_code=200)
        else:
            # Fallback HTML if index.html not found
            return HTMLResponse(content="""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Maverick AI - Setup Required</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .container { max-width: 600px; margin: 0 auto; }
                    .error { background: #fee; border: 1px solid #fcc; padding: 20px; border-radius: 8px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Maverick AI Server</h1>
                    <div class="error">
                        <h3>Setup Required</h3>
                        <p>Please add your static files (index.html, styles.css, script.js) to the <code>static/</code> directory.</p>
                        <p>Server is running on: <a href="/api/docs">/api/docs</a></p>
                    </div>
                </div>
            </body>
            </html>
            """, status_code=200)
    except Exception as e:
        logger.error(f"Error serving index: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat"""
    await manager.connect(websocket, session_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")
            
            if not user_message.strip():
                continue
                
            # Get Maverick instance for this session
            maverick = manager.get_maverick(session_id)
            if not maverick:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Session not found. Please refresh the page."
                }))
                continue
            
            # Send typing indicator
            await websocket.send_text(json.dumps({
                "type": "typing",
                "message": "Maverick is thinking..."
            }))
            
            try:
                # Get AI response
                ai_response = await maverick.chat_async(user_message)
                
                # Send response back to client
                response_data = {
                    "type": "message",
                    "message": ai_response,
                    "timestamp": datetime.now().isoformat(),
                    "session_id": session_id
                }
                
                await websocket.send_text(json.dumps(response_data))
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Sorry, I encountered an error processing your message. Please try again."
                }))
                
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(session_id)

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatMessage):
    """REST API endpoint for chat (alternative to WebSocket)"""
    try:
        session_id = chat_request.session_id or str(uuid.uuid4())
        
        # Get or create Maverick instance
        if session_id not in manager.session_mavericks:
            manager.session_mavericks[session_id] = create_maverick()
        
        maverick = manager.session_mavericks[session_id]
        
        # Process message
        response = await maverick.chat_async(chat_request.message)
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            timestamp=datetime.now(),
            search_used=True  # This would be determined by the actual search usage
        )
        
    except Exception as e:
        logger.error(f"Chat API error: {e}")
        raise HTTPException(status_code=500, detail="Error processing chat message")

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get application statistics"""
    try:
        # Get stats from any Maverick instance or create temporary one
        maverick_stats = {"total_messages": 0, "available_search_providers": 0, "model": "unknown", "memory_window": 50}
        
        if manager.session_mavericks:
            sample_maverick = next(iter(manager.session_mavericks.values()))
            maverick_stats = sample_maverick.get_stats()
        
        uptime = (datetime.now() - app_start_time).total_seconds()
        
        return StatsResponse(
            total_messages=maverick_stats.get("total_messages", 0),
            available_search_providers=maverick_stats.get("available_search_providers", 0),
            model=maverick_stats.get("model", "unknown"),
            memory_window=maverick_stats.get("memory_window", 50),
            active_sessions=len(manager.active_connections),
            uptime_seconds=uptime
        )
        
    except Exception as e:
        logger.error(f"Stats API error: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving stats")

@app.post("/api/sessions/{session_id}/clear")
async def clear_session(session_id: str):
    """Clear conversation history for a session"""
    try:
        if session_id in manager.session_mavericks:
            maverick = manager.session_mavericks[session_id]
            result = maverick.clear_conversation()
            return {"message": result, "session_id": session_id}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Clear session error: {e}")
        raise HTTPException(status_code=500, detail="Error clearing session")

@app.get("/api/sessions")
async def get_sessions():
    """Get list of active sessions"""
    try:
        sessions = []
        for session_id in manager.session_mavericks.keys():
            sessions.append({
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),  # Would be tracked in real implementation
                "message_count": 0,  # Would be tracked in real implementation
                "is_active": session_id in manager.active_connections
            })
        
        return {"sessions": sessions, "total": len(sessions)}
        
    except Exception as e:
        logger.error(f"Sessions API error: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving sessions")

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session"""
    try:
        if session_id in manager.session_mavericks:
            manager.disconnect(session_id)
            return {"message": f"Session {session_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Delete session error: {e}")
        raise HTTPException(status_code=500, detail="Error deleting session")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": (datetime.now() - app_start_time).total_seconds(),
        "active_sessions": len(manager.active_connections),
        "total_requests": request_count
    }

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("🚀 Maverick AI FastAPI server starting up...")
    logger.info(f"Static files directory: {static_path.absolute()}")
    
    # Check if static files exist
    required_files = ["index.html", "styles.css", "script.js"]
    missing_files = []
    
    for file_name in required_files:
        if not (static_path / file_name).exists():
            missing_files.append(file_name)
    
    if missing_files:
        logger.warning(f"Missing static files: {missing_files}")
        logger.warning("Please add these files to the static/ directory")
    else:
        logger.info("✅ All static files found")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Maverick AI FastAPI server shutting down...")
    
    # Disconnect all WebSocket connections
    for session_id in list(manager.active_connections.keys()):
        manager.disconnect(session_id)
    
    logger.info("✅ Cleanup completed")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"The requested path '{request.url.path}' was not found",
            "suggestions": [
                "Check the URL spelling",
                "Visit /api/docs for API documentation",
                "Go to / for the main application"
            ]
        }
    )

@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    """Custom 500 handler"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

# Development server function
def run_dev_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = True):
    """Run development server"""
    logger.info(f"Starting Maverick AI server on http://{host}:{port}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        access_log=True,
        server_header=False,
        date_header=False
    )

if __name__ == "__main__":
    # Check environment variables
    required_env_vars = ["GROQ_API_KEY"]
    missing_env_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_env_vars:
        logger.warning(f"Missing environment variables: {missing_env_vars}")
        logger.warning("Some features may not work properly")
    
    # Run development server
    run_dev_server()