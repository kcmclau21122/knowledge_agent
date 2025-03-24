"""
Web Server module for serving the Knowledge Agent API
"""

"""
Web Server module for serving the Knowledge Agent API
"""

import logging
import os
from typing import List, Dict, Any
import shutil
import uuid

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Depends, File, UploadFile, Form, Cookie, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
import traceback

from config import Config
from knowledge_agent import KnowledgeAgent
from chat_session_manager import ChatSessionManager
from utils import log_gpu_memory

logger = logging.getLogger(__name__)

# API Models
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    message: str

"""
Web Server module for serving the Knowledge Agent API
"""

import logging
import os
from typing import List, Dict, Any
import shutil

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Depends, File, UploadFile, Form, Cookie, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
import traceback

from config import Config
from knowledge_agent import KnowledgeAgent
from chat_session_manager import ChatSessionManager
from utils import log_gpu_memory

logger = logging.getLogger(__name__)

# API Models
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    message: str

class WebServer:
    def __init__(self, knowledge_agent: KnowledgeAgent):
        """Initialize the web server with the knowledge agent"""
        self.app = FastAPI(title=Config.APP_NAME, version=Config.VERSION)
        self.knowledge_agent = knowledge_agent
        
        # Initialize the session manager
        self.session_manager = ChatSessionManager()
        
        # Set up templates
        self.templates = Jinja2Templates(directory="templates")
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Mount static files
        if os.path.exists("static"):
            self.app.mount("/static", StaticFiles(directory="static"), name="static")
        
        # Register routes
        self.register_routes()
    
    def register_routes(self):
        """Register all the routes for the API"""
        # Web UI routes
        @self.app.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            return self.templates.TemplateResponse("index.html", {"request": request})
        
        @self.app.get("/admin", response_class=HTMLResponse)
        async def admin(request: Request):
            return self.templates.TemplateResponse("admin.html", {"request": request})
        
        # WebSocket for real-time chat
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            logger.info("WebSocket connection opened")

            # Get session ID from query parameters or create a new one
            query_params = websocket.query_params
            session_id = query_params.get("session_id", str(uuid.uuid4()))
            logger.info(f"Using session ID: {session_id}")
            
            # Initialize session if it doesn't exist
            if not hasattr(self, 'chat_sessions'):
                self.chat_sessions = {}
            
            # Create session if it doesn't exist
            if session_id not in self.chat_sessions:
                self.chat_sessions[session_id] = []
                logger.info(f"Created new session: {session_id}")
            
            # Chat history for this session
            chat_history = self.chat_sessions.get(session_id, [])
            
            try:
                while True:
                    # Receive and parse the client message
                    data = await websocket.receive_json()
                    query = data.get("message", "")
                    logger.info(f"Received WebSocket message: {query[:50]}...")
                    
                    # Add to chat history
                    chat_history.append({"role": "user", "content": query})
                    self.chat_sessions[session_id] = chat_history
                    
                    # Get response
                    response = self.knowledge_agent.answer_question(query, chat_history)
                    logger.info(f"Sending response: {response[:50]}...")
                    
                    # Add to chat history
                    chat_history.append({"role": "assistant", "content": response})
                    self.chat_sessions[session_id] = chat_history
                    
                    # Send response
                    await websocket.send_json({"message": response})
                    logger.info("Response sent successfully")
            
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
                logger.error(traceback.format_exc())
                try:
                    # Try to send error message to client
                    await websocket.send_json({"message": "I'm sorry, an error occurred while processing your request."})
                except:
                    # Connection might be closed already
                    pass

        # REST API endpoints
        @self.app.post("/api/chat", response_model=ChatResponse)
        async def chat(request: ChatRequest):
            """
            Chat API endpoint
            
            Args:
                request: ChatRequest with message and optional history
                
            Returns:
                ChatResponse with the agent's response
            """
            response = self.knowledge_agent.answer_question(
                request.message, 
                request.history
            )
            
            return ChatResponse(message=response)
        
        # Session management endpoints
        @self.app.get("/api/sessions/{session_id}")
        async def get_session(session_id: str):
            """
            Get session history
            
            Args:
                session_id: Session ID
                
            Returns:
                Chat history for the session
            """
            history = self.session_manager.get_session_history(session_id)
            if not history:
                raise HTTPException(status_code=404, detail="Session not found")
            
            return {"messages": history}
        
        # Admin API endpoints
        @self.app.post("/api/admin/upload")
        async def upload_document(file: UploadFile = File(...)):
            """
            Upload a document to the knowledge base
            
            Args:
                file: The document file to upload
                
            Returns:
                JSON response with status message
            """
            try:
                file_path = os.path.join(Config.KNOWLEDGE_DIR, file.filename)
                
                # Ensure knowledge directory exists
                os.makedirs(Config.KNOWLEDGE_DIR, exist_ok=True)
                
                # Save uploaded file
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(file.file, f)
                
                return {"message": f"File {file.filename} uploaded successfully"}
            except Exception as e:
                logger.error(f"Error uploading document: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/admin/ingest")
        async def ingest_knowledge():
            """
            Ingest the knowledge base
            
            Returns:
                JSON response with status message
            """
            try:
                num_docs = self.knowledge_agent.ingest_knowledge_base()
                return {"message": f"Knowledge base ingested successfully with {num_docs} document chunks"}
            except Exception as e:
                logger.error(f"Error ingesting knowledge base: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Additional admin endpoints for session management
        @self.app.get("/api/admin/sessions")
        async def list_sessions():
            """List all active sessions"""
            try:
                count = self.session_manager.get_active_session_count()
                return {"message": f"Active sessions: {count}"}
            except Exception as e:
                logger.error(f"Error listing sessions: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/admin/sessions/cleanup")
        async def cleanup_sessions():
            """Manually trigger session cleanup"""
            try:
                self.session_manager.cleanup_expired_sessions()
                return {"message": "Session cleanup triggered"}
            except Exception as e:
                logger.error(f"Error cleaning up sessions: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            """
            Health check endpoint
            
            Returns:
                JSON response with status
            """
            return {
                "status": "ok", 
                "version": Config.VERSION,
                "active_sessions": self.session_manager.get_active_session_count()
            }
    
    def run(self, host=None, port=None):
        """
        Run the web server
        
        Args:
            host: Host address to bind to (default from Config)
            port: Port to bind to (default from Config)
        """
        host = host or Config.HOST
        port = port or Config.PORT
        
        logger.info(f"Starting web server at http://{host}:{port}")
        uvicorn.run(self.app, host=host, port=port)