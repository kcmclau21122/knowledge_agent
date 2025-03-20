"""
Web Server module for serving the Knowledge Agent API
"""

import logging
import os
from typing import List, Dict, Any
import shutil

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Depends, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
import traceback

from config import Config
from knowledge_agent import KnowledgeAgent
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

            # Chat history for this session
            chat_history = []
            
            try:
                while True:
                    # Receive and parse the client message
                    data = await websocket.receive_json()
                    query = data.get("message", "")
                    logger.info(f"Received WebSocket message: {query[:50]}...")
                    
                    # Add to chat history
                    chat_history.append({"role": "user", "content": query})
                    
                    # Get response - Make sure response is properly captured
                    response = self.knowledge_agent.answer_question(query, chat_history)
                    logger.info(f"Sending response: {response[:50]}...")
                    
                    # Add to chat history
                    chat_history.append({"role": "assistant", "content": response})
                    
                    # Send response - ensure proper JSON formatting
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
        
        # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            """
            Health check endpoint
            
            Returns:
                JSON response with status
            """
            return {"status": "ok", "version": Config.VERSION}
    
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
