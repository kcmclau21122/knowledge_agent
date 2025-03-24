"""
Chat Session Manager module for maintaining chat history across sessions
"""

import logging
import time
import json
import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import threading

from config import Config

logger = logging.getLogger(__name__)

class ChatSessionManager:
    def __init__(self, storage_dir=None, session_expiry=3600, save_interval=300):
        """Initialize the chat session manager
        
        Args:
            storage_dir: Directory to store persistent session data
            session_expiry: Time in seconds until a session expires (default: 1 hour)
            save_interval: Time in seconds between auto-saves (default: 5 minutes)
        """
        self.storage_dir = storage_dir or Path(Config.DATA_DIR) / "sessions"
        self.session_expiry = session_expiry
        self.save_interval = save_interval
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # In-memory storage for active sessions
        self.sessions = {}
        self.last_activity = {}  # Track when each session was last active
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Load existing sessions
        self._load_sessions()
        
        # Start background cleanup thread
        self._start_cleanup_thread()
    
    def create_session(self) -> str:
        """Create a new chat session
        
        Returns:
            Session ID string
        """
        session_id = str(uuid.uuid4())
        
        with self.lock:
            self.sessions[session_id] = []
            self.last_activity[session_id] = time.time()
            
            logger.info(f"Created new session: {session_id}")
        
        return session_id
    
    def get_session_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get chat history for a session
        
        Args:
            session_id: Session ID
            
        Returns:
            List of chat messages
        """
        with self.lock:
            # Update last activity time
            if session_id in self.last_activity:
                self.last_activity[session_id] = time.time()
            
            # Return chat history (empty list if session doesn't exist)
            return self.sessions.get(session_id, []).copy()
    
    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """Add a message to the session history
        
        Args:
            session_id: Session ID
            role: Message role ('user' or 'assistant')
            content: Message content
            
        Returns:
            True if successful, False if session doesn't exist
        """
        if not content or not content.strip():
            return False
            
        with self.lock:
            # Check if session exists
            if session_id not in self.sessions:
                logger.warning(f"Attempted to add message to non-existent session: {session_id}")
                return False
            
            # Add message to history
            self.sessions[session_id].append({
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update last activity time
            self.last_activity[session_id] = time.time()
            
            logger.debug(f"Added {role} message to session {session_id[:8]}...")
            
        # Auto-save after adding messages
        self._auto_save(session_id)
        
        return True
    
    def _session_path(self, session_id: str) -> Path:
        """Get the file path for a session"""
        return self.storage_dir / f"{session_id}.json"
    
    def _load_sessions(self):
        """Load saved sessions from disk"""
        try:
            session_files = list(self.storage_dir.glob("*.json"))
            logger.info(f"Found {len(session_files)} saved sessions")
            
            loaded_count = 0
            for session_file in session_files:
                try:
                    session_id = session_file.stem
                    
                    # Skip if already in memory
                    if session_id in self.sessions:
                        continue
                    
                    with open(session_file, "r") as f:
                        session_data = json.load(f)
                    
                    # Check if session has expired
                    last_activity = session_data.get("last_activity", 0)
                    if time.time() - last_activity > self.session_expiry:
                        # Delete expired session file
                        os.remove(session_file)
                        continue
                    
                    # Load session data
                    self.sessions[session_id] = session_data.get("messages", [])
                    self.last_activity[session_id] = last_activity
                    loaded_count += 1
                    
                except Exception as e:
                    logger.error(f"Error loading session from {session_file}: {e}")
            
            logger.info(f"Loaded {loaded_count} active sessions")
            
        except Exception as e:
            logger.error(f"Error loading sessions: {e}")
    
    def save_session(self, session_id: str) -> bool:
        """Save a session to disk
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            # Check if session exists
            if session_id not in self.sessions:
                return False
            
            session_path = self._session_path(session_id)
            
            try:
                # Prepare session data
                session_data = {
                    "messages": self.sessions[session_id],
                    "last_activity": self.last_activity[session_id]
                }
                
                # Save to disk
                with open(session_path, "w") as f:
                    json.dump(session_data, f)
                
                logger.debug(f"Saved session {session_id[:8]}...")
                return True
                
            except Exception as e:
                logger.error(f"Error saving session {session_id}: {e}")
                return False
    
    def _auto_save(self, session_id: str):
        """Auto-save a session if enough messages have been added"""
        # Currently just delegates to save_session
        # Could be enhanced with batching or delayed saving
        self.save_session(session_id)
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions from memory and disk"""
        current_time = time.time()
        expired_sessions = []
        
        with self.lock:
            # Find expired sessions
            for session_id, last_active in list(self.last_activity.items()):
                if current_time - last_active > self.session_expiry:
                    expired_sessions.append(session_id)
            
            # Remove expired sessions from memory
            for session_id in expired_sessions:
                if session_id in self.sessions:
                    del self.sessions[session_id]
                if session_id in self.last_activity:
                    del self.last_activity[session_id]
                
                # Try to remove from disk
                try:
                    session_path = self._session_path(session_id)
                    if os.path.exists(session_path):
                        os.remove(session_path)
                except Exception as e:
                    logger.error(f"Error removing expired session file for {session_id}: {e}")
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def _start_cleanup_thread(self):
        """Start background thread for periodic cleanup"""
        def cleanup_task():
            while True:
                time.sleep(self.session_expiry / 4)  # Run cleanup periodically
                try:
                    self.cleanup_expired_sessions()
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")
        
        # Start as daemon thread so it doesn't prevent app shutdown
        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()
    
    def get_active_session_count(self) -> int:
        """Get the number of active sessions"""
        with self.lock:
            return len(self.sessions)
