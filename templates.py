"""
Templates module for generating HTML and static files
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def create_templates():
    """Create the HTML templates for the web interface"""
    os.makedirs("templates", exist_ok=True)
    
    # Create index.html template with session management support
    index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ request.app.title }}</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        .chat-container {
            height: calc(100vh - 200px);
            overflow-y: auto;
        }
        .chat-bubble {
            position: relative;
            padding: 10px 15px;
            border-radius: 15px;
            margin-bottom: 10px;
            max-width: 80%;
        }
        .user-bubble {
            background-color: #e3f2fd;
            margin-left: auto;
            border-bottom-right-radius: 0;
        }
        .assistant-bubble {
            background-color: #f5f5f5;
            margin-right: auto;
            border-bottom-left-radius: 0;
        }
        .typing-indicator {
            display: none;
            margin-bottom: 10px;
        }
        .typing-indicator span {
            display: inline-block;
            width: 10px;
            height: 10px;
            background-color: #888;
            border-radius: 50%;
            margin: 0 2px;
            animation: typing 1s infinite;
        }
        .typing-indicator span:nth-child(2) {
            animation-delay: 0.2s;
        }
        .typing-indicator span:nth-child(3) {
            animation-delay: 0.4s;
        }
        @keyframes typing {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
        }
    </style>
</head>
<body class="bg-gray-100">
    <header class="bg-blue-600 text-white p-4">
        <h1 class="text-2xl font-bold">Knowledge Agent</h1>
    </header>
    
    <main class="container mx-auto p-4">
        <div class="bg-white rounded-lg shadow-md p-4">
            <div id="chat-container" class="chat-container mb-4 p-2">
                <div id="chat-messages"></div>
                <div class="typing-indicator" id="typing-indicator">
                    <div class="assistant-bubble" style="padding: 8px 12px;">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
            </div>
            
            <div class="flex">
                <input type="text" id="user-input" class="flex-grow border rounded-l p-2" placeholder="Ask a question...">
                <button id="send-btn" class="bg-blue-600 text-white px-4 py-2 rounded-r">Send</button>
            </div>
        </div>
        
        <div class="mt-4 text-center text-sm text-gray-500">
            <p>Session ID: <span id="session-id-display">Not connected</span></p>
        </div>
    </main>
    
    <script>
        // Get or create session ID from localStorage
        let sessionId = localStorage.getItem('knowledgeAgentSessionId');
        
        // Set up WebSocket connection with session ID if available
        const wsUrl = sessionId 
            ? `ws://${window.location.host}/ws?session_id=${sessionId}`
            : `ws://${window.location.host}/ws`;
        
        const socket = new WebSocket(wsUrl);
        const chatMessages = document.getElementById('chat-messages');
        const userInput = document.getElementById('user-input');
        const sendButton = document.getElementById('send-btn');
        const typingIndicator = document.getElementById('typing-indicator');
        const sessionIdDisplay = document.getElementById('session-id-display');
        
        // Chat history
        let chatHistory = [];
        let isWaitingForResponse = false;
        
        // Send message function
        function sendMessage() {
            const message = userInput.value.trim();
            if (!message || isWaitingForResponse) return;
            
            // Add user message to chat
            addMessage('user', message);
            
            // Show typing indicator
            typingIndicator.style.display = 'block';
            isWaitingForResponse = true;
            
            // Send to websocket with session ID
            socket.send(JSON.stringify({
                message: message,
                session_id: sessionId
            }));
            
            // Clear input
            userInput.value = '';
        }
        
        // Add message to chat
        function addMessage(role, content) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `chat-bubble ${role === 'user' ? 'user-bubble' : 'assistant-bubble'}`;
            
            // Process markdown-like formatting in content
            let formattedContent = content
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')  // Bold
                .replace(/\*(.*?)\*/g, '<em>$1</em>')              // Italic
                .replace(/- (.*?)\\n/g, '<li>$1</li>')             // List items
                .replace(/\\n/g, '<br>');                          // Line breaks
            
            messageDiv.innerHTML = formattedContent;
            chatMessages.appendChild(messageDiv);
            
            // Scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            // Add to local history (not the session history which is managed by the server)
            chatHistory.push({role, content});
        }
        
        // Load chat history from server response
        function loadChatHistory(messages) {
            // Clear existing messages
            chatMessages.innerHTML = '';
            chatHistory = [];
            
            // Add each message to the UI
            messages.forEach(msg => {
                addMessage(msg.role, msg.content);
            });
            
            console.log(`Loaded ${messages.length} messages from history`);
        }
        
        // Event listeners
        socket.onopen = () => {
            console.log('WebSocket connected');
            if (sessionId) {
                sessionIdDisplay.textContent = sessionId.substring(0, 8) + '...';
            }
        };
        
        socket.onmessage = (event) => {
            // Hide typing indicator
            typingIndicator.style.display = 'none';
            isWaitingForResponse = false;
            
            const data = JSON.parse(event.data);
            
            // Handle different message types
            if (data.type === 'session_init') {
                // Save the session ID
                sessionId = data.session_id;
                localStorage.setItem('knowledgeAgentSessionId', sessionId);
                console.log(`Session initialized: ${sessionId}`);
                sessionIdDisplay.textContent = sessionId.substring(0, 8) + '...';
                
                // Add welcome message if this is a new session
                if (!data.message) {
                    addMessage('assistant', 'Hello! How can I help you today?');
                }
            }
            else if (data.type === 'history') {
                // Load chat history from server
                loadChatHistory(data.messages);
            }
            else if (data.type === 'error') {
                // Display error message
                addMessage('assistant', data.message);
            }
            else if (data.message) {
                // Regular message
                addMessage('assistant', data.message);
            }
        };
        
        socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            typingIndicator.style.display = 'none';
            isWaitingForResponse = false;
            addMessage('assistant', 'Sorry, there was an error connecting to the server. Please try again later.');
        };
        
        // Handle reconnection
        socket.onclose = () => {
            console.log('WebSocket closed. Will attempt to reconnect in 3 seconds...');
            typingIndicator.style.display = 'none';
            isWaitingForResponse = false;
            
            setTimeout(() => {
                // Attempt reconnection with session ID
                window.location.reload();
            }, 3000);
        };
        
        // Clear session button
        function clearSession() {
            localStorage.removeItem('knowledgeAgentSessionId');
            window.location.reload();
        }
        
        sendButton.addEventListener('click', sendMessage);
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
        
        // Add clear session button
        const clearSessionBtn = document.createElement('button');
        clearSessionBtn.textContent = 'New Chat';
        clearSessionBtn.className = 'ml-2 bg-gray-300 hover:bg-gray-400 text-gray-800 px-3 py-1 rounded text-sm';
        clearSessionBtn.addEventListener('click', clearSession);
        document.querySelector('.mt-4').appendChild(clearSessionBtn);
    </script>
</body>
</html>"""
    
    with open("templates/index.html", "w") as f:
        f.write(index_html)
    
    # Create admin.html template with enhanced session management
    admin_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ request.app.title }} - Admin</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body class="bg-gray-100">
    <header class="bg-blue-600 text-white p-4">
        <h1 class="text-2xl font-bold">Knowledge Agent - Admin Panel</h1>
    </header>
    
    <main class="container mx-auto p-4">
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <!-- Document Upload -->
            <div class="bg-white rounded-lg shadow-md p-4">
                <h2 class="text-xl font-bold mb-4">Upload Documents</h2>
                <form id="upload-form" class="space-y-4">
                    <div>
                        <label class="block mb-2">Select Document</label>
                        <input type="file" id="document-file" class="w-full p-2 border rounded">
                    </div>
                    <button type="submit" class="bg-blue-600 text-white px-4 py-2 rounded">Upload</button>
                </form>
                <div id="upload-status" class="mt-4"></div>
            </div>
            
            <!-- Knowledge Base Management -->
            <div class="bg-white rounded-lg shadow-md p-4">
                <h2 class="text-xl font-bold mb-4">Knowledge Base Management</h2>
                <button id="ingest-btn" class="bg-green-600 text-white px-4 py-2 rounded w-full mb-4">
                    Ingest Knowledge Base
                </button>
                <div id="ingest-status" class="mt-4"></div>
            </div>
        </div>

        <!-- Session Management -->
        <div class="mt-4 bg-white rounded-lg shadow-md p-4">
            <h2 class="text-xl font-bold mb-4">Session Management</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                    <button id="list-sessions-btn" class="bg-blue-500 text-white px-4 py-2 rounded w-full mb-2">
                        List Active Sessions
                    </button>
                    <button id="cleanup-sessions-btn" class="bg-yellow-500 text-white px-4 py-2 rounded w-full">
                        Clean Up Expired Sessions
                    </button>
                </div>
                <div id="session-status" class="p-2 border rounded bg-gray-50 min-h-16"></div>
            </div>
        </div>
    </main>
    
    <script>
        // File upload
        const uploadForm = document.getElementById('upload-form');
        const uploadStatus = document.getElementById('upload-status');
        
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const fileInput = document.getElementById('document-file');
            const file = fileInput.files[0];
            
            if (!file) {
                uploadStatus.innerHTML = '<p class="text-red-600">Please select a file</p>';
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            uploadStatus.innerHTML = '<p class="text-blue-600">Uploading...</p>';
            
            try {
                const response = await fetch('/api/admin/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    uploadStatus.innerHTML = `<p class="text-green-600">${data.message}</p>`;
                    fileInput.value = '';
                } else {
                    uploadStatus.innerHTML = `<p class="text-red-600">Error: ${data.detail}</p>`;
                }
            } catch (error) {
                uploadStatus.innerHTML = `<p class="text-red-600">Error: ${error.message}</p>`;
            }
        });
        
        // Ingest knowledge base
        const ingestBtn = document.getElementById('ingest-btn');
        const ingestStatus = document.getElementById('ingest-status');
        
        ingestBtn.addEventListener('click', async () => {
            ingestStatus.innerHTML = '<p class="text-blue-600">Ingesting knowledge base...</p>';
            
            try {
                const response = await fetch('/api/admin/ingest', {
                    method: 'POST'
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    ingestStatus.innerHTML = `<p class="text-green-600">${data.message}</p>`;
                } else {
                    ingestStatus.innerHTML = `<p class="text-red-600">Error: ${data.detail}</p>`;
                }
            } catch (error) {
                ingestStatus.innerHTML = `<p class="text-red-600">Error: ${error.message}</p>`;
            }
        });

        // Session Management
        const listSessionsBtn = document.getElementById('list-sessions-btn');
        const cleanupSessionsBtn = document.getElementById('cleanup-sessions-btn');
        const sessionStatus = document.getElementById('session-status');
        
        listSessionsBtn.addEventListener('click', async () => {
            sessionStatus.innerHTML = '<p class="text-blue-600">Fetching active sessions...</p>';
            
            try {
                const response = await fetch('/api/admin/sessions');
                const data = await response.json();
                
                if (response.ok) {
                    sessionStatus.innerHTML = `<p class="text-green-600">${data.message}</p>`;
                } else {
                    sessionStatus.innerHTML = `<p class="text-red-600">Error: ${data.detail}</p>`;
                }
            } catch (error) {
                sessionStatus.innerHTML = `<p class="text-red-600">Error: ${error.message}</p>`;
            }
        });
        
        cleanupSessionsBtn.addEventListener('click', async () => {
            sessionStatus.innerHTML = '<p class="text-blue-600">Cleaning up expired sessions...</p>';
            
            try {
                const response = await fetch('/api/admin/sessions/cleanup', {
                    method: 'POST'
                });
                const data = await response.json();
                
                if (response.ok) {
                    sessionStatus.innerHTML = `<p class="text-green-600">${data.message}</p>`;
                } else {
                    sessionStatus.innerHTML = `<p class="text-red-600">Error: ${data.detail}</p>`;
                }
            } catch (error) {
                sessionStatus.innerHTML = `<p class="text-red-600">Error: ${error.message}</p>`;
            }
        });
    </script>
</body>
</html>"""
    
    with open("templates/admin.html", "w") as f:
        f.write(admin_html)
    
    # Create static directory
    os.makedirs("static", exist_ok=True)
    
    # Create custom.css
    custom_css = """/* Custom styles for Knowledge Agent */
.chat-bubble {
    position: relative;
    padding: 10px 15px;
    border-radius: 15px;
    margin-bottom: 10px;
    max-width: 80%;
}

.user-bubble {
    background-color: #e3f2fd;
    margin-left: auto;
    border-bottom-right-radius: 0;
}

.assistant-bubble {
    background-color: #f5f5f5;
    margin-right: auto;
    border-bottom-left-radius: 0;
}

/* Typing indicator animation */
.typing-indicator {
    display: none;
    padding: 10px;
}

.typing-indicator span {
    display: inline-block;
    width: 8px;
    height: 8px;
    background-color: #888;
    border-radius: 50%;
    margin: 0 2px;
    animation: typing 1s infinite;
}

.typing-indicator span:nth-child(2) {
    animation-delay: 0.2s;
}

.typing-indicator span:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes typing {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-5px); }
}

/* Session display */
.session-info {
    font-size: 0.8rem;
    color: #666;
    text-align: center;
    margin-top: 10px;
}
"""
    
    with open("static/custom.css", "w") as f:
        f.write(custom_css)
    
    logger.info("Templates and static files created with session management support")

def create_embedding_script():
    """Create a standalone script for generating the JavaScript embedding code"""
    embedding_script = """#!/usr/bin/env python
import argparse

def generate_embedding_code(server_url, button_text="Chat with Us", header_text="Customer Support"):
    code = f'''<script>
// Chatbot configuration
const chatbotConfig = {{
    serverUrl: '{server_url}',
    buttonText: '{button_text}',
    headerText: '{header_text}',
}};

// Create chat widget
const chatWidget = document.createElement('div');
chatWidget.innerHTML = `
    <div id="chatbot-widget" class="chatbot-widget">
        <div class="chatbot-button" id="chatbot-toggle">
            ${{chatbotConfig.buttonText}}
        </div>
        <div class="chatbot-container" id="chatbot-container" style="display: none;">
            <div class="chatbot-header">
                ${{chatbotConfig.headerText}}
                <span class="chatbot-close" id="chatbot-close">×</span>
            </div>
            <div class="chatbot-messages" id="chatbot-messages"></div>
            <div class="chatbot-typing" id="chatbot-typing">
                <span></span><span></span><span></span>
            </div>
            <div class="chatbot-input">
                <input type="text" id="chatbot-input" placeholder="Type your question...">
                <button id="chatbot-send">Send</button>
            </div>
            <div class="chatbot-footer">
                <span id="chatbot-session" class="chatbot-session"></span>
                <button id="chatbot-new" class="chatbot-new-btn">New Chat</button>
            </div>
        </div>
    </div>
`;

// Add styles
const styles = document.createElement('style');
styles.innerHTML = `
    .chatbot-widget {{
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
        font-family: Arial, sans-serif;
    }}
    .chatbot-button {{
        background-color: #0084ff;
        color: white;
        padding: 10px 20px;
        border-radius: 20px;
        cursor: pointer;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        text-align: center;
    }}
    .chatbot-container {{
        position: absolute;
        bottom: 60px;
        right: 0;
        width: 320px;
        height: 450px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        display: flex;
        flex-direction: column;
    }}
    .chatbot-header {{
        padding: 10px 15px;
        background-color: #0084ff;
        color: white;
        border-top-left-radius: 10px;
        border-top-right-radius: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-weight: bold;
    }}
    .chatbot-close {{
        cursor: pointer;
        font-size: 20px;
    }}
    .chatbot-messages {{
        flex-grow: 1;
        padding: 10px;
        overflow-y: auto;
    }}
    .chatbot-message {{
        margin-bottom: 10px;
        padding: 8px 12px;
        border-radius: 15px;
        max-width: 85%;
        word-wrap: break-word;
    }}
    .chatbot-user {{
        background-color: #e3f2fd;
        margin-left: auto;
        border-bottom-right-radius: 0;
        text-align: right;
    }}
    .chatbot-bot {{
        background-color: #f5f5f5;
        margin-right: auto;
        border-bottom-left-radius: 0;
    }}
    .chatbot-typing {{
        display: none;
        padding: 8px 12px;
        background-color: #f5f5f5;
        border-radius: 15px;
        border-bottom-left-radius: 0;
        width: fit-content;
        margin-bottom: 10px;
    }}
    .chatbot-typing span {{
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #888;
        border-radius: 50%;
        margin: 0 2px;
        animation: chatbot-typing 1s infinite;
    }}
    .chatbot-typing span:nth-child(2) {{
        animation-delay: 0.2s;
    }}
    .chatbot-typing span:nth-child(3) {{
        animation-delay: 0.4s;
    }}
    @keyframes chatbot-typing {{
        0%, 100% {{ transform: translateY(0); }}
        50% {{ transform: translateY(-5px); }}
    }}
    .chatbot-input {{
        display: flex;
        padding: 10px;
        border-top: 1px solid #eee;
    }}
    .chatbot-input input {{
        flex-grow: 1;
        padding: 8px 10px;
        border: 1px solid #ddd;
        border-radius: 20px;
        outline: none;
    }}
    .chatbot-input button {{
        margin-left: 5px;
        padding: 8px 15px;
        background-color: #0084ff;
        color: white;
        border: none;
        border-radius: 20px;
        cursor: pointer;
    }}
    .chatbot-footer {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 10px;
        font-size: 11px;
        color: #666;
        border-top: 1px solid #eee;
    }}
    .chatbot-new-btn {{
        background-color: #f0f0f0;
        border: none;
        padding: 5px 8px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 11px;
    }}
`;

// Add to document
document.head.appendChild(styles);
document.body.appendChild(chatWidget);

// Chat functionality
let chatSocket = null;
let sessionId = localStorage.getItem('knowledgeAgentSessionId');
let chatHistory = [];
let isWaitingForResponse = false;

// Get elements
const container = document.getElementById('chatbot-container');
const toggleBtn = document.getElementById('chatbot-toggle');
const closeBtn = document.getElementById('chatbot-close');
const messagesDiv = document.getElementById('chatbot-messages');
const inputField = document.getElementById('chatbot-input');
const sendBtn = document.getElementById('chatbot-send');
const typingIndicator = document.getElementById('chatbot-typing');
const sessionDisplay = document.getElementById('chatbot-session');
const newChatBtn = document.getElementById('chatbot-new');

// Toggle chat widget
toggleBtn.addEventListener('click', () => {{
    if (container.style.display === 'none') {{
        container.style.display = 'flex';
        if (!chatSocket || chatSocket.readyState !== WebSocket.OPEN) {{
            connectWebSocket();
        }}
    }} else {{
        container.style.display = 'none';
    }}
}});

// Close chat widget
closeBtn.addEventListener('click', () => {{
    container.style.display = 'none';
}});

// New chat button
newChatBtn.addEventListener('click', () => {{
    // Clear session ID and reload
    localStorage.removeItem('knowledgeAgentSessionId');
    sessionId = null;
    
    // Clear messages
    messagesDiv.innerHTML = '';
    chatHistory = [];
    
    // Reconnect websocket
    if (chatSocket && chatSocket.readyState === WebSocket.OPEN) {{
        chatSocket.close();
    }}
    connectWebSocket();
}});

// Send message
function sendMessage() {{
    const message = inputField.value.trim();
    if (!message || isWaitingForResponse) return;
    
    // Add user message to chat
    addMessage('user', message);
    
    // Show typing indicator
    typingIndicator.style.display = 'block';
    isWaitingForResponse = true;
    
    // Send to websocket
    if (chatSocket && chatSocket.readyState === WebSocket.OPEN) {{
        chatSocket.send(JSON.stringify({{ 
            message,
            session_id: sessionId
        }}));
    }} else {{
        // Fallback to AJAX if WebSocket is not connected
        fetch(`${{chatbotConfig.serverUrl}}/api/chat`, {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ message, history: chatHistory }})
        }})
        .then(response => response.json())
        .then(data => {{
            typingIndicator.style.display = 'none';
            isWaitingForResponse = false;
            addMessage('bot', data.message);
        }})
        .catch(error => {{
            console.error('Error:', error);
            typingIndicator.style.display = 'none';
            isWaitingForResponse = false;
            addMessage('bot', 'Sorry, there was an error. Please try again.');
        }});
    }}
    
    // Clear input
    inputField.value = '';
}

// Add message to chat
function addMessage(role, content) {{
    const messageDiv = document.createElement('div');
    messageDiv.className = `chatbot-message chatbot-${{role === 'user' ? 'user' : 'bot'}}`;
    
    // Process markdown-like formatting in content
    let formattedContent = content
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')  // Bold
        .replace(/\*(.*?)\*/g, '<em>$1</em>')              // Italic
        .replace(/\\n/g, '<br>');                          // Line breaks
    
    messageDiv.innerHTML = formattedContent;
    messagesDiv.appendChild(messageDiv);
    
    // Scroll to bottom
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    
    // Add to history (for local tracking only)
    chatHistory.push({{ role: role === 'user' ? 'user' : 'assistant', content }});
}}

// Load chat history from server response
function loadChatHistory(messages) {{
    // Clear existing messages
    messagesDiv.innerHTML = '';
    chatHistory = [];
    
    // Add each message to the UI
    messages.forEach(msg => {{
        const role = msg.role === 'user' ? 'user' : 'bot';
        addMessage(role, msg.content);
    }});
    
    console.log(`Loaded ${{messages.length}} messages from history`);
}}

function connectWebSocket() {{
    // Use WebSocket or fallback to AJAX if WebSocket is not available
    if ('WebSocket' in window) {{
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = sessionId 
            ? `${{wsProtocol}}//${{chatbotConfig.serverUrl.replace(/^https?:\\/\\//, '')}}/ws?session_id=${{sessionId}}`
            : `${{wsProtocol}}//${{chatbotConfig.serverUrl.replace(/^https?:\\/\\//, '')}}/ws`;
        
        chatSocket = new WebSocket(wsUrl);
        
        chatSocket.onopen = () => {{
            console.log('WebSocket connected');
            if (sessionId) {{
                sessionDisplay.textContent = `Session: ${{sessionId.substring(0, 8)}}...`;
            }}
        }};
        
        chatSocket.onmessage = (event) => {{
            // Hide typing indicator
            typingIndicator.style.display = 'none';
            isWaitingForResponse = false;
            
            const data = JSON.parse(event.data);
            
            // Handle different message types
            if (data.type === 'session_init') {{
                // Save the session ID
                sessionId = data.session_id;
                localStorage.setItem('knowledgeAgentSessionId', sessionId);
                console.log(`Session initialized: ${{sessionId}}`);
                sessionDisplay.textContent = `Session: ${{sessionId.substring(0, 8)}}...`;
                
                // Add welcome message if this is a new session
                if (!data.message) {{
                    addMessage('bot', 'Hello! How can I help you today?');
                }}
            }}
            else if (data.type === 'history') {{
                // Load chat history from server
                loadChatHistory(data.messages);
            }}
            else if (data.type === 'error') {{
                // Display error message
                addMessage('bot', data.message);
            }}
            else if (data.message) {{
                // Regular message
                addMessage('bot', data.message);
            }}
        }};
        
        chatSocket.onerror = (error) => {{
            console.error('WebSocket error:', error);
            typingIndicator.style.display = 'none';
            isWaitingForResponse = false;
            addMessage('bot', 'Sorry, there was an error connecting to the chat. Please try again later.');
        }};
        
        chatSocket.onclose = () => {{
            console.log('WebSocket closed. Will attempt to reconnect in 5 seconds...');
            typingIndicator.style.display = 'none';
            isWaitingForResponse = false;
            
            // Attempt reconnection after 5 seconds
            setTimeout(connectWebSocket, 5000);
        }};
    }} else {{
        // Fallback for browsers without WebSocket support
        addMessage('bot', 'Hello! How can I help you today?');
    }}
}}

// Send message on button click
sendBtn.addEventListener('click', sendMessage);

// Send message on Enter key
inputField.addEventListener('keypress', (e) => {{
    if (e.key === 'Enter') sendMessage();
}});

// Connect WebSocket when the chat is first toggled
toggleBtn.addEventListener('click', function initialToggle() {{
    // Only run once
    toggleBtn.removeEventListener('click', initialToggle);
    
    // If container is now visible, connect
    if (container.style.display !== 'none') {{
        connectWebSocket();
    }}
}});

// For proactive opening (if needed)
function openChat() {{
    container.style.display = 'flex';
    if (!chatSocket || chatSocket.readyState !== WebSocket.OPEN) {{
        connectWebSocket();
    }}
}}

// Expose to window for external access
window.knowledgeAgent = {{
    openChat: openChat
}};
</script>'''
    
    return code

def main():
    parser = argparse.ArgumentParser(description="Generate chatbot embedding code")
    parser.add_argument("--server", required=True, help="Server URL (e.g., http://example.com:8000)")
    parser.add_argument("--button", default="Chat with Us", help="Button text")
    parser.add_argument("--header", default="Customer Support", help="Header text")
    parser.add_argument("--output", default="chatbot-embed.html", help="Output file")
    
    args = parser.parse_args()
    
    code = generate_embedding_code(args.server, args.button, args.header)
    
    with open(args.output, "w") as f:
        f.write(code)
    
    print(f"Embedding code generated and saved to {args.output}")

if __name__ == "__main__":
    main()
"""
    
    with open("generate_embed.py", "w") as f:
        f.write(embedding_script)