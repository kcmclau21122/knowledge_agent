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
    
    # Create index.html template
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
            </div>
            
            <div class="flex">
                <input type="text" id="user-input" class="flex-grow border rounded-l p-2" placeholder="Ask a question...">
                <button id="send-btn" class="bg-blue-600 text-white px-4 py-2 rounded-r">Send</button>
            </div>
        </div>
    </main>
    
    <script>
        // WebSocket connection
        const socket = new WebSocket(`ws://${window.location.host}/ws`);
        const chatMessages = document.getElementById('chat-messages');
        const userInput = document.getElementById('user-input');
        const sendButton = document.getElementById('send-btn');
        
        // Chat history
        let chatHistory = [];
        
        // Send message function
        function sendMessage() {
            const message = userInput.value.trim();
            if (!message) return;
            
            // Add user message to chat
            addMessage('user', message);
            
            // Send to websocket
            socket.send(JSON.stringify({message}));
            
            // Clear input
            userInput.value = '';
        }
        
        // Add message to chat
        function addMessage(role, content) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `mb-2 p-2 rounded ${role === 'user' ? 'bg-blue-100 ml-12' : 'bg-gray-100 mr-12'}`;
            messageDiv.innerHTML = `<p>${content}</p>`;
            chatMessages.appendChild(messageDiv);
            
            // Scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            // Add to history
            chatHistory.push({role, content});
        }
        
        // Event listeners
        socket.onopen = () => {
            console.log('WebSocket connected');
            addMessage('assistant', 'Hello! How can I help you today?');
        };
        
        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            addMessage('assistant', data.message);
        };
        
        socket.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        
        sendButton.addEventListener('click', sendMessage);
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
    </script>
</body>
</html>"""
    
    with open("templates/index.html", "w") as f:
        f.write(index_html)
    
    # Create admin.html template for document management
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
"""
    
    with open("static/custom.css", "w") as f:
        f.write(custom_css)
    
    logger.info("Templates and static files created")

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
            <div class="chatbot-input">
                <input type="text" id="chatbot-input" placeholder="Type your question...">
                <button id="chatbot-send">Send</button>
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
    }}
    .chatbot-container {{
        position: absolute;
        bottom: 60px;
        right: 0;
        width: 300px;
        height: 400px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        display: flex;
        flex-direction: column;
    }}
    .chatbot-header {{
        padding: 10px;
        background-color: #0084ff;
        color: white;
        border-top-left-radius: 10px;
        border-top-right-radius: 10px;
        display: flex;
        justify-content: space-between;
    }}
    .chatbot-close {{
        cursor: pointer;
    }}
    .chatbot-messages {{
        flex-grow: 1;
        padding: 10px;
        overflow-y: auto;
    }}
    .chatbot-message {{
        margin-bottom: 10px;
        padding: 8px;
        border-radius: 5px;
    }}
    .chatbot-user {{
        background-color: #e6f2ff;
        margin-left: 20px;
    }}
    .chatbot-bot {{
        background-color: #f0f0f0;
        margin-right: 20px;
    }}
    .chatbot-input {{
        display: flex;
        padding: 10px;
        border-top: 1px solid #eee;
    }}
    .chatbot-input input {{
        flex-grow: 1;
        padding: 5px;
        border: 1px solid #ddd;
        border-radius: 5px;
    }}
    .chatbot-input button {{
        margin-left: 5px;
        padding: 5px 10px;
        background-color: #0084ff;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }}
`;

// Add to document
document.head.appendChild(styles);
document.body.appendChild(chatWidget);

// Chat functionality
let chatSocket = null;
const chatHistory = [];

// Toggle chat widget
document.getElementById('chatbot-toggle').addEventListener('click', () => {{
    const container = document.getElementById('chatbot-container');
    if (container.style.display === 'none') {{
        container.style.display = 'flex';
        if (!chatSocket) {{
            connectWebSocket();
        }}
    }} else {{
        container.style.display = 'none';
    }}
}});

// Close chat widget
document.getElementById('chatbot-close').addEventListener('click', () => {{
    document.getElementById('chatbot-container').style.display = 'none';
}});

// Send message
document.getElementById('chatbot-send').addEventListener('click', sendMessage);
document.getElementById('chatbot-input').addEventListener('keypress', (e) => {{
    if (e.key === 'Enter') sendMessage();
}});

function connectWebSocket() {{
    // Use WebSocket or fallback to AJAX if WebSocket is not available
    if ('WebSocket' in window) {{
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${{wsProtocol}}//${{chatbotConfig.serverUrl.replace(/^https?:\\/\\//, '')}}/ws`;
        
        chatSocket = new WebSocket(wsUrl);
        
        chatSocket.onopen = () => {{
            console.log('WebSocket connected');
            addMessage('bot', 'Hello! How can I help you today?');
        }};
        
        chatSocket.onmessage = (event) => {{
            const data = JSON.parse(event.data);
            addMessage('bot', data.message);
        }};
        
        chatSocket.onerror = (error) => {{
            console.error('WebSocket error:', error);
            addMessage('bot', 'Sorry, there was an error connecting to the chat. Please try again later.');
        }};
        
        chatSocket.onclose = () => {{
            console.log('WebSocket closed');
            chatSocket = null;
        }};
    }} else {{
        // Fallback to AJAX if WebSocket is not available
        addMessage('bot', 'Hello! How can I help you today?');
    }}
}}

function sendMessage() {{
    const input = document.getElementById('chatbot-input');
    const message = input.value.trim();
    
    if (!message) return;
    
    addMessage('user', message);
    input.value = '';
    
    if (chatSocket && chatSocket.readyState === WebSocket.OPEN) {{
        chatSocket.send(JSON.stringify({{ message, history: chatHistory }}));
    }} else {{
        // Fallback to AJAX
        fetch(`${{chatbotConfig.serverUrl}}/api/chat`, {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ message, history: chatHistory }})
        }})
        .then(response => response.json())
        .then(data => {{
            addMessage('bot', data.message);
        }})
        .catch(error => {{
            console.error('Error:', error);
            addMessage('bot', 'Sorry, there was an error. Please try again.');
        }});
    }}
}}

function addMessage(role, content) {{
    const messagesDiv = document.getElementById('chatbot-messages');
    const messageDiv = document.createElement('div');
    
    messageDiv.className = `chatbot-message chatbot-${{role}}`;
    messageDiv.textContent = content;
    
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    
    // Save to history
    chatHistory.push({{ role, content }});
}}
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
