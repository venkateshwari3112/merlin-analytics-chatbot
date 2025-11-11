"""
Chatbot Backend API for Merlin Analytics - Production Version
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from groq import Groq
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Initialize components
print("🚀 Initializing chatbot backend...")

# Load embedding model
print("📦 Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the data
print("💾 Loading company data...")
data_path = os.path.join(os.path.dirname(__file__), 'data', 'merlin_data.pkl')
with open(data_path, 'rb') as f:
    data = pickle.load(f)
    chunks = data['chunks']
    embeddings = data['embeddings']

print(f"   Loaded {len(chunks)} chunks")

# Initialize Groq client
print("🤖 Setting up Groq API...")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("⚠️  WARNING: GROQ_API_KEY not found in environment variables!")
else:
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("✅ Backend ready!")

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_relevant_context(question, n_results=3):
    """Retrieve relevant information using cosine similarity"""
    question_embedding = embedding_model.encode([question])[0]
    
    similarities = []
    for i, emb in enumerate(embeddings):
        sim = cosine_similarity(question_embedding, emb)
        similarities.append((i, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in similarities[:n_results]]
    
    context = "\n\n".join([chunks[i] for i in top_indices])
    return context

def generate_answer(question, context):
    """Generate answer using Groq LLM"""
    system_prompt = """You are a helpful assistant for Merlin Analytics, a specialist consultancy 
focused on EPM-based finance transformations. Answer questions about the company using the provided context. 
Be professional, concise, and friendly. If you don't know something based on the context, say so politely."""
    
    user_prompt = f"""Context about Merlin Analytics:
{context}

Question: {question}

Please provide a helpful answer based on the context above."""
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=500
        )
        
        return chat_completion.choices[0].message.content
    
    except Exception as e:
        print(f"Error generating answer: {str(e)}")
        return f"Sorry, I encountered an error generating the response. Please try again."

@app.route('/health', methods=['GET'])
def health_check():
    """Check if the API is running"""
    return jsonify({
        "status": "healthy", 
        "message": "Chatbot backend is running!",
        "chunks_loaded": len(chunks)
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    if not GROQ_API_KEY:
        return jsonify({
            "error": "Service configuration error. Please contact administrator.",
            "success": False
        }), 500
    
    try:
        data = request.json
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({"error": "No question provided", "success": False}), 400
        
        print(f"❓ Question: {question}")
        
        context = retrieve_relevant_context(question)
        answer = generate_answer(question, context)
        
        print(f"✅ Answer generated")
        
        return jsonify({
            "question": question,
            "answer": answer,
            "success": True
        })
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({
            "error": "An error occurred processing your request.",
            "success": False
        }), 500

@app.route('/')
def home():
    """Serve the chat interface"""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Merlin Analytics - AI Assistant</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .chat-container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            width: 100%;
            max-width: 800px;
            height: 600px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            text-align: center;
        }
        .chat-header h1 { font-size: 24px; margin-bottom: 5px; }
        .chat-header p { font-size: 14px; opacity: 0.9; }
        .messages-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }
        .message {
            margin-bottom: 20px;
            display: flex;
            animation: fadeIn 0.3s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.user { justify-content: flex-end; }
        .message-content {
            max-width: 70%;
            padding: 15px 20px;
            border-radius: 18px;
            line-height: 1.5;
        }
        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .message.bot .message-content {
            background: white;
            color: #333;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        .typing-indicator {
            display: none;
            padding: 15px 20px;
            background: white;
            border-radius: 18px;
            max-width: 70%;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        .typing-indicator.active { display: block; }
        .typing-indicator span {
            height: 10px;
            width: 10px;
            background: #667eea;
            border-radius: 50%;
            display: inline-block;
            margin-right: 5px;
            animation: bounce 1.4s infinite;
        }
        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes bounce {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }
        .input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
            display: flex;
            gap: 10px;
        }
        #userInput {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 15px;
            outline: none;
            transition: border-color 0.3s;
        }
        #userInput:focus { border-color: #667eea; }
        #sendButton {
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        #sendButton:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        #sendButton:active { transform: translateY(0); }
        #sendButton:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .welcome-message {
            text-align: center;
            color: #666;
            margin-top: 50px;
            padding: 20px;
        }
        .welcome-message h2 { color: #667eea; margin-bottom: 10px; }
        .suggestions {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
            margin-top: 20px;
        }
        .suggestion-btn {
            padding: 10px 20px;
            background: white;
            border: 2px solid #667eea;
            color: #667eea;
            border-radius: 20px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
        }
        .suggestion-btn:hover {
            background: #667eea;
            color: white;
        }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #f1f1f1; }
        ::-webkit-scrollbar-thumb { background: #667eea; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>🧙‍♂️ Merlin Analytics AI Assistant</h1>
            <p>Ask me anything about Merlin Analytics</p>
        </div>
        <div class="messages-container" id="messagesContainer">
            <div class="welcome-message">
                <h2>Welcome! 👋</h2>
                <p>I'm here to help you learn about Merlin Analytics. Try asking:</p>
                <div class="suggestions">
                    <button class="suggestion-btn" onclick="askQuestion('What does Merlin Analytics do?')">What does Merlin Analytics do?</button>
                    <button class="suggestion-btn" onclick="askQuestion('Who are the directors?')">Who are the directors?</button>
                    <button class="suggestion-btn" onclick="askQuestion('Tell me about your team')">Tell me about your team</button>
                </div>
            </div>
            <div class="typing-indicator" id="typingIndicator">
                <span></span><span></span><span></span>
            </div>
        </div>
        <div class="input-container">
            <input type="text" id="userInput" placeholder="Type your question here..." onkeypress="handleKeyPress(event)"/>
            <button id="sendButton" onclick="sendMessage()">Send</button>
        </div>
    </div>
    <script>
        const messagesContainer = document.getElementById('messagesContainer');
        const userInput = document.getElementById('userInput');
        const sendButton = document.getElementById('sendButton');
        const typingIndicator = document.getElementById('typingIndicator');

        function addMessage(content, isUser) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = content;
            messageDiv.appendChild(contentDiv);
            const welcomeMsg = document.querySelector('.welcome-message');
            if (welcomeMsg) welcomeMsg.remove();
            messagesContainer.insertBefore(messageDiv, typingIndicator);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function showTyping() {
            typingIndicator.classList.add('active');
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function hideTyping() {
            typingIndicator.classList.remove('active');
        }

        async function sendMessage() {
            const question = userInput.value.trim();
            if (!question) return;
            
            userInput.disabled = true;
            sendButton.disabled = true;
            addMessage(question, true);
            userInput.value = '';
            showTyping();
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question })
                });
                const data = await response.json();
                hideTyping();
                
                if (data.success) {
                    addMessage(data.answer, false);
                } else {
                    addMessage('Sorry, I encountered an error. Please try again.', false);
                }
            } catch (error) {
                hideTyping();
                addMessage('Sorry, I could not process your request.', false);
                console.error('Error:', error);
            }
            
            userInput.disabled = false;
            sendButton.disabled = false;
            userInput.focus();
        }

        function askQuestion(question) {
            userInput.value = question;
            sendMessage();
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter') sendMessage();
        }

        userInput.focus();
    </script>
</body>
</html>"""

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)