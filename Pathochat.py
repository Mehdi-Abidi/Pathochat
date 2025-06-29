import os
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from datetime import datetime
import time
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="PathoCare AI - Medical Pathology Assistant", 
    page_icon="🔬", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Streamlit Cloud optimized file path handling
@st.cache_data
def get_database_path():
    """Get the correct database path for different deployment environments"""
    try:
        # Check multiple possible locations
        possible_paths = [
            "vector_store/faiss_database",  # Relative to app root
            "./vector_store/faiss_database",  # Current directory
            os.path.join(os.getcwd(), "vector_store", "faiss_database"),  # Absolute path
            "/app/vector_store/faiss_database",  # Streamlit Cloud path
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found database at: {path}")
                return path
        
        # If no database found, return the most likely path for error reporting
        return possible_paths[0]
    except Exception as e:
        logger.error(f"Error determining database path: {e}")
        return "vector_store/faiss_database"

# Enhanced CSS with better mobile responsiveness
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 25%, #334155 50%, #1e293b 75%, #0f172a 100%);
        font-family: 'Inter', sans-serif;
        color: #f1f5f9;
    }
    
    /* Medical brand header - Optimized for cloud */
    .medical-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(51, 65, 85, 0.9) 100%);
        border: 2px solid rgba(148, 163, 184, 0.2);
        border-radius: 20px;
        padding: 2rem 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        text-align: center;
        position: relative;
    }
    
    .medical-title {
        font-size: clamp(1.8rem, 4vw, 3rem);
        font-weight: 700;
        background: linear-gradient(135deg, #94a3b8 0%, #cbd5e1 50%, #e2e8f0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .medical-subtitle {
        font-size: clamp(1rem, 2.5vw, 1.3rem);
        color: #cbd5e1;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    
    .medical-description {
        color: #94a3b8;
        font-size: clamp(0.9rem, 2vw, 1rem);
        margin-top: 1rem;
    }
    
    /* Status cards - Mobile optimized */
    .status-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .status-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.6) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease;
        position: relative;
    }
    
    .status-card:hover {
        transform: translateY(-4px);
    }
    
    .status-icon {
        font-size: 2rem;
        margin-bottom: 0.8rem;
        display: block;
    }
    
    .status-title {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 0.3rem;
    }
    
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    
    .indicator-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.6);
        animation: pulse-glow 2s infinite;
    }
    
    @keyframes pulse-glow {
        0%, 100% { 
            transform: scale(1);
            box-shadow: 0 0 12px rgba(16, 185, 129, 0.6);
        }
        50% { 
            transform: scale(1.1);
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.8);
        }
    }
    
    /* Enhanced chat interface - Cloud optimized */
    .user-message {
        background: linear-gradient(135deg, #475569 0%, #334155 50%, #1e293b 100%);
        color: #f8fafc;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 8px 20px;
        margin: 1rem 5% 1rem 10%;
        box-shadow: 0 6px 20px rgba(71, 85, 105, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        word-wrap: break-word;
        animation: slideInRight 0.3s ease-out;
    }
    
    .assistant-message {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.8) 100%);
        color: #f1f5f9;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 8px;
        margin: 1rem 10% 1rem 5%;
        border-left: 4px solid #10b981;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(16, 185, 129, 0.2);
        word-wrap: break-word;
        animation: slideInLeft 0.3s ease-out;
    }
    
    @keyframes slideInRight {
        from { transform: translateX(30px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-30px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Input styling - Cloud optimized */
    .stTextArea textarea {
        background: rgba(15, 23, 42, 0.7) !important;
        border: 2px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 12px !important;
        color: #f1f5f9 !important;
        font-size: 1rem !important;
        padding: 1rem !important;
        min-height: 100px !important;
        resize: vertical !important;
        transition: all 0.3s ease !important;
        line-height: 1.5 !important;
    }
    
    .stTextArea textarea:focus {
        border-color: rgba(148, 163, 184, 0.5) !important;
        box-shadow: 0 0 20px rgba(148, 163, 184, 0.2) !important;
        outline: none !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #94a3b8 !important;
        font-style: italic !important;
    }
    
    /* Loading animation */
    .loading-dots {
        display: inline-flex;
        gap: 4px;
        align-items: center;
    }
    
    .loading-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #94a3b8;
        animation: loading-bounce 1.4s ease-in-out infinite both;
    }
    
    .loading-dot:nth-child(1) { animation-delay: -0.32s; }
    .loading-dot:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes loading-bounce {
        0%, 80%, 100% { 
            transform: scale(0);
            opacity: 0.5;
        }
        40% { 
            transform: scale(1);
            opacity: 1;
        }
    }
    
    /* Button styling - Cloud optimized */
    .stButton > button {
        background: linear-gradient(135deg, #475569 0%, #334155 100%) !important;
        color: #f8fafc !important;
        border: 2px solid rgba(148, 163, 184, 0.3) !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #64748b 0%, #475569 100%) !important;
        border-color: rgba(148, 163, 184, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Footer styling */
    .medical-footer {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.8) 100%);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    .disclaimer-box {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
        color: #fca5a5;
        font-size: 0.9rem;
    }
    
    /* Hide Streamlit elements */
    .stDeployButton { display: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .medical-header { padding: 1.5rem 1rem; }
        .user-message, .assistant-message { 
            margin-left: 2%; 
            margin-right: 2%; 
            padding: 0.8rem 1rem;
        }
        .status-grid { grid-template-columns: 1fr; }
        .status-card { padding: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

# Cache the vector store loading with better error handling
@st.cache_resource
def load_vector_store():
    """Load vector store with comprehensive error handling for Streamlit Cloud"""
    try:
        # Get the database path
        db_path = get_database_path()
        
        # Log current environment info
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Attempting to load from: {db_path}")
        logger.info(f"Database path exists: {os.path.exists(db_path)}")
        
        # List directory contents for debugging
        if os.path.exists(os.path.dirname(db_path) or "."):
            dir_contents = os.listdir(os.path.dirname(db_path) or ".")
            logger.info(f"Directory contents: {dir_contents}")
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # Ensure CPU usage for cloud deployment
        )
        
        # Load the database
        if os.path.exists(db_path):
            db = FAISS.load_local(
                db_path, 
                embeddings=embeddings, 
                allow_dangerous_deserialization=True
            )
            logger.info("Vector database loaded successfully!")
            return db, None
        else:
            error_msg = f"Database not found at {db_path}. Please ensure the vector_store directory is included in your repository."
            logger.error(error_msg)
            return None, error_msg
            
    except Exception as e:
        error_msg = f"Failed to load vector store: {str(e)}"
        logger.error(error_msg)
        return None, error_msg

# Prompt template
PROMPT_TEMPLATE = """
You are PathoCare AI, a medical pathology assistant. Generate accurate answers using only the provided context.

Guidelines:
- Use only the context provided below
- If the context doesn't contain the answer, respond with: "I don't have sufficient information in my medical database to answer this query accurately."
- Provide clear, concise medical information
- Include relevant diagnostic insights when available

Context:
{context}

Question: {question}

Answer:
"""

def get_prompt(template):
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

# Environment-aware HF token handling
@st.cache_data
def get_hf_token():
    """Get HuggingFace token from environment or secrets"""
    # Try Streamlit secrets first (for cloud deployment)
    try:
        if hasattr(st, 'secrets') and 'HF_TOKEN' in st.secrets:
            return st.secrets['HF_TOKEN']
    except:
        pass
    
    # Try environment variable
    token = os.getenv("HF_TOKEN")
    if token:
        return token
    
    # If no token found, return None (will prompt user)
    return None

# LLM Endpoint config with better error handling
@st.cache_resource
def get_hf_endpoint(hf_rep_id, hf_token):
    """Initialize HuggingFace endpoint with error handling"""
    try:
        if not hf_token:
            return None, "HuggingFace token not provided"
        
        llm = HuggingFaceEndpoint(
            repo_id=hf_rep_id,
            temperature=0.5,
            max_new_tokens=1024,
            huggingfacehub_api_token=hf_token,
            timeout=30  # Add timeout for cloud deployment
        )
        return llm, None
    except Exception as e:
        error_msg = f"Failed to initialize LLM: {str(e)}"
        logger.error(error_msg)
        return None, error_msg

def display_chat_message(role, content):
    """Display a chat message with enhanced styling"""
    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            <strong>👨‍⚕️ Doctor:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        # Format assistant response
        parts = content.split("\n\nSource Docs:\n")
        main_answer = parts[0]
        sources = parts[1] if len(parts) > 1 else ""
        
        st.markdown(f"""
        <div class="assistant-message">
            <strong>🔬 PathoCare AI:</strong><br>
            {main_answer}
        </div>
        """, unsafe_allow_html=True)
        
        if sources and len(sources.strip()) > 0:
            with st.expander("📚 View Medical Literature Sources", expanded=False):
                st.text(sources)

def show_setup_instructions():
    """Show setup instructions if database or token is missing"""
    st.warning("⚠️ Setup Required")
    
    st.markdown("""
    ### For Streamlit Cloud Deployment:
    
    1. **Add your HuggingFace Token:**
       - Go to your Streamlit Cloud app settings
       - Add a secret: `HF_TOKEN = "your_huggingface_token_here"`
       - Get a token from: https://huggingface.co/settings/tokens
    
    2. **Upload Vector Database:**
       - Ensure your `vector_store/` folder is in your repository
       - The folder should contain your FAISS database files
       - Commit and push all files to your repository
    
    3. **Required Files Structure:**
       ```
       your-repo/
       ├── app.py (this file)
       ├── requirements.txt
       └── vector_store/
           └── faiss_database/
               ├── index.faiss
               └── index.pkl
       ```
    """)

def main():
    # Enhanced medical header
    st.markdown("""
    <div class="medical-header">
        <div class="medical-title">🔬 PathoCare AI</div>
        <div class="medical-subtitle">Advanced Pathology Diagnostic Assistant</div>
        <div class="medical-description">
            🩺 Precision diagnostics powered by AI • Evidence-based pathology consultation
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Check system status
    hf_token = get_hf_token()
    db, db_error = load_vector_store()
    
    # Status grid with actual system status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_color = "🟢" if (hf_token and db) else "🟡" if (hf_token or db) else "🔴"
        status_text = "Online & Ready" if (hf_token and db) else "Partial Setup" if (hf_token or db) else "Setup Required"
        
        st.markdown(f"""
        <div class="status-card">
            <div class="status-icon">{status_color}</div>
            <div class="status-title">System Status</div>
            <div class="status-value">
                <div class="status-indicator">
                    <div class="indicator-dot"></div>
                    {status_text}
                </div>
            </div>
            <small style="color: #94a3b8;">System health check</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="status-card">
            <div class="status-icon">🧠</div>
            <div class="status-title">AI Model</div>
            <div class="status-value">Mistral-7B</div>
            <small style="color: #94a3b8;">Medical Instruct v0.3</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        db_status = "Ready" if db else "Loading..."
        st.markdown(f"""
        <div class="status-card">
            <div class="status-icon">📚</div>
            <div class="status-title">Database</div>
            <div class="status-value">{db_status}</div>
            <small style="color: #94a3b8;">Medical literature</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Show setup instructions if needed
    if not hf_token or not db:
        show_setup_instructions()
        if db_error:
            st.error(f"Database Error: {db_error}")
        if not hf_token:
            st.error("HuggingFace token not found. Please add HF_TOKEN to your Streamlit secrets.")
        return
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display welcome message if no messages
    if not st.session_state.messages:
        st.markdown("""
        <div class="assistant-message">
            <strong>🔬 PathoCare AI:</strong><br>
            Welcome to your advanced pathology diagnostic assistant! I'm ready to help you with 
            evidence-based medical consultations, diagnostic insights, and pathological analysis.
            <br><br>
            <em>💡 Example queries:</em><br>
            • "What are the diagnostic features of influenza?"<br>
            • "Explain the pathophysiology of pneumonia"<br>
            • "What are the key symptoms of COVID-19?"
        </div>
        """, unsafe_allow_html=True)
    
    # Show previous messages
    for message in st.session_state.messages:
        display_chat_message(message['role'], message['content'])
    
    # Input section
    st.markdown("### 💬 Enter Your Medical Query")
    
    user_query = st.text_area(
        label="Medical query input",
        placeholder="Type your pathology or medical question here... (e.g., 'What are the key diagnostic features of influenza?')",
        key="medical_query",
        help="💡 Tip: Be specific with your medical queries for more accurate diagnostic insights",
        label_visibility="hidden",
        height=120
    )
    
    # Submit button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit_button = st.button(
            "🔍 Analyze Query",
            use_container_width=True,
            type="primary"
        )
    
    # Process query when submitted
    if submit_button and user_query.strip():
        # Add user message to session state
        st.session_state.messages.append({'role': 'user', 'content': user_query})
        
        # Show loading animation
        loading_placeholder = st.empty()
        with loading_placeholder.container():
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background: rgba(148, 163, 184, 0.1); border-radius: 12px; margin: 1rem 0;">
                <div class="loading-dots">
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                </div>
                <br>
                <span style="color: #94a3b8; font-weight: 500;">🔬 Analyzing pathology data and consulting medical literature...</span>
            </div>
            """, unsafe_allow_html=True)
        
        try:
            # Setup retrieval
            retriever = db.as_retriever(search_kwargs={"k": 4})
            hf_rep_id = "mistralai/Mistral-7B-Instruct-v0.3"
            llm, llm_error = get_hf_endpoint(hf_rep_id, hf_token)
            
            if llm is None:
                error_msg = f"❌ AI diagnostic engine error: {llm_error}"
                loading_placeholder.empty()
                st.error(error_msg)
                st.session_state.messages.append({'role': 'assistant', 'content': error_msg})
                st.rerun()
            
            prompt = get_prompt(PROMPT_TEMPLATE)
            
            # Setup RetrievalQA chain
            retrieval_qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )
            
            # Get response
            response = retrieval_qa.invoke({"query": user_query})
            
            result = response["result"]
            source_documents = response["source_documents"]
            
            # Format full response
            original_res = result + "\n\nSource Docs:\n" + str(source_documents)
            
            # Clear loading animation
            loading_placeholder.empty()
            
            # Add assistant message to session state
            st.session_state.messages.append({'role': 'assistant', 'content': original_res})
            
            # Rerun to display the new messages
            st.rerun()
            
        except Exception as e:
            loading_placeholder.empty()
            error_msg = f"❌ Diagnostic analysis error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({'role': 'assistant', 'content': error_msg})
            st.rerun()
    
    # Medical footer
    st.markdown("---")
    st.markdown("""
    <div class="medical-footer">
        <h3 style="color: #94a3b8; margin-bottom: 1rem;">🔬 PathoCare AI - Medical Pathology Assistant</h3>
        <p style="color: #cbd5e1; margin-bottom: 1rem;">
            Advanced AI-powered diagnostic support for healthcare professionals
        </p>
        <div class="disclaimer-box">
            <strong>⚠️ Medical Disclaimer:</strong><br>
            This AI assistant provides educational and informational content only. 
            All diagnostic decisions and treatment plans must be validated by qualified 
            healthcare professionals. Not intended for emergency medical situations.
        </div>
        <div style="margin-top: 1.5rem; color: #94a3b8; font-size: 0.9rem;">
            🏥 Developed for pathology education and clinical decision support<br>
            📞 For emergencies, contact your local emergency services immediately
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
