import os
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="PathoCare AI - Medical Pathology Assistant", 
    page_icon="🔬", 
    layout="wide"
)

# Vector DB path
db_path = os.path.join(os.getcwd(), "faiss_database")
st.write("Current dir:", os.getcwd())
st.write("Files in dir:", os.listdir("faiss_database"))

st.write("DB path:", db_path)
st.write("All files:", os.listdir(db_path))


# Enhanced slate-themed CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 25%, #334155 50%, #1e293b 75%, #0f172a 100%);
        font-family: 'Inter', sans-serif;
        color: #f1f5f9;
    }
    
    /* Medical brand header */
    .medical-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(51, 65, 85, 0.9) 100%);
        border: 2px solid rgba(148, 163, 184, 0.2);
        border-radius: 20px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .medical-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(148, 163, 184, 0.05), transparent);
        animation: shimmer 8s infinite linear;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .medical-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #94a3b8 0%, #cbd5e1 50%, #e2e8f0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 2;
    }
    
    .medical-subtitle {
        font-size: 1.3rem;
        color: #cbd5e1;
        font-weight: 500;
        margin-bottom: 1rem;
        position: relative;
        z-index: 2;
    }
    
    .medical-description {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 1rem;
        position: relative;
        z-index: 2;
    }
    
    /* Status cards with slate theme */
    .status-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .status-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.6) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .status-card:hover {
        transform: translateY(-8px);
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.4),
            0 0 20px rgba(148, 163, 184, 0.1);
        border-color: rgba(148, 163, 184, 0.3);
    }
    
    .status-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #94a3b8, transparent);
        transition: left 0.5s ease;
    }
    
    .status-card:hover::before {
        left: 100%;
    }
    
    .status-icon {
        font-size: 2.2rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .status-title {
        font-size: 0.9rem;
        color: #94a3b8;
        font-weight: 500;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-value {
        font-size: 1.6rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 0.5rem;
    }
    
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    
    .indicator-dot {
        width: 12px;
        height: 12px;
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
    
    /* Enhanced chat interface */
    .chat-container {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 
            0 12px 48px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        min-height: 500px;
        max-height: 700px;
        overflow-y: auto;
        position: relative;
    }
    
    .chat-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.3), transparent);
    }
    
    /* Slate-themed message bubbles */
    .user-message {
        background: linear-gradient(135deg, #475569 0%, #334155 50%, #1e293b 100%);
        color: #f8fafc;
        padding: 1.2rem 1.8rem;
        border-radius: 20px 20px 8px 20px;
        margin: 1rem 0 1rem 15%;
        box-shadow: 
            0 6px 20px rgba(71, 85, 105, 0.3),
            0 2px 4px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        word-wrap: break-word;
        animation: slideInRight 0.3s ease-out;
    }
    
    .assistant-message {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.8) 100%);
        color: #f1f5f9;
        padding: 1.2rem 1.8rem;
        border-radius: 20px 20px 20px 8px;
        margin: 1rem 15% 1rem 0;
        border-left: 4px solid #10b981;
        box-shadow: 
            0 6px 20px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(16, 185, 129, 0.2);
        position: relative;
        word-wrap: break-word;
        animation: slideInLeft 0.3s ease-out;
    }
    
    @keyframes slideInRight {
        from { transform: translateX(50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Enhanced input section */
    .input-section {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.9) 0%, rgba(51, 65, 85, 0.8) 100%);
        border: 2px solid rgba(148, 163, 184, 0.2);
        border-radius: 20px;
        padding: 1.5rem;
        margin-top: 2rem;
        box-shadow: 
            0 12px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        position: relative;
    }
    
    .input-section::before {
        content: '';
        position: absolute;
        top: -1px;
        left: -1px;
        right: -1px;
        bottom: -1px;
        background: linear-gradient(135deg, rgba(148, 163, 184, 0.3), rgba(203, 213, 225, 0.3));
        border-radius: 20px;
        z-index: -1;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .input-section:hover::before {
        opacity: 1;
    }
    
    /* Dynamic textarea styling */
    .stTextArea textarea {
        background: rgba(15, 23, 42, 0.7) !important;
        border: 2px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 12px !important;
        color: #f1f5f9 !important;
        font-size: 1rem !important;
        padding: 1rem !important;
        min-height: 120px !important;
        resize: vertical !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease !important;
        line-height: 1.5 !important;
    }
    
    .stTextArea textarea:focus {
        border-color: rgba(148, 163, 184, 0.5) !important;
        box-shadow: 0 0 20px rgba(148, 163, 184, 0.2) !important;
        outline: none !important;
        min-height: 150px !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #94a3b8 !important;
        font-style: italic !important;
    }
    
    /* Medical footer */
    .medical-footer {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.8) 100%);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 16px;
        padding: 2rem;
        margin-top: 3rem;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    .disclaimer-box {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
        color: #fca5a5;
    }
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
    
    /* Enhanced scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #94a3b8, #cbd5e1);
        border-radius: 5px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #64748b, #94a3b8);
    }
    
    /* Hide Streamlit elements */
    .stDeployButton { display: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    
    /* Button styling */
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
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #64748b 0%, #475569 100%) !important;
        border-color: rgba(148, 163, 184, 0.5) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .medical-title { font-size: 2rem; }
        .user-message, .assistant-message { 
            margin-left: 5%; 
            margin-right: 5%; 
        }
        .status-grid { grid-template-columns: 1fr; }
    }
</style>
""", unsafe_allow_html=True)

# Cache the vector store loading
@st.cache_resource
def load_vector_store():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local(db_path,embeddings=embeddings, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.write(f"Loading from: {db_path}")
        st.error(f"Failed to load vector store: {str(e)}")
        return None

# Prompt template
PROMPT_TEMPLATE = """
You are a focused assistant. Your task is to generate accurate answers using only the context provided.
- Do not include external knowledge.
- Do not speculate. If the context lacks the answer, respond with: "I don't know."
- No fluff, greetings, or commentary — jump straight to the answer.

[Context]
{context}

[Question]
{question}

[Answer]
"""

def get_prompt(template):
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

# HF token
HF_TOKEN = os.getenv("HF_TOKEN")

# LLM Endpoint config
@st.cache_resource
def get_hf_endpoint(hf_rep_id):
    try:
        llm = HuggingFaceEndpoint(
            repo_id=hf_rep_id,
            temperature=0.5,
            provider="hf-inference",
            max_new_tokens=1024,
            huggingfacehub_api_token=HF_TOKEN
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize LLM: {str(e)}")
        return None

def display_chat_message(role, content):
    """Display a chat message with enhanced slate styling"""
    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            <strong>👨‍⚕️ Doctor:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        # Format assistant response to separate main answer from sources
        parts = content.split("\n\nSource Docs:\n")
        main_answer = parts[0]
        sources = parts[1] if len(parts) > 1 else ""
        
        st.markdown(f"""
        <div class="assistant-message">
            <strong>🔬 PathoCare AI:</strong><br>
            {main_answer}
        </div>
        """, unsafe_allow_html=True)
        
        if sources:
            with st.expander("📚 View Medical Literature Sources", expanded=False):
                st.text(sources)

def main():
    # Enhanced medical header
    st.markdown("""
    <div class="medical-header">
        <div class="medical-title">🔬 PathoCare AI</div>
        <div class="medical-subtitle">Advanced Pathology Diagnostic Assistant</div>
        <div class="medical-description">
            🩺 Precision diagnostics powered by AI • Evidence-based pathology consultation
            <br>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced status grid
    st.markdown('<div class="status-grid">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="status-card">
            <div class="status-icon">🟢</div>
            <div class="status-title">System Status</div>
            <div class="status-value">
                <div class="status-indicator">
                    <div class="indicator-dot"></div>
                    Online & Active
                </div>
            </div>
            <small style="color: #94a3b8;">Real-time diagnostics ready</small>
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
        st.markdown("""
        <div class="status-card">
            <div class="status-icon">⏰</div>
            <div class="status-title">Database Status</div>
            <div class="status-value">{}</div>
            <small style="color: #94a3b8;">Medical literature indexed</small>
        </div>
        """.format(datetime.now().strftime('%H:%M:%S')), unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Enhanced chat container
    # st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display welcome message if no messages
    if not st.session_state.messages:
        st.markdown("""
        <div class="assistant-message">
            <strong>🔬 PathoCare AI:</strong><br>
            Welcome to your advanced pathology diagnostic assistant! I'm here to help you with 
            evidence-based medical consultations, diagnostic insights, and pathological analysis.
            <em>💡 Example queries:</em><br>
            • "What if flu?"<br>
        </div>
        """, unsafe_allow_html=True)
    
    # Show previous messages
    for message in st.session_state.messages:
        display_chat_message(message['role'], message['content'])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced input section with dynamic sizing
    # st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### 💬 Enter Your Medical Query")
    
    # Use text_area with improved dynamic height
    user_query = st.text_area(
    label="Medical query input",
    placeholder="Type your pathology or medical question here... (e.g., 'What are the key diagnostic features of flu?')",
    key="medical_query",
    help="💡 Tip: Be specific with your medical queries for more accurate diagnostic insights",
    label_visibility="hidden"
)
    
    # Add submit button with medical styling
    submit_col1, submit_col2, submit_col3 = st.columns([1, 2, 1])
    
    with submit_col2:
        submit_button = st.button(
            "🔍 Analyze Query",
            use_container_width=True,
            type="primary"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process query when submitted
    if submit_button and user_query.strip():
        # Add user message to session state
        st.session_state.messages.append({'role': 'user', 'content': user_query})
        
        # Show loading animation without container
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
            # Load vector store
            db = load_vector_store()
            if db is None:
                error_msg = "❌ Medical database unavailable. Please ensure the pathology knowledge base is properly loaded."
                st.error(error_msg)
                st.session_state.messages.append({'role': 'assistant', 'content': error_msg})
                st.rerun()
            
            # Setup retrieval
            retriever = db.as_retriever(search_kwargs={"k": 4})
            hf_rep_id = "mistralai/Mistral-7B-Instruct-v0.3"
            llm = get_hf_endpoint(hf_rep_id)
            
            if llm is None:
                error_msg = "❌ AI diagnostic engine initialization failed. Please verify HuggingFace authentication."
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
            
            # Add assistant message to session state
            st.session_state.messages.append({'role': 'assistant', 'content': original_res})
            
            # Rerun to display the new messages
            st.rerun()
            
        except Exception as e:
            error_msg = f"❌ Diagnostic analysis error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({'role': 'assistant', 'content': error_msg})
            st.rerun()
    
    # Medical footer with proper rendering
    st.markdown("---")
    
    # Use simple HTML that Streamlit can handle
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









