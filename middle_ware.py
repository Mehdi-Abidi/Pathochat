import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Load HF token from environment
HF_TOKEN = os.getenv("HF_TOKEN")

# Hugging Face model repo
hf_rep_id = "mistralai/Mistral-7B-Instruct-v0.3"
# Function to get Hugging Face endpoint
def get_hf_endpoint(hf_rep_id):
    llm = HuggingFaceEndpoint(
        repo_id=hf_rep_id,
        temperature=0.5,
        provider="hf-inference",
        max_new_tokens= 1024,
        huggingfacehub_api_token=HF_TOKEN
    )
    return llm

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
# Function to create a prompt template
def get_prompt(template):
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

# Vector store path and embedding model
db_path = "vector_store/faiss_database"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS vector DB
db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

# Create the retriever and LLM
retriever = db.as_retriever(search_kwargs={"k": 5})
llm = get_hf_endpoint(hf_rep_id)
prompt = get_prompt(PROMPT_TEMPLATE)

# Final RetrievalQA setup
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

query = input("Enter your query: ")
response = retrieval_qa.invoke({"query": query})

print("Answer:", response["result"])
#print("Source Documents:", response["source_documents"])
# import os
# import streamlit as st
# from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from datetime import datetime

# # Configure page
# st.set_page_config(
#     page_title="Pathochat - Medical Assistant", 
#     page_icon="🩺", 
#     layout="wide"
# )

# # Vector DB path
# db_path = "vector_store/faiss_database"

# # Modern CSS styling inspired by the AQI app
# st.markdown("""
# <style>
#     .stApp {
#         background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
#         color: #E5E7EB;
#     }
    
#     h1, h2, h3 {
#         color: #F0F9FF;
#         text-shadow: 0 2px 4px rgba(0,0,0,0.3);
#     }
    
#     .main-header {
#         background: rgba(30, 41, 59, 0.8);
#         border-radius: 16px;
#         padding: 2rem;
#         margin-bottom: 2rem;
#         box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
#         backdrop-filter: blur(10px);
#         border: 1px solid rgba(255, 255, 255, 0.1);
#         text-align: center;
#     }
    
#     .chat-container {
#         background: rgba(30, 41, 59, 0.7);
#         border-radius: 16px;
#         padding: 1.5rem;
#         margin-bottom: 1rem;
#         box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
#         backdrop-filter: blur(10px);
#         border: 1px solid rgba(255, 255, 255, 0.1);
#         min-height: 400px;
#         max-height: 600px;
#         overflow-y: auto;
#     }
    
#     .user-message {
#         background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
#         color: white;
#         padding: 1rem 1.5rem;
#         border-radius: 18px 18px 6px 18px;
#         margin: 0.5rem 0 0.5rem 20%;
#         box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
#         word-wrap: break-word;
#     }
    
#     .assistant-message {
#         background: rgba(17, 24, 39, 0.8);
#         color: #E5E7EB;
#         padding: 1rem 1.5rem;
#         border-radius: 18px 18px 18px 6px;
#         margin: 0.5rem 20% 0.5rem 0;
#         border-left: 4px solid #10b981;
#         box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
#         word-wrap: break-word;
#     }
    
#     .info-card {
#         background: rgba(17, 24, 39, 0.7);
#         border-radius: 12px;
#         padding: 1.5rem;
#         margin-bottom: 1rem;
#         box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
#         border: 1px solid rgba(255, 255, 255, 0.05);
#     }
    
#     .feature-grid {
#         display: grid;
#         grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
#         gap: 1rem;
#         margin: 1rem 0;
#     }
    
#     .feature-card {
#         background: rgba(17, 24, 39, 0.6);
#         border-radius: 12px;
#         padding: 1.5rem;
#         text-align: center;
#         border: 1px solid rgba(255, 255, 255, 0.05);
#         transition: transform 0.3s ease, box-shadow 0.3s ease;
#     }
    
#     .feature-card:hover {
#         transform: translateY(-4px);
#         box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
#     }
    
#     .status-indicator {
#         display: inline-block;
#         width: 12px;
#         height: 12px;
#         border-radius: 50%;
#         margin-right: 8px;
#     }
    
#     .status-online {
#         background-color: #10b981;
#         box-shadow: 0 0 8px rgba(16, 185, 129, 0.5);
#     }
    
#     .status-loading {
#         background-color: #f59e0b;
#         box-shadow: 0 0 8px rgba(245, 158, 11, 0.5);
#         animation: pulse 2s infinite;
#     }
    
#     @keyframes pulse {
#         0%, 100% { opacity: 1; }
#         50% { opacity: 0.5; }
#     }
    
#     .metric-box {
#         background: rgba(17, 24, 39, 0.7);
#         border-radius: 8px;
#         padding: 1rem;
#         text-align: center;
#         margin-bottom: 0.5rem;
#     }
    
#     .metric-title {
#         font-size: 0.9rem;
#         color: #94A3B8;
#         margin-bottom: 0.5rem;
#     }
    
#     .metric-value {
#         font-size: 1.5rem;
#         font-weight: 600;
#         color: #F0F9FF;
#     }
    
#     .input-container {
#         background: rgba(30, 41, 59, 0.7);
#         border-radius: 16px;
#         padding: 1rem;
#         margin-top: 1rem;
#         box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
#         border: 1px solid rgba(255, 255, 255, 0.1);
#     }
    
#     .footer {
#         text-align: center;
#         color: #94A3B8;
#         margin-top: 2rem;
#         padding: 2rem;
#         background: rgba(17, 24, 39, 0.5);
#         border-radius: 12px;
#         backdrop-filter: blur(10px);
#     }
    
#     /* Hide Streamlit default elements */
#     .stDeployButton { display: none; }
#     #MainMenu { visibility: hidden; }
#     footer { visibility: hidden; }
#     header { visibility: hidden; }
    
#     /* Custom scrollbar */
#     ::-webkit-scrollbar {
#         width: 8px;
#     }
    
#     ::-webkit-scrollbar-track {
#         background: rgba(30, 41, 59, 0.3);
#         border-radius: 4px;
#     }
    
#     ::-webkit-scrollbar-thumb {
#         background: rgba(59, 130, 246, 0.6);
#         border-radius: 4px;
#     }
    
#     ::-webkit-scrollbar-thumb:hover {
#         background: rgba(59, 130, 246, 0.8);
#     }
# </style>
# """, unsafe_allow_html=True)

# # Cache the vector store loading
# @st.cache_resource
# def load_vector_store():
#     try:
#         embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
#         return db
#     except Exception as e:
#         st.error(f"Failed to load vector store: {str(e)}")
#         return None

# # Prompt template
# PROMPT_TEMPLATE = """
# You are a focused assistant. Your task is to generate accurate answers using only the context provided.
# - Do not include external knowledge.
# - Do not speculate. If the context lacks the answer, respond with: "I don't know."
# - No fluff, greetings, or commentary — jump straight to the answer.

# [Context]
# {context}

# [Question]
# {question}

# [Answer]
# """

# def get_prompt(template):
#     return PromptTemplate(
#         template=template,
#         input_variables=["context", "question"]
#     )

# # HF token
# HF_TOKEN = os.getenv("HF_TOKEN")

# # LLM Endpoint config
# @st.cache_resource
# def get_hf_endpoint(hf_rep_id):
#     try:
#         llm = HuggingFaceEndpoint(
#             repo_id=hf_rep_id,
#             temperature=0.5,
#             provider="hf-inference",
#             max_new_tokens=1024,
#             huggingfacehub_api_token=HF_TOKEN
#         )
#         return llm
#     except Exception as e:
#         st.error(f"Failed to initialize LLM: {str(e)}")
#         return None

# def display_chat_message(role, content):
#     """Display a chat message with modern styling"""
#     if role == "user":
#         st.markdown(f"""
#         <div class="user-message">
#             <strong>🩺 You:</strong><br>
#             {content}
#         </div>
#         """, unsafe_allow_html=True)
#     else:
#         # Format assistant response to separate main answer from sources
#         parts = content.split("\n\nSource Docs:\n")
#         main_answer = parts[0]
#         sources = parts[1] if len(parts) > 1 else ""
        
#         st.markdown(f"""
#         <div class="assistant-message">
#             <strong>🤖 Pathochat:</strong><br>
#             {main_answer}
#         </div>
#         """, unsafe_allow_html=True)
        
#         if sources:
#             with st.expander("📚 View Source Documents", expanded=False):
#                 st.text(sources)

# def main():
#     # App header
#     st.markdown("""
#     <div class="main-header">
#         <h1>🩺 Pathochat</h1>
#         <h3>Your AI Medical Assistant in Pathology</h3>
#         <p style="color: #94A3B8; margin-top: 1rem;">
#             Advanced medical consultation powered by AI • Specialized in Pathology
#         </p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # System status and info
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.markdown("""
#         <div class="info-card">
#             <div class="metric-box">
#                 <div class="metric-title">System Status</div>
#                 <div class="metric-value">
#                     <span class="status-indicator status-online"></span>
#                     Online
#                 </div>
#             </div>
#         </div>
#         """, unsafe_allow_html=True)
    
#     with col2:
#         st.markdown("""
#         <div class="info-card">
#             <div class="metric-box">
#                 <div class="metric-title">Model</div>
#                 <div class="metric-value">Mistral-7B</div>
#                 <small style="color: #94A3B8;">Instruct v0.3</small>
#             </div>
#         </div>
#         """, unsafe_allow_html=True)
    
#     with col3:
#         st.markdown("""
#         <div class="info-card">
#             <div class="metric-box">
#                 <div class="metric-title">Last Updated</div>
#                 <div class="metric-value">{}</div>
#                 <small style="color: #94A3B8;">Vector DB Status</small>
#             </div>
#         </div>
#         """.format(datetime.now().strftime('%H:%M')), unsafe_allow_html=True)
    
 
        
    
    
#     # Initialize session state
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []
    
#     # Chat container
#     # st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
#     # Display welcome message if no messages
#     if not st.session_state.messages:
#         st.markdown("""
#         <div class="assistant-message">
#             <strong>🤖 Pathochat:</strong><br>
#             Welcome! I'm your AI medical assistant specialized in pathology. 
#             Ask me any questions about medical conditions, diagnostic procedures, 
#             or pathological findings. I'll provide evidence-based answers using 
#             my medical knowledge base.
#             <br><br>
#             <em>Example questions:</em><br>
#             • "What are the characteristics of malignant cells?"<br>
#         </div>
#         """, unsafe_allow_html=True)
    
#     # Show previous messages
#     for message in st.session_state.messages:
#         display_chat_message(message['role'], message['content'])
    
#     st.markdown('</div>', unsafe_allow_html=True)
    
#     # Input section
#     st.markdown('<div class="input-container">', unsafe_allow_html=True)
#     user_query = st.chat_input("💬 Enter your medical query here...")
#     st.markdown('</div>', unsafe_allow_html=True)
    
#     if user_query:  # Only run when user input exists
#         # Add user message to session state
#         st.session_state.messages.append({'role': 'user', 'content': user_query})
        
#         with st.spinner("🔍 Analyzing your query and searching medical database..."):
#             try:
#                 # Load vector store
#                 db = load_vector_store()
#                 if db is None:
#                     error_msg = "❌ Vector store not found. Please ensure the medical database is loaded correctly."
#                     st.error(error_msg)
#                     st.session_state.messages.append({'role': 'assistant', 'content': error_msg})
#                     st.rerun()
                
#                 # Setup retrieval
#                 retriever = db.as_retriever(search_kwargs={"k": 4})
#                 hf_rep_id = "mistralai/Mistral-7B-Instruct-v0.3"
#                 llm = get_hf_endpoint(hf_rep_id)
                
#                 if llm is None:
#                     error_msg = "❌ Failed to initialize the language model. Please check your HuggingFace token."
#                     st.error(error_msg)
#                     st.session_state.messages.append({'role': 'assistant', 'content': error_msg})
#                     st.rerun()
                
#                 prompt = get_prompt(PROMPT_TEMPLATE)
                
#                 # Setup RetrievalQA chain
#                 retrieval_qa = RetrievalQA.from_chain_type(
#                     llm=llm,
#                     chain_type="stuff",
#                     retriever=retriever,
#                     return_source_documents=True,
#                     chain_type_kwargs={"prompt": prompt}
#                 )
                
#                 # Get response
#                 response = retrieval_qa.invoke({"query": user_query})
                
#                 result = response["result"]
#                 source_documents = response["source_documents"]
                
#                 # Format full response
#                 original_res = result + "\n\nSource Docs:\n" + str(source_documents)
                
#                 # Add assistant message to session state
#                 st.session_state.messages.append({'role': 'assistant', 'content': original_res})
                
#                 # Rerun to display the new messages
#                 st.rerun()
                
#             except Exception as e:
#                 error_msg = f"❌ An error occurred while processing your query: {str(e)}"
#                 st.error(error_msg)
#                 st.session_state.messages.append({'role': 'assistant', 'content': error_msg})
#                 st.rerun()
    
#     # Footer
#     st.markdown("""
#     <div class="footer">
#         <p><strong>🩺 Pathochat Medical Assistant</strong></p>
#         <p style="font-size: 0.8rem; margin-top: 1rem;">
#             ⚠️ <strong>Disclaimer:</strong> This AI assistant provides information for educational purposes only. 
#             Always consult with qualified healthcare professionals for medical advice and diagnosis.
#         </p>
#     </div>
#     """, unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()













