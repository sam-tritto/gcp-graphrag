import os
import streamlit as st
import google.generativeai as genai
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever
from utils.graph_ops import load_environment, get_driver, verify_connection
from sync_pipeline import GCP_EXAMS

# Load configuration settings
load_environment()

# Streamlit Page Design System Setup
st.set_page_config(
    page_title="GCP Multi-Exam Reasoning Engine",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Fidelity UI Styling via CSS Injection
st.markdown("""
<style>
    /* Google product typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Google Cloud Console Dark Theme Canvas */
    .stApp {
        background-color: #131314 !important;
        color: #e3e3e3 !important;
    }
    
    /* Top Navigation Console Bar */
    .gcp-navbar {
        background-color: #1e1f20;
        border-bottom: 1px solid #3c4043;
        padding: 12px 24px;
        margin: -80px -20px 24px -20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .gcp-navbar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .gcp-logo {
        width: 24px;
        height: 24px;
    }
    .gcp-brand-text {
        font-weight: 500;
        font-size: 1.15rem;
        color: #e3e3e3;
        letter-spacing: 0.2px;
    }
    .gcp-project-selector {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid #3c4043;
        border-radius: 4px;
        padding: 4px 12px;
        font-size: 0.85rem;
        color: #8ab4f8;
        cursor: default;
    }
    
    /* Main Console Workspace Card */
    .console-header-card {
        background: linear-gradient(135deg, #1e1f20 0%, #17181a 100%);
        border: 1px solid #3c4043;
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    .console-title {
        color: #ffffff;
        font-weight: 600;
        font-size: 1.6rem;
        margin: 0 0 6px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .console-title-accent {
        color: #8ab4f8;
        font-weight: 400;
    }
    .console-subtitle {
        color: #9aa0a6;
        font-size: 0.95rem;
        margin: 0;
        font-weight: 300;
        line-height: 1.4;
    }
    
    /* Database status GCP style panel */
    .gcp-status-panel {
        background-color: #202124;
        border: 1px solid #3c4043;
        border-radius: 6px;
        padding: 12px;
        margin-top: 10px;
    }
    .status-badge {
        display: flex;
        align-items: center;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .status-online {
        color: #81c995;
    }
    .status-offline {
        color: #f28b82;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
        display: inline-block;
    }
    .status-online .pulse-dot {
        background-color: #81c995;
        box-shadow: 0 0 8px #81c995;
        animation: pulse 2s infinite;
    }
    .status-offline .pulse-dot {
        background-color: #f28b82;
    }
    
    /* Blueprint detail cards */
    .blueprint-card {
        background-color: #202124;
        border: 1px solid #3c4043;
        border-radius: 6px;
        padding: 10px 12px;
        margin-bottom: 8px;
    }
    .blueprint-domain {
        color: #e3e3e3;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 4px;
    }
    .blueprint-services {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        margin-top: 4px;
    }
    .service-tag {
        background-color: rgba(138, 180, 248, 0.1);
        border: 1px solid rgba(138, 180, 248, 0.2);
        color: #8ab4f8;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    /* Streamlit Sidebar Material Overrides */
    div[data-testid="stSidebar"] {
        background-color: #1e1f20 !important;
        border-right: 1px solid #3c4043 !important;
    }
    
    /* Conversational Gemini interface bubbles */
    .chat-bubble {
        padding: 16px 20px;
        margin-bottom: 20px;
        line-height: 1.6;
        border-radius: 12px;
        font-size: 0.95rem;
        max-width: 85%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .chat-user {
        background-color: #202124;
        border: 1px solid #3c4043;
        color: #e3e3e3;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    .chat-assistant {
        background-color: #1e1f20;
        border: 1px solid #3c4043;
        color: #e3e3e3;
        margin-right: auto;
        border-bottom-left-radius: 4px;
        border-left: 4px solid #8ab4f8;
        position: relative;
    }
    
    /* GCP Cloud Shell output code styling */
    pre, code {
        font-family: 'Roboto Mono', monospace !important;
        background-color: #000000 !important;
        color: #81c995 !important;
        border: 1px solid #3c4043 !important;
        border-radius: 4px !important;
    }
    
    /* GraphRAG source retrieval cards */
    .source-card {
        background: #131314;
        border: 1px solid #3c4043;
        border-radius: 6px;
        padding: 12px 14px;
        margin: 6px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .source-title {
        color: #8ab4f8;
        font-weight: 500;
        font-size: 0.9rem;
        margin-bottom: 4px;
    }
    .source-meta {
        color: #9aa0a6;
        font-size: 0.75rem;
        margin-bottom: 8px;
    }
    
    /* CSS overrides for Streamlit elements to look flat and premium */
    div[data-testid="stChatInput"] {
        border-radius: 24px !important;
        border: 1px solid #747775 !important;
        background-color: #1e1f20 !important;
    }
    .stSpinner {
        color: #8ab4f8 !important;
    }
    
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(129, 201, 149, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(129, 201, 149, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(129, 201, 149, 0); }
    }
</style>
""", unsafe_allow_html=True)

# Helper function to generate query embedding from Gemini API
def get_query_embedding(query_text):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")
    genai.configure(api_key=api_key)
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=query_text,
        task_type="retrieval_query"
    )
    return result.get("embedding")

# Core retrieval method using Neo4j-graphrag with fallback logic
def retrieve_graphrag_context(driver, query_text, exam_id):
    # Try using Neo4j GraphRAG's VectorRetriever with filters
    try:
        retriever = VectorRetriever(driver, index_name="gcp_exam_embeddings")
        # Apply filters to restrict retrieval search space to the selected exam
        retrieval_results = retriever.search(
            query_text=query_text, 
            top_k=3,
            filters={"source_exam": exam_id}
        )
        
        nodes = []
        for item in retrieval_results.items:
            content = getattr(item, "content", "") or getattr(item, "text", "")
            metadata = getattr(item, "metadata", {}) or {}
            source = metadata.get("source", "Unknown Source")
            title = metadata.get("title", "Untitled Chunk")
            
            # Extract attributes from raw node if wrapped differently
            if not content and hasattr(item, "node"):
                node = item.node
                content = node.get("text", "")
                source = node.get("source", "Unknown Source")
                title = node.get("title", "Untitled Chunk")
                
            if content:
                nodes.append({
                    "content": content,
                    "source": source,
                    "title": title,
                    "score": getattr(item, "score", 0.0)
                })
        if nodes:
            return nodes
    except Exception as e:
        # Fallback to direct Cypher vector index lookup if VectorRetriever fails
        print(f"VectorRetriever failed: {e}. Falling back to Cypher vector search...")
        
    try:
        query_embedding = get_query_embedding(query_text)
        cypher_query = """
        CALL db.index.vector.queryNodes('gcp_exam_embeddings', 50, $embedding)
        YIELD node, score
        WHERE node.source_exam = $exam_id
        RETURN node.text AS content, node.source AS source, node.title AS title, score
        LIMIT 3
        """
        nodes = []
        with driver.session() as session:
            result = session.run(cypher_query, embedding=query_embedding, exam_id=exam_id)
            for record in result:
                nodes.append({
                    "content": record["content"],
                    "source": record["source"],
                    "title": record["title"],
                    "score": record["score"]
                })
        return nodes
    except Exception as e:
        st.sidebar.error(f"Context Retrieval failed: {e}")
        return []

# Sidebar Navigation Panel
with st.sidebar:
    st.image("https://www.gstatic.com/images/branding/product/2x/google_cloud_64dp.png", width=64)
    st.markdown("### GCP Exam GraphRAG")
    st.markdown("Multi-Exam Grounded Study Assistant backed by Neo4j blueprints and Gemini 1.5 Flash.")
    
    st.markdown("---")
    st.markdown("#### 📚 Active Certification")
    selected_exam = st.selectbox(
        "Select Your Study Path:",
        [
            "Cloud Digital Leader",
            "Associate Cloud Engineer",
            "Associate Data Practitioner",
            "Professional Cloud Architect",
            "Professional Data Engineer",
            "Professional Machine Learning Engineer"
        ]
    )
    
    exam_mapping = {
        "Cloud Digital Leader": "cdl",
        "Associate Cloud Engineer": "ace",
        "Associate Data Practitioner": "adp",
        "Professional Cloud Architect": "pca",
        "Professional Data Engineer": "pde",
        "Professional Machine Learning Engineer": "pmle"
    }
    target_exam_id = exam_mapping[selected_exam]
    
    # Handle Study Session Switch & reset messages state
    if "current_exam" not in st.session_state:
        st.session_state.current_exam = target_exam_id
        
    if st.session_state.current_exam != target_exam_id:
        st.session_state.current_exam = target_exam_id
        st.session_state.messages = [{
            "role": "assistant",
            "content": f"Switched target study certification to **{selected_exam}**. Ask me any exam blueprint or scenario question!",
            "sources": []
        }]
    
    st.markdown("---")
    st.markdown("#### System Health Check")
    
    # Check Neo4j Connectivity
    db_connected = False
    try:
        driver = get_driver()
        db_connected = verify_connection(driver)
    except Exception as e:
        driver = None
        
    status_class = "status-online" if db_connected else "status-offline"
    status_label = "Neo4j Aura Online" if db_connected else "Neo4j Offline"
    
    # Check Gemini API Key
    gemini_configured = bool(os.getenv("GEMINI_API_KEY")) and os.getenv("GEMINI_API_KEY") != "your_gemini_api_key"
    gemini_class = "status-online" if gemini_configured else "status-offline"
    gemini_label = "Gemini API Key Verified" if gemini_configured else "Gemini API Key Missing"
    
    st.markdown(f"""
    <div class="gcp-status-panel">
        <div class="status-badge {status_class}" style="margin-bottom: 8px;">
            <span class="pulse-dot"></span>{status_label}
        </div>
        <div class="status-badge {gemini_class}">
            <span class="pulse-dot"></span>{gemini_label}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not db_connected:
        st.warning("Please configure database credentials in the .env folder.")

    # Display active blueprint details fetched dynamically from Neo4j (or local fallback)
    st.markdown("---")
    st.markdown("#### 🎯 Active Exam Blueprint")
    
    blueprint_loaded = False
    if db_connected and driver:
        try:
            query = """
            MATCH (e:Exam {id: $exam_id})-[:HAS_DOMAIN]->(d:Domain)
            OPTIONAL MATCH (d)-[:TESTS_KNOWLEDGE_OF]->(s:Service)
            RETURN d.name AS domain, collect(s.name) AS services
            ORDER BY d.name
            """
            with driver.session() as session:
                result = session.run(query, exam_id=target_exam_id)
                records = list(result)
                if records:
                    blueprint_loaded = True
                    for record in records:
                        domain = record["domain"]
                        services = record["services"]
                        tags_html = "".join([f'<span class="service-tag">{s}</span>' for s in services if s])
                        st.markdown(f"""
                        <div class="blueprint-card">
                            <div class="blueprint-domain">📘 {domain}</div>
                            <div class="blueprint-services">{tags_html}</div>
                        </div>
                        """, unsafe_allow_html=True)
        except Exception as e:
            pass
            
    # Fallback to local static registry representation if database holds no records yet
    if not blueprint_loaded:
        local_config = GCP_EXAMS[target_exam_id]
        for domain, services in local_config.get("domains", {}).items():
            tags_html = "".join([f'<span class="service-tag">{s}</span>' for s in services])
            st.markdown(f"""
            <div class="blueprint-card">
                <div class="blueprint-domain">📘 {domain}</div>
                <div class="blueprint-services">{tags_html}</div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    st.caption("Developed using Streamlit, Neo4j, and Gemini 1.5 Flash.")

# Header Component - Styled as Google Cloud Console Navbar and Title
st.markdown(f"""
<div class="gcp-navbar">
    <div class="gcp-navbar-brand">
        <svg class="gcp-logo" viewBox="0 0 192 192" width="24" height="24">
            <polygon fill="#ea4335" points="96,16 172,60 172,148"/>
            <polygon fill="#4285f4" points="96,16 20,60 20,148"/>
            <polygon fill="#34a853" points="96,176 172,132 172,60"/>
            <polygon fill="#fbbc05" points="96,176 20,132 20,60"/>
            <polygon fill="#ffffff" points="96,48 144,76 144,116 96,144 48,116 48,76" fill-opacity="0.15"/>
        </svg>
        <span class="gcp-brand-text">Google Cloud Console</span>
    </div>
    <div class="gcp-project-selector">
        <span>gcp-exam-reasoning-engine</span>
    </div>
</div>
<div class="console-header-card">
    <h2 class="console-title">
        GCP Study Reasoning Engine <span class="console-title-accent">| {selected_exam}</span>
    </h2>
    <div class="console-subtitle">Grounded context retrieval restricted to official GCP certification resources. Currently isolating knowledge vectors tagged for <b>{target_exam_id.upper()}</b>.</div>
</div>
""", unsafe_allow_html=True)

# Application Message State Logic
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": f"Hi! Let's study for the **{selected_exam}** certification. Ask me a scenario or study blueprint question!",
        "sources": []
    }]

# Display Conversation History
for msg in st.session_state.messages:
    role_class = "chat-user" if msg["role"] == "user" else "chat-assistant"
    with st.container():
        st.markdown(f"""
        <div class="chat-bubble {role_class}">
            <strong>{"👤 User" if msg["role"] == "user" else "🤖 GCP Tutor"}:</strong><br>
            {msg["content"]}
        </div>
        """, unsafe_allow_html=True)
        
        # Display retrieved sources for assistant messages if present
        if msg.get("sources"):
            with st.expander("🔍 Retrieved GraphRAG Grounding Subgraph"):
                cols = st.columns(len(msg["sources"]))
                for idx, source in enumerate(msg["sources"]):
                    with cols[idx]:
                        score_text = f" (Score: {source['score']:.3f})" if source.get('score') else ""
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-title">{source['title']}</div>
                            <div class="source-meta"><b>Source:</b> {source['source']}{score_text}</div>
                            <div style="font-size: 0.85rem; color: #cbd5e1;">
                                {source['content'][:250]}...
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

# User query capture
if user_query := st.chat_input("Ask a question related to this certification blueprint..."):
    # Append user question
    st.session_state.messages.append({
        "role": "user",
        "content": user_query,
        "sources": []
    })
    st.rerun()

# Process new request in separate flow after rerun state refresh
if st.session_state.messages[-1]["role"] == "user":
    last_query = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        with st.spinner("Querying vector space and traversing Neo4j document subgraphs..."):
            
            # Verify setup requirements before executing APIs
            if not driver or not db_connected:
                response_text = "ERROR: Neo4j database is disconnected. Please configure credentials in your environment."
                sources = []
            elif not gemini_configured:
                response_text = "ERROR: Gemini API Key is missing. Please add a valid key to your configuration file."
                sources = []
            else:
                # 1. Retrieve context nodes from Neo4j restricted to current exam
                sources = retrieve_graphrag_context(driver, last_query, target_exam_id)
                context_str = "\n\n".join([f"[{item['title']} - {item['source']}]\n{item['content']}" for item in sources])
                
                # 2. Query Gemini model using prompt context
                try:
                    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    prompt = f"""
                    You are a professional Google Cloud Principal Architect and GCP exam tutor. 
                    Solve the user's exam scenario using the provided authoritative documentation context.
                    Provide clear, structured explanations of specific technical trade-offs (e.g., cost, performance, operational overhead).
                    Always ground your response strictly in the context; if the context does not answer the question, use your general GCP knowledge but note where details are outside the provided context.
                    
                    Context:
                    {context_str}
                    
                    Scenario Question:
                    {last_query}
                    """
                    
                    response = model.generate_content(prompt)
                    response_text = response.text
                except Exception as e:
                    response_text = f"API Error: Failed to generate response from Gemini model. Details: {e}"
            
            # Save Assistant Response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "sources": sources
            })
            st.rerun()
