import os
import json
import numpy as np
import streamlit as st
import google.generativeai as genai
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever
from utils.graph_ops import load_environment, get_driver, verify_connection, UserStateController
from sync_pipeline import GCP_EXAMS
from utils.bayesian_engine import (
    build_bayesian_model,
    VariableElimination,
    select_question_active_learning,
    select_domain_thompson_sampling,
    select_service_thompson_sampling,
    get_overall_entropy
)

# Load configuration settings
load_environment()

# Initialize session state variables
if "user_id" not in st.session_state:
    st.session_state.user_id = "default_student_1"
user_id = st.session_state.user_id

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

def retrieve_troubleshooting_context(driver, service_name, exam_id):
    """Retrieves troubleshooting and error resolution documentation chunks for a service."""
    query = """
    MATCH (c:Chunk)-[:DISCUSSES]->(s:Service {name: $service_name})
    WHERE c.source_exam = $exam_id AND (
        c.text CONTAINS "troubleshoot" OR 
        c.text CONTAINS "error" OR 
        c.text CONTAINS "permission" OR 
        c.text CONTAINS "fail" OR 
        c.text CONTAINS "issue" OR
        c.text CONTAINS "diagnose"
    )
    RETURN c.text AS content, c.source AS source, c.title AS title
    LIMIT 1
    """
    query_fallback = """
    MATCH (c:Chunk)-[:DISCUSSES]->(s:Service {name: $service_name})
    WHERE c.source_exam = $exam_id
    RETURN c.text AS content, c.source AS source, c.title AS title
    LIMIT 1
    """
    nodes = []
    try:
        with driver.session() as session:
            result = session.run(query, service_name=service_name, exam_id=exam_id)
            records = list(result)
            if not records:
                result = session.run(query_fallback, service_name=service_name, exam_id=exam_id)
                records = list(result)
            for record in records:
                nodes.append({
                    "content": record["content"],
                    "source": record["source"],
                    "title": record["title"] + " (Adaptive Remediation)",
                    "score": 1.0
                })
    except Exception as e:
        print(f"Failed to fetch remediation chunk for {service_name}: {e}")
    return nodes


# Initialize Neo4j Driver and User State Controller
db_connected = False
driver = None
state_controller = None

try:
    driver = get_driver()
    db_connected = verify_connection(driver)
    if db_connected:
        state_controller = UserStateController(driver)
        state_controller.touch_user(st.session_state.user_id)
except Exception as e:
    driver = None
    db_connected = False

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
            
    # ⚠️ Danger Zone (Reset Priors)
    if db_connected and driver and state_controller:
        st.markdown("---")
        with st.sidebar.expander("⚠️ Danger Zone (Reset Priors)"):
            st.markdown("**Reset Specific Topic**")
            # Fetch active services dynamically
            try:
                query = """
                MATCH (e:Exam {id: $exam_id})-[:REQUIRES_SERVICE]->(s:Service)
                RETURN s.name AS name ORDER BY s.name
                """
                with driver.session() as session:
                    res = session.run(query, exam_id=target_exam_id)
                    all_active_services = [r["name"] for r in res]
            except Exception:
                all_active_services = []
                
            if not all_active_services:
                # Fallback to local config representation
                all_active_services = []
                local_config = GCP_EXAMS[target_exam_id]
                for domain, services in local_config.get("domains", {}).items():
                    all_active_services.extend(services)
                all_active_services = sorted(list(set(all_active_services)))
                
            selected_node = st.selectbox("Select Topic to Reset:", all_active_services, key="reset_topic_select")
            
            if st.button(f"Reset {selected_node}", key="reset_topic_btn"):
                state_controller.reset_single_node(st.session_state.user_id, selected_node)
                st.toast(f"Priors reset for {selected_node}!", icon="🔄")
                # Reset quiz state for the service
                for key in list(st.session_state.keys()):
                    if key.startswith("quiz_"):
                        del st.session_state[key]
                st.rerun()
                
            st.markdown("---")
            
            st.markdown("**Reset Certification Track**")
            if st.button(f"Reset Entire {selected_exam} Track", key="reset_track_btn"):
                state_controller.reset_entire_exam(st.session_state.user_id, target_exam_id)
                st.toast(f"Wiped all records for {selected_exam}!", icon="🧹")
                # Reset chat messages
                st.session_state.messages = [{
                    "role": "assistant",
                    "content": f"Exam priors for **{selected_exam}** have been cleared. What would you like to review?",
                    "sources": []
                }]
                # Reset quiz state
                for key in list(st.session_state.keys()):
                    if key.startswith("quiz_"):
                        del st.session_state[key]
                st.rerun()
                
            st.markdown("---")
            
            st.markdown("**Reset Global Profile**")
            if st.button("🚨 Reset All Google Exams", key="reset_global_btn"):
                state_controller.reset_global_profile(st.session_state.user_id)
                st.toast("Wiped global learning history!", icon="💥")
                st.session_state.messages = [{
                    "role": "assistant",
                    "content": "All learning parameters have been set back to default priors.",
                    "sources": []
                }]
                # Reset quiz state
                for key in list(st.session_state.keys()):
                    if key.startswith("quiz_"):
                        del st.session_state[key]
                st.rerun()
            
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

# Set up Workspace Tabs
tab_chat, tab_quiz = st.tabs(["💬 Grounded Study Chat", "🧠 Adaptive Practice Quiz"])

with tab_chat:
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

    # User query capture - we use a unique key for the chat tab chat input
    if user_query := st.chat_input("Ask a question related to this certification blueprint...", key="chat_input"):
        # Append user question
        st.session_state.messages.append({
            "role": "user",
            "content": user_query,
            "sources": []
        })
        st.rerun()

# ----------------- ADAPTIVE PRACTICE QUIZ TAB -----------------
with tab_quiz:
    st.markdown("### 🧠 Adaptive Practice Quiz")
    
    if not db_connected or not driver or not state_controller:
        st.warning("Please connect to the Neo4j database to start the Adaptive Practice Quiz.")
    elif not gemini_configured:
        st.warning("Please verify your Gemini API key configuration to generate questions.")
    else:
        # Load user stats and blueprint
        with st.spinner("Calibrating student model from Neo4j taxonomy..."):
            # 1. Fetch user stats
            user_stats = state_controller.get_user_mastery_stats(st.session_state.user_id, target_exam_id)
            
            # 2. Fetch blueprint domains and services
            blueprint_records = []
            try:
                query = """
                MATCH (e:Exam {id: $exam_id})-[:HAS_DOMAIN]->(d:Domain)
                OPTIONAL MATCH (d)-[:TESTS_KNOWLEDGE_OF]->(s:Service)
                RETURN d.name AS domain, collect(s.name) AS services
                ORDER BY d.name
                """
                with driver.session() as session:
                    res = session.run(query, exam_id=target_exam_id)
                    blueprint_records = [{"domain": r["domain"], "services": r["services"]} for r in res]
            except Exception:
                pass
                
            if not blueprint_records:
                # Fallback local config
                local_config = GCP_EXAMS[target_exam_id]
                blueprint_records = [{"domain": d, "services": s} for d, s in local_config.get("domains", {}).items()]
                
            # 3. Build Bayesian model
            try:
                bbn_model, latent_nodes, candidate_questions = build_bayesian_model(blueprint_records, user_stats)
            except Exception as e:
                st.error(f"Failed to build Bayesian network: {e}")
                bbn_model = None
                
            # 4. Count total answered questions across the exam
            total_answered = 0
            if user_stats and user_stats.get("services"):
                total_answered = sum(
                    s_stats.get("alpha", 1) + s_stats.get("beta", 1) - 2 
                    for s_stats in user_stats.get("services", {}).values()
                )
                
        # Calculate current network entropy and certainty
        if bbn_model and len(latent_nodes) > 0:
            try:
                inference = VariableElimination(bbn_model)
                current_entropy = get_overall_entropy(bbn_model, inference, latent_nodes)
                max_entropy = len(latent_nodes)
                normalized_entropy = current_entropy / max_entropy if max_entropy > 0 else 0.0
            except Exception as e:
                current_entropy = 0.0
                normalized_entropy = 1.0
                inference = None
        else:
            current_entropy = 0.0
            normalized_entropy = 1.0
            inference = None
            
        # Determine Phase
        is_diagnostic = (total_answered < 5) or (normalized_entropy > 0.75)
        phase_name = "Diagnostic Phase (Active Information Gain)" if is_diagnostic else "Tutoring Phase (Thompson Sampling)"
        phase_description = (
            "We are running active information queries to map your knowledge state in the fewest steps."
            if is_diagnostic else
            "We are targeting your diagnosed weaknesses (user deficits) while exploring other areas."
        )
        
        # Display Dashboard Metrics
        st.markdown("""
        <style>
            .metric-card {
                background: #1e1f20;
                border: 1px solid #3c4043;
                border-radius: 6px;
                padding: 14px 18px;
                text-align: center;
            }
            .metric-val {
                font-size: 1.8rem;
                font-weight: 600;
                color: #8ab4f8;
                margin-bottom: 4px;
            }
            .metric-lbl {
                font-size: 0.8rem;
                color: #9aa0a6;
            }
        </style>
        """, unsafe_allow_html=True)
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{1.0 - normalized_entropy:.1%}</div>
                <div class="metric-lbl">Model Certainty</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{total_answered}</div>
                <div class="metric-lbl">Total Quiz Questions Answered</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{"Diagnostic" if is_diagnostic else "Tutoring"}</div>
                <div class="metric-lbl">Remediation Engine Mode</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("")
        st.markdown(f"🎯 **Active Strategy:** `{phase_name}`")
        st.caption(phase_description)
        st.progress(float(max(0.0, min(1.0, 1.0 - normalized_entropy))))
        
        st.markdown("---")
        
        # Quiz state management
        if "quiz_active" not in st.session_state:
            st.session_state.quiz_active = False
            
        if not st.session_state.quiz_active:
            st.markdown("Ready for your next question? Click the button below to generate a tailored challenge.")
            if st.button("🚀 Generate Next Adaptive Question", key="quiz_generate_btn"):
                with st.spinner("Selecting target topic and generating grounded question..."):
                    # 1. Select the next Service
                    service_name = None
                    if is_diagnostic and inference and candidate_questions:
                        try:
                            service_name = select_question_active_learning(
                                bbn_model, inference, latent_nodes, candidate_questions
                            )
                        except Exception as e:
                            print(f"Error in active learning selection: {e}")
                            
                    if not service_name:
                        # Fallback/Tutoring Phase: Thompson Sampling
                        try:
                            # Domain selection
                            domain_stats = user_stats.get("domains", {})
                            # Build domain stats map
                            domain_stats_map = {}
                            for record in blueprint_records:
                                d = record["domain"]
                                domain_stats_map[d] = domain_stats.get(d, {"alpha": 1, "beta": 1})
                            selected_domain, _ = select_domain_thompson_sampling(domain_stats_map)
                            
                            # Service selection within chosen domain
                            domain_services = next(r["services"] for r in blueprint_records if r["domain"] == selected_domain)
                            service_name = select_service_thompson_sampling(user_stats.get("services", {}), domain_services)
                        except Exception as e:
                            print(f"Error in Thompson sampling selection: {e}")
                            # Final absolute fallback: select first available service
                            if blueprint_records and blueprint_records[0]["services"]:
                                service_name = blueprint_records[0]["services"][0]
                                
                    if not service_name:
                        st.error("Could not determine next service node. Please check your exam taxonomy.")
                    else:
                        st.session_state.quiz_service = service_name
                        
                        # Calculate Prior Mastery
                        s_node = f"{service_name}_Service"
                        if inference:
                            try:
                                prior_mastery = inference.query(variables=[s_node], show_progress=False).values[1]
                                st.session_state.quiz_prior_mastery = float(prior_mastery)
                            except Exception:
                                st.session_state.quiz_prior_mastery = 0.50
                        else:
                            st.session_state.quiz_prior_mastery = 0.50
                            
                        # 2. Try fetching a pre-seeded Question from Neo4j
                        seeded_question = None
                        try:
                            # Note: rand() is Neo4j Cypher standard for randomizing results
                            query = """
                            MATCH (q:Question)-[:TESTS]->(s:Service {name: $service_name})
                            WHERE q.source_exam = $exam_id
                            RETURN q.question AS question, q.options AS options, q.correct_answer AS correct_answer, q.explanation AS explanation
                            ORDER BY rand()
                            LIMIT 1
                            """
                            with driver.session() as session:
                                res = session.run(query, service_name=service_name, exam_id=target_exam_id)
                                record = res.single()
                                if record:
                                    options_val = record["options"]
                                    if isinstance(options_val, str):
                                        try:
                                            options_dict = json.loads(options_val)
                                        except Exception:
                                            options_dict = options_val
                                    else:
                                        options_dict = options_val
                                        
                                    seeded_question = {
                                        "question": record["question"],
                                        "options": options_dict,
                                        "correct_answer": record["correct_answer"],
                                        "explanation": record["explanation"]
                                    }
                        except Exception as e:
                            print(f"Failed to check for pre-seeded question: {e}")
                            
                        if seeded_question:
                            st.session_state.quiz_question_data = seeded_question
                            st.session_state.quiz_active = True
                            st.session_state.quiz_submitted = False
                            st.session_state.quiz_chunks = []
                            st.rerun()
                            
                        # 3. Fetch context chunks from Neo4j (Fallback for dynamic generation)
                        context_chunks = []
                        try:
                            query = """
                            MATCH (c:Chunk)-[:DISCUSSES]->(s:Service {name: $service_name})
                            WHERE c.source_exam = $exam_id
                            RETURN c.text AS text, c.title AS title, c.source AS source
                            LIMIT 3
                            """
                            with driver.session() as session:
                                res = session.run(query, service_name=service_name, exam_id=target_exam_id)
                                context_chunks = [{"text": r["text"], "title": r["title"], "source": r["source"]} for r in res]
                        except Exception as e:
                            print(f"Failed to query quiz context chunks: {e}")
                            
                        if not context_chunks:
                            # Fallback vector search
                            fallback_sources = retrieve_graphrag_context(driver, f"GCP {service_name} concepts", target_exam_id)
                            context_chunks = [{"text": s["content"], "title": s["title"], "source": s["source"]} for s in fallback_sources]
                            
                        # 3. Generate question using Gemini API in JSON mode
                        context_str = "\n\n".join([f"[{c['title']} - {c['source']}]\n{c['text']}" for c in context_chunks])
                        prompt = f"""
                        You are a professional Google Cloud exam content creator and tutor.
                        Based on the following authoritative documentation context, generate one high-quality multiple-choice question (MCQ) testing knowledge of the '{service_name}' service.
                        
                        The question must follow these strict requirements:
                        1. It must be highly relevant to the '{selected_exam}' certification.
                        2. It must have 4 distinct options (A, B, C, D) that are technically plausible but with only one clearly correct option.
                        3. You must output the result strictly in JSON format matching this schema:
                        {{
                          "question": "Clear scenario-based question text...",
                          "options": {{
                            "A": "Option A text...",
                            "B": "Option B text...",
                            "C": "Option C text...",
                            "D": "Option D text..."
                          }},
                          "correct_answer": "A",
                          "explanation": "Detailed explanation of why the correct option is right and others are wrong, referencing the GCP services involved."
                        }}
                        Do not wrap in markdown or backticks, return ONLY the raw JSON string.
                        
                        Context:
                        {context_str}
                        """
                        try:
                            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                            model = genai.GenerativeModel('gemini-1.5-flash')
                            response = model.generate_content(
                                prompt,
                                generation_config={"response_mime_type": "application/json"}
                            )
                            question_data = json.loads(response.text)
                            st.session_state.quiz_question_data = question_data
                            st.session_state.quiz_active = True
                            st.session_state.quiz_submitted = False
                            st.session_state.quiz_chunks = context_chunks
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to generate question: {e}")
        else:
            # Active Quiz View
            service_name = st.session_state.quiz_service
            question_data = st.session_state.quiz_question_data
            
            st.markdown(f"### 📋 Question Topic: **{service_name}**")
            st.markdown(f"*{question_data['question']}*")
            
            options_dict = question_data["options"]
            options_list = [f"{k}: {v}" for k, v in options_dict.items()]
            
            # Disable selection after submission
            if not st.session_state.quiz_submitted:
                selected_val = st.radio("Choose the correct option:", options_list, index=None, key="quiz_options_radio")
                
                if st.button("Submit Answer", key="quiz_submit_btn"):
                    if not selected_val:
                        st.warning("Please select an option before submitting.")
                    else:
                        key = selected_val[0]  # Get key like 'A', 'B', etc.
                        passed = (key == question_data["correct_answer"])
                        
                        # 1. Update Mastery in Neo4j (Service and parent domains)
                        state_controller.update_mastery(st.session_state.user_id, service_name, passed)
                        s_node = f"{service_name}_Service"
                        if bbn_model:
                            parents = sorted(list(bbn_model.get_parents(s_node)))
                            for parent in parents:
                                parent_domain_name = parent.replace("_Domain", "")
                                state_controller.update_mastery(st.session_state.user_id, parent_domain_name, passed)
                                
                        # 2. Run BBN inference for posterior calculation
                        if inference:
                            q_node = f"{service_name}_Question"
                            pass_state = 1 if passed else 0
                            try:
                                posterior_mastery = inference.query(variables=[s_node], evidence={q_node: pass_state}, show_progress=False).values[1]
                                st.session_state.quiz_posterior_mastery = float(posterior_mastery)
                            except Exception:
                                st.session_state.quiz_posterior_mastery = 0.90 if passed else 0.10
                        else:
                            st.session_state.quiz_posterior_mastery = 0.90 if passed else 0.10
                            
                        st.session_state.quiz_user_answer = key
                        st.session_state.quiz_passed = passed
                        st.session_state.quiz_submitted = True
                        st.rerun()
            else:
                # Answer is submitted - show results
                selected_val = st.session_state.quiz_user_answer
                passed = st.session_state.quiz_passed
                
                # Render static selection to indicate what was chosen
                st.info(f"Your selection: **{selected_val}: {options_dict[selected_val]}**")
                
                if passed:
                    st.success("🎉 **Correct!** Great job mastering this service!")
                else:
                    st.error(f"❌ **Incorrect.** The correct option is **{question_data['correct_answer']}: {options_dict[question_data['correct_answer']]}**.")
                    
                # Display BBN Mastery Shift
                prior = st.session_state.quiz_prior_mastery
                post = st.session_state.quiz_posterior_mastery
                diff = post - prior
                diff_str = f"+{diff:.1%}" if diff >= 0 else f"{diff:.1%}"
                color = "#81c995" if diff >= 0 else "#f28b82"
                
                st.markdown(f"""
                <div style="background-color: #202124; border: 1px solid #3c4043; border-radius: 6px; padding: 14px; margin-bottom: 20px;">
                    <div style="font-size: 0.85rem; color: #9aa0a6; margin-bottom: 6px;">BBN Service Mastery Belief Shift:</div>
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 1.1rem; font-weight: 500;">{prior:.1%}</span>
                        <span style="color: #9aa0a6;">➔</span>
                        <span style="font-size: 1.1rem; font-weight: 600; color: #8ab4f8;">{post:.1%}</span>
                        <span style="font-size: 0.95rem; font-weight: 500; color: {color};">({diff_str})</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Explanation
                st.markdown("#### 📘 Explanation:")
                st.markdown(question_data["explanation"])
                
                # Collapsible Grounding Context
                if st.session_state.get("quiz_chunks"):
                    with st.expander("🔍 Grounding documentation chunks used for this question"):
                        cols = st.columns(len(st.session_state.quiz_chunks))
                        for idx, chunk in enumerate(st.session_state.quiz_chunks):
                            with cols[idx]:
                                st.markdown(f"""
                                <div class="source-card">
                                    <div class="source-title">{chunk['title']}</div>
                                    <div class="source-meta"><b>Source:</b> {chunk['source']}</div>
                                    <div style="font-size: 0.85rem; color: #cbd5e1;">
                                        {chunk['text']}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                if st.button("⏭️ Next Question", key="quiz_next_btn"):
                    st.session_state.quiz_active = False
                    st.rerun()

# Process new request in separate flow after rerun state refresh (Chat Loop)
if st.session_state.messages[-1]["role"] == "user":
    last_query = st.session_state.messages[-1]["content"]
    
    with tab_chat:
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
                    
                    # 2. Check student mastery and apply remediation path if needed
                    extra_prompt = ""
                    if state_controller:
                        user_stats = state_controller.get_user_mastery_stats(st.session_state.user_id, target_exam_id)
                        low_mastery_services = []
                        for s_name, s_stats in user_stats.get("services", {}).items():
                            alpha = s_stats.get("alpha", 1)
                            beta = s_stats.get("beta", 1)
                            if beta / (alpha + beta) < 0.50:
                                low_mastery_services.append(s_name)
                                
                        triggered_low_services = [s for s in low_mastery_services if s.lower() in last_query.lower()]
                        if triggered_low_services:
                            # Fetch troubleshooting chunks for these services
                            for s_name in triggered_low_services:
                                remediation_chunks = retrieve_troubleshooting_context(driver, s_name, target_exam_id)
                                sources.extend(remediation_chunks)
                            extra_prompt = f"\n\nNOTE: The student is currently struggling with the following topic(s): {', '.join(triggered_low_services)}. Proactively address common pitfalls, explain concepts with extra pedagogical care, and focus on troubleshooting/resolution pathways."
                    
                    context_str = "\n\n".join([f"[{item['title']} - {item['source']}]\n{item['content']}" for item in sources])
                    
                    # 3. Query Gemini model using prompt context
                    try:
                        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        prompt = f"""
                        You are a professional Google Cloud Principal Architect and GCP exam tutor. 
                        Solve the user's exam scenario using the provided authoritative documentation context.
                        Provide clear, structured explanations of specific technical trade-offs (e.g., cost, performance, operational overhead).
                        Always ground your response strictly in the context; if the context does not answer the question, use your general GCP knowledge but note where details are outside the provided context.{extra_prompt}
                        
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
