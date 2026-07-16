import os
import json
import time
import html
import asyncio
import numpy as np
from nicegui import ui, app, run
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
from utils.graph_pyvis import generate_bbn_pyvis_graph
from utils.markdown_parser import render_markdown_to_html

# Load configuration settings
load_environment()

# Initialize static files directory for Pyvis graph
app.add_static_files('/static/lib', 'lib')
app.add_static_files('/static', 'data')

# Initialize Neo4j Driver and User State Controller
db_connected = False
driver = None
state_controller = None
db_connection_error = None

try:
    driver = get_driver()
    db_connected = verify_connection(driver)
    if db_connected:
        state_controller = UserStateController(driver)
except Exception as e:
    driver = None
    db_connected = False
    db_connection_error = str(e)

# Helper functions to support database calls with offline safety
def selected_exam_label(exam_id):
    exam_mapping_rev = {
        "cdl": "Cloud Digital Leader",
        "ace": "Associate Cloud Engineer",
        "adp": "Associate Data Practitioner",
        "pca": "Professional Cloud Architect",
        "pde": "Professional Data Engineer",
        "pmle": "Professional Machine Learning Engineer"
    }
    return exam_mapping_rev.get(exam_id, "Cloud Digital Leader")

def fetch_blueprint_records(exam_id):
    blueprint_records = []
    if db_connected and driver:
        try:
            query = """
            MATCH (e:Exam {id: $exam_id})-[:HAS_DOMAIN]->(d:Domain)
            OPTIONAL MATCH (d)-[:TESTS_KNOWLEDGE_OF]->(s:Service)
            OPTIONAL MATCH (s)-[:HAS_SUBCONCEPT]->(sub:SubConcept)
            RETURN d.name AS domain, s.name AS service, collect(sub.name) AS subconcepts
            ORDER BY d.name, s.name
            """
            with driver.session() as session:
                res = session.run(query, exam_id=exam_id)
                for r in res:
                    if r["service"]:
                        blueprint_records.append({
                            "domain": r["domain"],
                            "service": r["service"],
                            "subconcepts": r["subconcepts"]
                        })
        except Exception as ex:
            print(f"Error fetching database blueprint: {ex}")
            
    if not blueprint_records:
        local_config = GCP_EXAMS[exam_id]
        for domain, services_dict in local_config.get("domains", {}).items():
            if isinstance(services_dict, list):
                for s in services_dict:
                    blueprint_records.append({
                        "domain": domain,
                        "service": s,
                        "subconcepts": [s]
                    })
            else:
                for s, subconcepts in services_dict.items():
                    blueprint_records.append({
                        "domain": domain,
                        "service": s,
                        "subconcepts": subconcepts
                    })
    return blueprint_records

def fetch_full_graph_records(exam_id):
    """
    Fetches the full blueprint graph from Neo4j including:
    - Domains, Services, Subconcepts
    - Use cases (BEST_FOR)
    - CLI commands (CONFIGURED_BY, CREATED_VIA, etc.)
    - Functional roles (USED_FOR, ACTS_AS, etc.)
    - Hierarchy nodes (PARENT_OF)
    - Preferable over relations (PREFERABLE_OVER)
    - AntiPatterns (COMMON_PITFALL, RESOLVED_BY)
    """
    records = []
    if db_connected and driver:
        try:
            query = """
            MATCH (e:Exam {id: $exam_id})-[:HAS_DOMAIN]->(d:Domain)
            OPTIONAL MATCH (d)-[:TESTS_KNOWLEDGE_OF]->(s:Service)
            
            // Subconcepts
            OPTIONAL MATCH (s)-[:HAS_SUBCONCEPT]->(sub:SubConcept)
            
            // Outgoing leaf connections (UseCase, CLICommand, FunctionalRole, HierarchyNode, etc.)
            OPTIONAL MATCH (s)-[r_out]->(leaf_out)
            WHERE NOT leaf_out:Domain AND NOT leaf_out:Exam AND NOT leaf_out:Chunk AND NOT leaf_out:AntiPattern AND NOT leaf_out:SubConcept
            
            // Incoming leaf connections
            OPTIONAL MATCH (leaf_in)-[r_in]->(s)
            WHERE NOT leaf_in:Domain AND NOT leaf_in:Exam AND NOT leaf_in:Chunk AND NOT leaf_in:AntiPattern AND NOT leaf_in:SubConcept
            
            // Preferable over relations
            OPTIONAL MATCH (s)-[r_pref:PREFERABLE_OVER]->(s_pref:Service)
            
            // AntiPattern relations
            OPTIONAL MATCH (s)-[r_pit:COMMON_PITFALL]->(ap:AntiPattern)-[r_res:RESOLVED_BY]->(s_res:Service)
            
            RETURN d.name AS domain, s.name AS service, sub.name AS subconcept,
                   type(r_out) AS rel_out, labels(leaf_out)[0] AS label_out, coalesce(leaf_out.description, leaf_out.syntax, leaf_out.name) AS val_out,
                   type(r_in) AS rel_in, labels(leaf_in)[0] AS label_in, coalesce(leaf_in.description, leaf_in.syntax, leaf_in.name) AS val_in,
                   type(r_pref) AS rel_pref, s_pref.name AS pref_service,
                   ap.description AS ap_desc, s_res.name AS ap_res
            """
            with driver.session() as session:
                res = session.run(query, exam_id=exam_id)
                for r in res:
                    records.append(dict(r))
        except Exception as ex:
            print(f"Error fetching full database graph: {ex}")
            
    return records

def fetch_active_services(exam_id):
    all_active_services = []
    if db_connected and driver:
        try:
            query = """
            MATCH (e:Exam {id: $exam_id})-[:REQUIRES_SERVICE]->(s:Service)
            RETURN s.name AS name ORDER BY s.name
            """
            with driver.session() as session:
                res = session.run(query, exam_id=exam_id)
                all_active_services = [r["name"] for r in res]
        except Exception:
            pass
            
    if not all_active_services:
        local_config = GCP_EXAMS[exam_id]
        for domain, services in local_config.get("domains", {}).items():
            if isinstance(services, list):
                all_active_services.extend(services)
            else:
                all_active_services.extend(services.keys())
        all_active_services = sorted(list(set(all_active_services)))
    return all_active_services

def get_user_mastery_stats_safe(user_id, exam_id):
    if state_controller:
        try:
            return state_controller.get_user_mastery_stats(user_id, exam_id)
        except Exception as ex:
            print(f"Error getting mastery stats: {ex}")
    return {"domains": {}, "services": {}, "subconcepts": {}}

def update_mastery_safe(user_id, node_name, passed):
    if state_controller:
        try:
            state_controller.update_mastery(user_id, node_name, passed)
        except Exception as ex:
            print(f"Error updating mastery: {ex}")

# Embedding & Context Retrieval
def get_query_embedding(query_text):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")
    genai.configure(api_key=api_key)
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=query_text,
        task_type="retrieval_query",
        output_dimensionality=768
    )
    return result.get("embedding")

def retrieve_graphrag_context(driver, query_text, exam_id):
    if not db_connected or not driver:
        return []
    try:
        query_embedding = get_query_embedding(query_text)
    except Exception as e:
        print(f"Embedding generation failed: {e}")
        return []

    try:
        retriever = VectorRetriever(driver, index_name="gcp_exam_embeddings")
        retrieval_results = retriever.search(
            query_vector=query_embedding, 
            top_k=3,
            filters={"source_exam": exam_id}
        )
        
        nodes = []
        for item in retrieval_results.items:
            content = getattr(item, "content", "") or getattr(item, "text", "")
            metadata = getattr(item, "metadata", {}) or {}
            source = metadata.get("source", "Unknown Source")
            title = metadata.get("title", "Untitled Chunk")
            uid = metadata.get("uid", None)
            
            if not content and hasattr(item, "node"):
                node = item.node
                content = node.get("text", "")
                source = node.get("source", "Unknown Source")
                title = node.get("title", "Untitled Chunk")
                uid = node.get("uid", None)
                
            if content:
                nodes.append({
                    "uid": uid,
                    "content": content,
                    "source": source,
                    "title": title,
                    "score": getattr(item, "score", 0.0)
                })
        if nodes:
            return nodes
    except Exception as e:
        print(f"VectorRetriever failed: {e}. Falling back to Cypher vector search...")
        
    try:
        cypher_query = """
        CALL db.index.vector.queryNodes('gcp_exam_embeddings', 50, $embedding)
        YIELD node, score
        WHERE node.source_exam = $exam_id
        RETURN node.uid AS uid, node.text AS content, node.source AS source, node.title AS title, score
        LIMIT 3
        """
        nodes = []
        with driver.session() as session:
            result = session.run(cypher_query, embedding=query_embedding, exam_id=exam_id)
            for record in result:
                nodes.append({
                    "uid": record["uid"],
                    "content": record["content"],
                    "source": record["source"],
                    "title": record["title"],
                    "score": record["score"]
                })
        return nodes
    except Exception as e:
        print(f"Context Retrieval failed: {e}")
        return []

def retrieve_third_tier_context(driver, exam_id, chunk_uids):
    if not db_connected or not driver or not chunk_uids:
        return ""
        
    query = """
    MATCH (c:Chunk)-[:DISCUSSES]->(s:Service)
    WHERE c.uid IN $chunk_uids
    
    OPTIONAL MATCH (s)-[r_out]->(leaf_out)
    WHERE NOT leaf_out:Domain AND NOT leaf_out:Exam AND NOT leaf_out:Chunk AND NOT leaf_out:AntiPattern
    
    OPTIONAL MATCH (leaf_in)-[r_in]->(s)
    WHERE NOT leaf_in:Domain AND NOT leaf_in:Exam AND NOT leaf_in:Chunk AND NOT leaf_in:AntiPattern
    
    OPTIONAL MATCH (s)-[:COMMON_PITFALL]->(ap:AntiPattern)-[:RESOLVED_BY]->(res:Service)
    
    RETURN s.name AS service, 
           type(r_out) AS rel_out, labels(leaf_out)[0] AS label_out, coalesce(leaf_out.description, leaf_out.syntax, leaf_out.name) AS val_out,
           r_out.condition AS r_out_cond, r_out.reason AS r_out_reason,
           type(r_in) AS rel_in, labels(leaf_in)[0] AS label_in, coalesce(leaf_in.description, leaf_in.syntax, leaf_in.name) AS val_in,
           ap.description AS ap_desc, res.name AS ap_res
    """
    
    lines = set()
    try:
        with driver.session() as session:
            result = session.run(query, chunk_uids=chunk_uids)
            for record in result:
                service = record["service"]
                
                if record["rel_out"] and record["label_out"] and record["val_out"]:
                    if record["rel_out"] == "PREFERABLE_OVER":
                        cond = record["r_out_cond"]
                        reason = record["r_out_reason"]
                        lines.add(f"GCP Architectural Decision Boundary: Prefer ({service}) over ({record['val_out']}) when: '{cond}' (Reason: {reason})")
                    else:
                        lines.add(f"GCP Architectural Knowledge: ({service}) -[:{record['rel_out']}]-> ({record['label_out']}: \"{record['val_out']}\")")
                
                if record["rel_in"] and record["label_in"] and record["val_in"]:
                    lines.add(f"GCP Architectural Knowledge: ({record['label_in']}: \"{record['val_in']}\") -[:{record['rel_in']}]-> ({service})")
                    
                if record["ap_desc"] and record["ap_res"]:
                    lines.add(f"GCP Architectural Anti-Pattern: Using ({service}) for '{record['ap_desc']}' is an anti-pattern. RESOLUTION: Use ({record['ap_res']}) instead.")
    except Exception as e:
        print(f"Error fetching third-tier context: {e}")
        
    return "\n".join(sorted(list(lines)))

def retrieve_troubleshooting_context(driver, service_name, exam_id):
    if not db_connected or not driver:
        return []
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

# Pyvis Graph Generation
def update_mastery_graph(user_id, exam_id, filter_domain, show_subs, orientation):
    user_stats = get_user_mastery_stats_safe(user_id, exam_id)
    blueprint_records = fetch_blueprint_records(exam_id)
    graph_records = fetch_full_graph_records(exam_id)
    
    # Fallback to blueprint records if graph_records is empty
    if not graph_records:
        graph_records = []
        for r in blueprint_records:
            for sub in r.get("subconcepts", []):
                graph_records.append({
                    "domain": r["domain"],
                    "service": r["service"],
                    "subconcept": sub,
                    "rel_out": None, "label_out": None, "val_out": None,
                    "rel_in": None, "label_in": None, "val_in": None,
                    "rel_pref": None, "pref_service": None,
                    "ap_desc": None, "ap_res": None
                })
            if not r.get("subconcepts"):
                graph_records.append({
                    "domain": r["domain"],
                    "service": r["service"],
                    "subconcept": None,
                    "rel_out": None, "label_out": None, "val_out": None,
                    "rel_in": None, "label_in": None, "val_in": None,
                    "rel_pref": None, "pref_service": None,
                    "ap_desc": None, "ap_res": None
                })
                
    bbn_model = None
    inference = None
    
    try:
        bbn_model, latent_nodes, candidate_questions = build_bayesian_model(blueprint_records, user_stats)
        if bbn_model:
            inference = VariableElimination(bbn_model)
    except Exception as ex:
        print(f"Error building BBN in graph generation: {ex}")
        
    # Unconditionally force horizontal Left-to-Right layout so that levels stack as vertical columns
    # and expand horizontally from left to right.
    rankdir_val = "LR"
    
    generate_bbn_pyvis_graph(
        graph_records=graph_records,
        user_stats=user_stats,
        inference=inference,
        bbn_model=bbn_model,
        show_subconcepts=True,
        domain_filter=filter_domain,
        rankdir=rankdir_val,
        output_filename="data/mastery_graph.html"
    )

# Session storage setup
def init_user_storage():
    if "user_id" not in app.storage.user:
        app.storage.user["user_id"] = "default_student_1"
    if "current_exam" not in app.storage.user:
        app.storage.user["current_exam"] = "cdl"
    if "messages" not in app.storage.user:
        app.storage.user["messages"] = [{
            "role": "assistant",
            "content": "Hi! Let's study for the **Cloud Digital Leader** certification. Ask me a scenario or study blueprint question!",
            "sources": []
        }]
    if "quiz_active" not in app.storage.user:
        app.storage.user["quiz_active"] = False
    if "quiz_submitted" not in app.storage.user:
        app.storage.user["quiz_submitted"] = False
    if "bbn_rankdir" not in app.storage.user:
        app.storage.user["bbn_rankdir"] = "Left-to-Right"
    if "bbn_domain_filter" not in app.storage.user:
        app.storage.user["bbn_domain_filter"] = "All Domains"
    if "bbn_show_subs" not in app.storage.user:
        app.storage.user["bbn_show_subs"] = True

def clear_quiz_state():
    app.storage.user['quiz_active'] = False
    app.storage.user['quiz_subconcept'] = ''
    app.storage.user['quiz_service'] = ''
    app.storage.user['quiz_prior_mastery'] = 0.5
    app.storage.user['quiz_posterior_mastery'] = 0.5
    app.storage.user['quiz_question_data'] = {}
    app.storage.user['quiz_submitted'] = False
    app.storage.user['quiz_user_answer'] = ''
    app.storage.user['quiz_passed'] = False
    app.storage.user['quiz_chunks'] = []
    app.storage.user['quiz_selected_option'] = None

# Global Refreshables defined at module-level
@ui.refreshable
def chat_messages_container():
    messages = app.storage.user.get('messages', [])
    with ui.column().classes('w-full gap-4 overflow-y-auto max-h-[500px] p-2 bg-slate-50 rounded-lg border border-slate-200'):
        for msg in messages:
            role = msg['role']
            content = msg['content']
            sources = msg.get('sources', [])
            
            html_content = render_markdown_to_html(content)
            
            if role == 'user':
                with ui.column().classes('items-end w-full'):
                    with ui.card().classes('bg-blue-50 border border-blue-100 max-w-[85%] rounded-lg p-3'):
                        ui.label('👤 User').classes('text-xs font-semibold text-blue-800 mb-1')
                        ui.html(html_content).classes('text-sm text-gray-800')
            else:
                with ui.column().classes('items-start w-full'):
                    with ui.card().classes('bg-white border-l-4 border-blue-600 shadow-sm max-w-[85%] rounded-lg p-3 w-full'):
                        ui.label('🤖 GCP Tutor').classes('text-xs font-semibold text-blue-600 mb-1')
                        ui.html(html_content).classes('text-sm text-gray-800')
                        
                        if sources:
                            with ui.expansion('🔍 Retrieved Grounding Subgraph', icon='search').classes('w-full border border-gray-100 rounded mt-2'):
                                with ui.grid().classes('grid grid-cols-1 md:grid-cols-3 gap-2 p-2'):
                                    for s in sources:
                                        score_txt = f" (Score: {s['score']:.3f})" if s.get('score') else ""
                                        with ui.card().classes('bg-slate-50 border border-slate-200 p-2 text-[11px]'):
                                            ui.label(s['title']).classes('font-semibold text-blue-700 truncate')
                                            ui.label(f"Source: {s['source']}{score_txt}").classes('text-gray-400 font-medium')
                                            ui.label(s['content'][:150] + '...').classes('text-gray-600 leading-snug mt-1')

@ui.refreshable
def options_container():
    q_data = app.storage.user.get('quiz_question_data', {})
    submitted = app.storage.user.get('quiz_submitted', False)
    selected_option = app.storage.user.get('quiz_selected_option', None)
    correct_answer = q_data.get('correct_answer', '')
    
    with ui.column().classes('w-full gap-2 mt-2'):
        for key in ['A', 'B', 'C', 'D']:
            opt_text = q_data.get('options', {}).get(key, '')
            if not opt_text:
                continue
            
            border_cls = 'border-[#dadce0]'
            bg_cls = 'bg-white'
            text_cls = 'text-gray-800'
            
            if submitted:
                if key == correct_answer:
                    border_cls = 'border-emerald-500 border-2'
                    bg_cls = 'bg-emerald-50'
                    text_cls = 'text-emerald-800 font-medium'
                elif key == selected_option:
                    border_cls = 'border-rose-500 border-2'
                    bg_cls = 'bg-rose-50'
                    text_cls = 'text-rose-800 font-medium'
            else:
                if selected_option == key:
                    border_cls = 'border-blue-600 border-2'
                    bg_cls = 'bg-blue-50'
                    text_cls = 'text-blue-800 font-medium'
            
            def set_selection(k=key):
                if not submitted:
                    app.storage.user['quiz_selected_option'] = k
                    options_container.refresh()
                    
            with ui.card().classes(f'w-full p-3 cursor-pointer rounded-lg border transition-all duration-200 {border_cls} {bg_cls} hover:shadow-sm') \
                    .on('click', set_selection):
                with ui.row().classes('items-center gap-3 w-full'):
                    indicator_cls = 'border-gray-300'
                    indicator_bg = 'bg-transparent'
                    if selected_option == key:
                        indicator_cls = 'border-blue-600'
                        indicator_bg = 'bg-blue-600'
                    if submitted and key == correct_answer:
                        indicator_cls = 'border-emerald-600'
                        indicator_bg = 'bg-emerald-600'
                    elif submitted and key == selected_option:
                        indicator_cls = 'border-rose-600'
                        indicator_bg = 'bg-rose-600'
                        
                    ui.html(f'<div class="w-4 h-4 rounded-full border {indicator_cls} {indicator_bg} flex items-center justify-center text-[10px] text-white"></div>')
                    ui.label(f"{key}: {opt_text}").classes(f'text-sm {text_cls} flex-1 leading-snug')

@ui.refreshable
def quiz_tab_container():
    user_id = app.storage.user.get('user_id', 'default_student_1')
    exam_id = app.storage.user.get('current_exam', 'cdl')
    
    user_stats = get_user_mastery_stats_safe(user_id, exam_id)
    blueprint_records = fetch_blueprint_records(exam_id)
    
    bbn_model = None
    latent_nodes = []
    candidate_questions = []
    try:
        bbn_model, latent_nodes, candidate_questions = build_bayesian_model(blueprint_records, user_stats)
    except Exception as ex:
        print(f"Error building BBN for quiz stats: {ex}")
        
    total_answered = 0
    if user_stats and user_stats.get("subconcepts"):
        total_answered = sum(
            s_stats.get("alpha", 1) + s_stats.get("beta", 1) - 2 
            for s_stats in user_stats.get("subconcepts", {}).values()
        )
        
    normalized_entropy = 1.0
    if bbn_model and len(latent_nodes) > 0:
        try:
            inference = VariableElimination(bbn_model)
            current_entropy = get_overall_entropy(bbn_model, inference, latent_nodes)
            max_entropy = len(latent_nodes)
            normalized_entropy = current_entropy / max_entropy if max_entropy > 0 else 0.0
        except Exception:
            normalized_entropy = 1.0
            
    is_diagnostic = (total_answered < 5) or (normalized_entropy > 0.75)
    phase_name = "Diagnostic Phase (Active Information Gain)" if is_diagnostic else "Tutoring Phase (Thompson Sampling)"
    phase_description = (
        "We are running active information queries to map your knowledge state in the fewest steps."
        if is_diagnostic else
        "We are targeting your diagnosed weaknesses (user deficits) while exploring other areas."
    )
    
    with ui.row().classes('w-full gap-4 justify-between mb-4 flex-wrap md:flex-nowrap'):
        with ui.card().classes('flex-1 min-w-[120px] border border-gray-200 bg-white p-4 items-center'):
            ui.label(f"{1.0 - normalized_entropy:.1%}").classes('text-2xl font-bold text-blue-600')
            ui.label('Model Certainty').classes('text-xs text-gray-500 font-medium')
        with ui.card().classes('flex-1 min-w-[120px] border border-gray-200 bg-white p-4 items-center'):
            ui.label(str(total_answered)).classes('text-2xl font-bold text-blue-600')
            ui.label('Total Answered').classes('text-xs text-gray-500 font-medium')
        with ui.card().classes('flex-1 min-w-[120px] border border-gray-200 bg-white p-4 items-center'):
            ui.label("Diagnostic" if is_diagnostic else "Tutoring").classes('text-2xl font-bold text-blue-600')
            ui.label('Remediation Mode').classes('text-xs text-gray-500 font-medium')
            
    with ui.column().classes('w-full mb-4 gap-1'):
        ui.label(f"🎯 Strategy: {phase_name}").classes('text-sm font-semibold text-gray-700')
        ui.label(phase_description).classes('text-xs text-gray-500')
        ui.linear_progress(value=float(max(0.0, min(1.0, 1.0 - normalized_entropy)))).classes('w-full h-2 rounded bg-gray-100')
        
    quiz_active = app.storage.user.get('quiz_active', False)
    if not quiz_active:
        with ui.column().classes('w-full items-center p-6 bg-slate-50 border border-slate-200 rounded-lg gap-4'):
            ui.label('Ready for your next adaptive study question?').classes('text-sm font-medium text-gray-700')
            
            def run_gen():
                asyncio.create_task(generate_quiz_question(is_diagnostic, user_stats, blueprint_records, latent_nodes, candidate_questions))
            ui.button('🚀 Generate Next Adaptive Question', on_click=run_gen) \
                .classes('bg-blue-600 text-white rounded-md px-4 py-2 font-medium')
    else:
        subconcept = app.storage.user.get('quiz_subconcept', '')
        service = app.storage.user.get('quiz_service', '')
        q_data = app.storage.user.get('quiz_question_data', {})
        submitted = app.storage.user.get('quiz_submitted', False)
        
        with ui.column().classes('w-full gap-4'):
            ui.label(f"📋 Question Topic: {subconcept} ({service})").classes('text-base font-bold text-gray-700')
            with ui.card().classes('w-full border border-gray-200 p-4 bg-slate-50'):
                ui.label(q_data.get('question', '')).classes('text-sm font-medium italic text-gray-800')
                
            options_container()
            
            if not submitted:
                ui.button('Submit Answer', on_click=submit_quiz_answer) \
                    .classes('bg-blue-600 text-white w-full rounded-md py-2 font-medium')
            else:
                user_ans = app.storage.user.get('quiz_user_answer', '')
                passed = app.storage.user.get('quiz_passed', False)
                correct_ans = q_data.get('correct_answer', '')
                
                with ui.card().classes('w-full border border-gray-200 p-4 bg-white gap-3 shadow-sm'):
                    ui.label(f"Your Selection: {user_ans}: {q_data['options'].get(user_ans, '')}").classes('text-sm font-semibold text-gray-700')
                    
                    if passed:
                        ui.label('🎉 Correct! Excellent mastery of this service.').classes('text-base font-bold text-emerald-600')
                    else:
                        ui.label(f"❌ Incorrect. The correct answer is {correct_ans}: {q_data['options'].get(correct_ans, '')}.").classes('text-base font-bold text-rose-600')
                        
                    prior = app.storage.user.get('quiz_prior_mastery', 0.5)
                    post = app.storage.user.get('quiz_posterior_mastery', 0.5)
                    diff = post - prior
                    diff_str = f"+{diff:.1%}" if diff >= 0 else f"{diff:.1%}"
                    color_cls = 'text-emerald-600' if diff >= 0 else 'text-rose-600'
                    
                    with ui.row().classes('w-full bg-slate-50 border border-slate-200 p-3 rounded items-center justify-between'):
                        ui.label('BBN Mastery Shift:').classes('text-xs text-gray-500 font-semibold')
                        with ui.row().classes('gap-2 items-center'):
                            ui.label(f"{prior:.1%}").classes('text-sm font-medium')
                            ui.label('➔').classes('text-xs text-gray-400')
                            ui.label(f"{post:.1%}").classes('text-sm font-bold text-blue-600')
                            ui.label(f"({diff_str})").classes(f'text-xs font-bold {color_cls}')
                            
                    ui.label('📘 Explanation:').classes('text-sm font-bold text-gray-700')
                    ui.html(render_markdown_to_html(q_data.get('explanation', ''))).classes('text-sm text-gray-600')
                    
                    chunks = app.storage.user.get('quiz_chunks', [])
                    if chunks:
                        with ui.expansion('🔍 Grounding documentation chunks', icon='article').classes('w-full border border-gray-100 rounded mt-1'):
                            with ui.grid().classes('grid grid-cols-1 md:grid-cols-3 gap-2 p-2'):
                                for chunk in chunks:
                                    with ui.card().classes('bg-slate-50 border border-slate-200 p-2 text-[11px]'):
                                        ui.label(chunk['title']).classes('font-semibold text-blue-700 truncate')
                                        ui.label(f"Source: {chunk['source']}").classes('text-gray-400 font-medium')
                                        ui.label(chunk['text']).classes('text-gray-600 leading-snug mt-1')
                                        
                def next_q():
                    clear_quiz_state()
                    quiz_tab_container.refresh()
                ui.button('⏭️ Next Question', on_click=next_q) \
                    .classes('bg-blue-600 text-white w-full rounded-md py-2 font-medium mt-2')

@ui.refreshable
def draw_graph_view():
    t = time.time()
    ui.element('iframe') \
        .props(f'src="/static/mastery_graph.html?t={t}"') \
        .classes('w-full h-[680px] border border-gray-200 rounded-lg')

def refresh_graph_view():
    draw_graph_view.refresh()

# Graph updates triggers
async def regenerate_and_reload_graph():
    user_id = app.storage.user.get('user_id', 'default_student_1')
    exam_id = app.storage.user.get('current_exam', 'cdl')
    domain_filter = app.storage.user.get('bbn_domain_filter', 'All Domains')
    show_subs = app.storage.user.get('bbn_show_subs', True)
    orientation = app.storage.user.get('bbn_rankdir', 'Left-to-Right')
    
    await run.io_bound(update_mastery_graph, user_id, exam_id, domain_filter, show_subs, orientation)
    refresh_graph_view()

# Core process generators
async def generate_quiz_question(is_diagnostic, user_stats, blueprint_records, latent_nodes, candidate_questions):
    user_id = app.storage.user.get('user_id', 'default_student_1')
    exam_id = app.storage.user.get('current_exam', 'cdl')
    
    app.storage.user['quiz_active'] = True
    app.storage.user['quiz_question_data'] = {"question": "⌛ Generating grounded adaptive question...", "options": {"A": "", "B": "", "C": "", "D": ""}}
    app.storage.user['quiz_submitted'] = False
    quiz_tab_container.refresh()
    
    subconcept_name = None
    service_name = None
    
    try:
        bbn_model = None
        inference = None
        try:
            bbn_model, latent_nodes, candidate_questions = build_bayesian_model(blueprint_records, user_stats)
            if bbn_model:
                inference = VariableElimination(bbn_model)
        except Exception:
            pass
            
        if is_diagnostic and inference and candidate_questions:
            try:
                subconcept_name = await run.io_bound(
                    select_question_active_learning, bbn_model, inference, latent_nodes, candidate_questions
                )
            except Exception as e:
                print(f"Error in active learning selection: {e}")
                
        if not subconcept_name:
            try:
                domain_stats = user_stats.get("domains", {})
                domain_stats_map = {}
                for record in blueprint_records:
                    d = record["domain"]
                    domain_stats_map[d] = domain_stats.get(d, {"alpha": 1, "beta": 1})
                selected_domain, _ = select_domain_thompson_sampling(domain_stats_map)
                
                services_in_domain = [r for r in blueprint_records if r["domain"] == selected_domain]
                service_names = [r["service"] for r in services_in_domain]
                service_name = select_service_thompson_sampling(user_stats.get("services", {}), service_names)
                
                subconcepts_in_service = []
                for r in services_in_domain:
                    if r.get("service") == service_name:
                        subconcepts_in_service.extend(r.get("subconcepts", []))
                if not subconcepts_in_service:
                    subconcepts_in_service = [service_name]
                    
                sub_stats = user_stats.get("subconcepts", {})
                sub_sampled_scores = {}
                for sub in subconcepts_in_service:
                    stats = sub_stats.get(sub, {"alpha": 1, "beta": 1})
                    sub_sampled_scores[sub] = np.random.beta(stats.get("alpha", 1), stats.get("beta", 1))
                subconcept_name = max(sub_sampled_scores, key=sub_sampled_scores.get)
            except Exception as e:
                print(f"Error in Thompson sampling: {e}")
                if blueprint_records:
                    service_name = blueprint_records[0]["service"]
                    subconcept_name = blueprint_records[0]["subconcepts"][0] if blueprint_records[0]["subconcepts"] else service_name
                    
        if subconcept_name and not service_name:
            for r in blueprint_records:
                if subconcept_name in r.get("subconcepts", []):
                    service_name = r["service"]
                    break
            if not service_name:
                service_name = subconcept_name
                
        if not subconcept_name:
            ui.notify("Could not determine next subconcept node.", type="negative")
            app.storage.user['quiz_active'] = False
            quiz_tab_container.refresh()
            return
            
        prior_mastery = 0.50
        sub_node = f"{subconcept_name}_SubConcept"
        if inference:
            try:
                prior_mastery = float(inference.query(variables=[sub_node], show_progress=False).values[1])
            except Exception:
                pass
                
        app.storage.user['quiz_subconcept'] = subconcept_name
        app.storage.user['quiz_service'] = service_name
        app.storage.user['quiz_prior_mastery'] = prior_mastery
        
        context_chunks = []
        try:
            query = """
            MATCH (c:Chunk)-[:DISCUSSES]->(s:Service {name: $service_name})
            WHERE c.source_exam = $exam_id
            RETURN c.text AS text, c.title AS title, c.source AS source
            LIMIT 3
            """
            with driver.session() as session:
                res = session.run(query, service_name=service_name, exam_id=exam_id)
                context_chunks = [{"text": r["text"], "title": r["title"], "source": r["source"]} for r in res]
        except Exception:
            pass
            
        if not context_chunks:
            fallback_sources = await run.io_bound(retrieve_graphrag_context, driver, f"GCP {subconcept_name} {service_name} concepts", exam_id)
            context_chunks = [{"text": s["content"], "title": s["title"], "source": s["source"]} for s in fallback_sources]
            
        third_tier_info = []
        try:
            query_3rd = """
            MATCH (s:Service {name: $service_name})-[r]->(leaf)
            WHERE NOT leaf:Domain AND NOT leaf:Exam AND NOT leaf:Chunk
            RETURN type(r) AS rel_type, labels(leaf)[0] AS leaf_label, 
                   coalesce(leaf.description, leaf.syntax, leaf.name, leaf.condition) AS leaf_value,
                   leaf.reason AS leaf_reason
            UNION
            MATCH (leaf)-[r]->(s:Service {name: $service_name})
            WHERE NOT leaf:Domain AND NOT leaf:Exam AND NOT leaf:Chunk
            RETURN type(r) AS rel_type, labels(leaf)[0] AS leaf_label, 
                   coalesce(leaf.description, leaf.syntax, leaf.name, leaf.condition) AS leaf_value,
                   leaf.reason AS leaf_reason
            """
            with driver.session() as session:
                res_3rd = session.run(query_3rd, service_name=service_name)
                for r in res_3rd:
                    val_str = r['leaf_value']
                    if r['leaf_reason']:
                        val_str += f" (Reason: {r['leaf_reason']})"
                    third_tier_info.append(f"({service_name}) -[:{r['rel_type']}]-> ({r['leaf_label']}: '{val_str}')")
        except Exception:
            pass
            
        context_str = "\n\n".join([f"[{c['title']} - {c['source']}]\n{c['text']}" for c in context_chunks])
        if third_tier_info:
            context_str += "\n\n=== GCP STRUCTURAL BLUEPRINT RELATIONSHIPS ===\n" + "\n".join(third_tier_info)
            
        prompt = f"""
        You are a professional Google Cloud exam content creator and tutor.
        Based on the following authoritative documentation context, generate one high-quality multiple-choice question (MCQ) testing knowledge of the sub-concept '{subconcept_name}' under the '{service_name}' service.
        
        The question must follow these strict requirements:
        1. It must be highly relevant to the '{selected_exam_label(exam_id)}' certification.
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
        
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = await run.io_bound(model.generate_content, prompt)
        
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text.replace("```json", "", 1).replace("```", "", 1).strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text.replace("```", "", 1).replace("```", "", 1).strip()
            
        question_data = json.loads(raw_text)
        
        app.storage.user['quiz_question_data'] = question_data
        app.storage.user['quiz_active'] = True
        app.storage.user['quiz_submitted'] = False
        app.storage.user['quiz_chunks'] = context_chunks
        app.storage.user['quiz_selected_option'] = None
        
    except Exception as ex:
        ui.notify(f"Failed to generate quiz question: {ex}", type="negative")
        app.storage.user['quiz_active'] = False
        
    quiz_tab_container.refresh()

def submit_quiz_answer():
    user_id = app.storage.user.get('user_id', 'default_student_1')
    exam_id = app.storage.user.get('current_exam', 'cdl')
    subconcept_name = app.storage.user.get('quiz_subconcept', '')
    question_data = app.storage.user.get('quiz_question_data', {})
    selected_option = app.storage.user.get('quiz_selected_option', None)
    
    if not selected_option:
        ui.notify('Please select an option before submitting.', type='warning')
        return
        
    correct_answer = question_data.get('correct_answer')
    passed = (selected_option == correct_answer)
    
    update_mastery_safe(user_id, subconcept_name, passed)
    
    posterior_mastery = 0.50
    try:
        user_stats = get_user_mastery_stats_safe(user_id, exam_id)
        blueprint_records = fetch_blueprint_records(exam_id)
        bbn_model, latent_nodes, candidate_questions = build_bayesian_model(blueprint_records, user_stats)
        
        if bbn_model:
            sub_node = f"{subconcept_name}_SubConcept"
            parent_services = sorted(list(bbn_model.get_parents(sub_node)))
            for ps in parent_services:
                p_service_name = ps.replace("_Service", "")
                update_mastery_safe(user_id, p_service_name, passed)
                
                parents = sorted(list(bbn_model.get_parents(ps)))
                for parent in parents:
                    parent_domain_name = parent.replace("_Domain", "")
                    update_mastery_safe(user_id, parent_domain_name, passed)
                    
            user_stats = get_user_mastery_stats_safe(user_id, exam_id)
            bbn_model, latent_nodes, candidate_questions = build_bayesian_model(blueprint_records, user_stats)
            
            if bbn_model:
                inference = VariableElimination(bbn_model)
                q_node = f"{subconcept_name}_Question"
                pass_state = 1 if passed else 0
                try:
                    posterior_mastery = float(inference.query(variables=[sub_node], evidence={q_node: pass_state}, show_progress=False).values[1])
                except Exception:
                    posterior_mastery = 0.90 if passed else 0.10
        else:
            posterior_mastery = 0.90 if passed else 0.10
    except Exception as ex:
        print(f"Error in Bayesian updates: {ex}")
        posterior_mastery = 0.90 if passed else 0.10
        
    app.storage.user['quiz_user_answer'] = selected_option
    app.storage.user['quiz_passed'] = passed
    app.storage.user['quiz_posterior_mastery'] = posterior_mastery
    app.storage.user['quiz_submitted'] = True
    
    quiz_tab_container.refresh()
    asyncio.create_task(regenerate_and_reload_graph())

async def process_chat_response(query):
    # Show loading message
    messages = app.storage.user.get('messages', [])
    messages.append({
        "role": "assistant",
        "content": "⌛ *Querying vector space and traversing Neo4j document subgraphs...*",
        "sources": []
    })
    app.storage.user['messages'] = messages
    chat_messages_container.refresh()
    
    try:
        sources = await run.io_bound(retrieve_graphrag_context, driver, query, app.storage.user['current_exam'])
        
        extra_prompt = ""
        if state_controller:
            user_stats = await run.io_bound(get_user_mastery_stats_safe, app.storage.user['user_id'], app.storage.user['current_exam'])
            low_mastery_services = []
            for s_name, s_stats in user_stats.get("services", {}).items():
                alpha = s_stats.get("alpha", 1)
                beta = s_stats.get("beta", 1)
                if beta / (alpha + beta) < 0.50:
                    low_mastery_services.append(s_name)
                    
            triggered_low_services = [s for s in low_mastery_services if s.lower() in query.lower()]
            if triggered_low_services:
                for s_name in triggered_low_services:
                    remediation_chunks = await run.io_bound(retrieve_troubleshooting_context, driver, s_name, app.storage.user['current_exam'])
                    sources.extend(remediation_chunks)
                extra_prompt = f"\n\nNOTE: The student is currently struggling with the following topic(s): {', '.join(triggered_low_services)}. Proactively address common pitfalls, explain concepts with extra pedagogical care, and focus on troubleshooting/resolution pathways."
                
        chunk_uids = [item["uid"] for item in sources if item.get("uid")]
        third_tier_context = await run.io_bound(retrieve_third_tier_context, driver, app.storage.user['current_exam'], chunk_uids)
        
        context_str = "\n\n".join([f"[{item['title']} - {item['source']}]\n{item['content']}" for item in sources])
        if third_tier_context:
            context_str += "\n\n=== AUTHORITATIVE GCP STRUCTURAL KNOWLEDGE (THIRD-TIER RELATIONSHIPS) ===\n" + third_tier_context
            
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        You are a professional Google Cloud Principal Architect and GCP exam tutor. 
        Solve the user's exam scenario using the provided authoritative documentation context.
        Provide clear, structured explanations of specific technical trade-offs (e.g., cost, performance, operational overhead).
        Always ground your response strictly in the context; if the context does not answer the question, use your general GCP knowledge but note where details are outside the provided context.{extra_prompt}
        
        Context:
        {context_str}
        
        Scenario Question:
        {query}
        """
        
        response = await run.io_bound(model.generate_content, prompt)
        response_text = response.text
    except Exception as ex:
        response_text = f"API Error: Failed to generate response. Details: {ex}"
        sources = []
        
    # Replace loading message with actual response
    messages = app.storage.user.get('messages', [])
    if messages and messages[-1]['role'] == 'assistant':
        messages[-1] = {
            "role": "assistant",
            "content": response_text,
            "sources": sources
        }
    else:
        messages.append({
            "role": "assistant",
            "content": response_text,
            "sources": sources
        })
    app.storage.user['messages'] = messages
    chat_messages_container.refresh()

# NiceGUI Page Main Handler
@ui.page('/')
def index_page():
    init_user_storage()

    
    user_id = app.storage.user.get('user_id', 'default_student_1')
    exam_id = app.storage.user.get('current_exam', 'cdl')
    domain_filter = app.storage.user.get('bbn_domain_filter', 'All Domains')
    show_subs = app.storage.user.get('bbn_show_subs', True)
    orientation = app.storage.user.get('bbn_rankdir', 'Left-to-Right')
    
    # Initialize Pyvis graph once on first connection
    update_mastery_graph(user_id, exam_id, domain_filter, show_subs, orientation)
    
    # Styles for light Google Cloud styling
    ui.add_head_html("""
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Outfit', sans-serif !important;
            background-color: #f8f9fa !important;
            color: #202124 !important;
        }
        .google-card {
            background-color: #ffffff !important;
            border: 1px solid #dadce0 !important;
            border-radius: 8px !important;
            box-shadow: 0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15) !important;
        }
        .status-badge {
            display: flex;
            align-items: center;
            font-size: 0.85rem;
            font-weight: 500;
        }
        .status-online {
            color: #1e8e3e;
        }
        .status-offline {
            color: #d93025;
        }
        .pulse-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
            display: inline-block;
        }
        .status-online .pulse-dot {
            background-color: #1e8e3e;
            box-shadow: 0 0 8px #1e8e3e;
            animation: pulse 2s infinite;
        }
        .status-offline .pulse-dot {
            background-color: #d93025;
        }
        
        pre, code {
            font-family: 'Roboto Mono', monospace !important;
            background-color: #f1f3f4 !important;
            color: #1a73e8 !important;
            border: 1px solid #dadce0 !important;
            border-radius: 4px !important;
            padding: 2px 6px !important;
        }
        pre code {
            background-color: transparent !important;
            border: none !important;
            padding: 0 !important;
            display: block;
            color: #202124 !important;
        }
        pre {
            background-color: #202124 !important;
            color: #f8f9fa !important;
            padding: 12px !important;
            overflow-x: auto;
        }
        pre code {
            color: #81c995 !important;
        }
        
        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(30, 142, 62, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(30, 142, 62, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(30, 142, 62, 0); }
        }
    </style>
    """)
    
    # Outer Layout grid (Responsive Dashboard)
    with ui.grid().classes('grid grid-cols-1 lg:grid-cols-4 gap-6 p-6 w-full max-w-[1680px] mx-auto bg-[#f8f9fa] min-h-screen'):
        
        # COLUMN 1: LEFT SIDEBAR
        with ui.column().classes('col-span-1 gap-6 w-full'):
            
            # Header Console branding card
            with ui.card().classes('w-full border border-[#dadce0] rounded-xl p-5 bg-white shadow-sm gap-4'):
                with ui.row().classes('items-center gap-3'):
                    ui.html("""
                    <svg viewBox="0 0 192 192" width="32" height="32" class="flex-shrink-0">
                        <polygon fill="#4285f4" points="96,16 172,60 172,148"/>
                        <polygon fill="#34a853" points="96,16 20,60 20,148"/>
                        <polygon fill="#fbbc05" points="96,176 172,132 172,60"/>
                        <polygon fill="#ea4335" points="96,176 20,132 20,60"/>
                        <polygon fill="#ffffff" points="96,48 144,76 144,116 96,144 48,116 48,76" fill-opacity="0.2"/>
                    </svg>
                    """)
                    with ui.column().classes('gap-0'):
                        ui.label('Google Cloud').classes('text-sm font-semibold tracking-tight text-gray-800 leading-none')
                        ui.label('Study Assistant').classes('text-[11px] text-gray-500 font-medium')
                        
                ui.label('Multi-Exam Reasoning Engine backed by Neo4j blueprints & Gemini.').classes('text-xs text-gray-500 leading-relaxed')
                
                ui.label('📚 Active Certification').classes('text-[11px] font-semibold text-gray-700 uppercase tracking-wider mt-2')
                
                exam_labels = [
                    "Cloud Digital Leader",
                    "Associate Cloud Engineer",
                    "Associate Data Practitioner",
                    "Professional Cloud Architect",
                    "Professional Data Engineer",
                    "Professional Machine Learning Engineer"
                ]
                exam_mapping = {
                    "Cloud Digital Leader": "cdl",
                    "Associate Cloud Engineer": "ace",
                    "Associate Data Practitioner": "adp",
                    "Professional Cloud Architect": "pca",
                    "Professional Data Engineer": "pde",
                    "Professional Machine Learning Engineer": "pmle"
                }
                exam_mapping_rev = {v: k for k, v in exam_mapping.items()}
                
                current_exam_label = exam_mapping_rev.get(exam_id, "Cloud Digital Leader")
                
                async def on_exam_select(e):
                    new_exam_id = exam_mapping[e.value]
                    if app.storage.user['current_exam'] != new_exam_id:
                        app.storage.user['current_exam'] = new_exam_id
                        app.storage.user['messages'] = [{
                            "role": "assistant",
                            "content": f"Switched target study certification to **{e.value}**. Ask me any exam blueprint or scenario question!",
                            "sources": []
                        }]
                        clear_quiz_state()
                        
                        sidebar_blueprint.refresh()
                        sidebar_danger_zone.refresh()
                        chat_messages_container.refresh()
                        quiz_tab_container.refresh()
                        
                        await run.io_bound(update_mastery_graph, user_id, new_exam_id, domain_filter, show_subs, orientation)
                        refresh_graph_view()
                        
                ui.select(options=exam_labels, value=current_exam_label, on_change=on_exam_select) \
                    .classes('w-full border border-gray-300 rounded px-2 text-sm text-gray-800')
            
            # System Status Card
            with ui.card().classes('w-full border border-[#dadce0] rounded-xl p-5 bg-white shadow-sm gap-2'):
                ui.label('System Health').classes('text-[11px] font-semibold text-gray-700 uppercase tracking-wider')
                
                db_status_class = "status-online" if db_connected else "status-offline"
                db_status_label = "Neo4j Aura Online" if db_connected else "Neo4j Offline"
                
                gemini_configured = bool(os.getenv("GEMINI_API_KEY")) and os.getenv("GEMINI_API_KEY") != "your_gemini_api_key"
                gemini_status_class = "status-online" if gemini_configured else "status-offline"
                gemini_status_label = "Gemini API Verified" if gemini_configured else "Gemini API Key Missing"
                
                with ui.column().classes('w-full gap-2 mt-1'):
                    with ui.row().classes(f'items-center status-badge {db_status_class}'):
                        ui.html('<span class="pulse-dot"></span>')
                        ui.label(db_status_label).classes('text-xs font-semibold')
                    with ui.row().classes(f'items-center status-badge {gemini_status_class}'):
                        ui.html('<span class="pulse-dot"></span>')
                        ui.label(gemini_status_label).classes('text-xs font-semibold')
                        
                if not db_connected:
                    ui.label('Neo4j disconnected. Please configure credentials in your environment.').classes('text-[11px] text-rose-500 leading-snug mt-2')
                    if db_connection_error:
                        ui.label(f"Details: {db_connection_error[:100]}...").classes('text-[10px] text-gray-400 font-mono')
            
            # Active Blueprint Card (Refreshable)
            @ui.refreshable
            def sidebar_blueprint():
                curr_exam = app.storage.user.get('current_exam', 'cdl')
                domain_filter = app.storage.user.get('bbn_domain_filter', 'All Domains')
                
                async def toggle_domain(d):
                    current = app.storage.user.get('bbn_domain_filter', 'All Domains')
                    if current == d:
                        app.storage.user['bbn_domain_filter'] = 'All Domains'
                    else:
                        app.storage.user['bbn_domain_filter'] = d
                    sidebar_blueprint.refresh()
                    # Trigger Pyvis regeneration and reload
                    await regenerate_and_reload_graph()
                    
                with ui.card().classes('w-full border border-[#dadce0] rounded-xl p-5 bg-white shadow-sm gap-3'):
                    ui.label('🎯 Active Exam Blueprint').classes('text-[11px] font-semibold text-gray-700 uppercase tracking-wider')
                    ui.label('Click a domain below to filter the mastery network:').classes('text-[10px] text-gray-400 font-medium leading-none')
                    
                    records = fetch_blueprint_records(curr_exam)
                    domain_services = {}
                    for r in records:
                        d = r["domain"]
                        s = r["service"]
                        if d not in domain_services:
                            domain_services[d] = set()
                        if s:
                            domain_services[d].add(s)
                            
                    with ui.column().classes('w-full gap-3 overflow-y-auto max-h-[300px] pr-1'):
                        for dom, svcs in domain_services.items():
                            with ui.column().classes('w-full gap-1 border-b border-gray-100 pb-2 last:border-0'):
                                is_active = (domain_filter == dom)
                                bg_class = 'bg-blue-50 border-blue-200' if is_active else 'bg-transparent border-transparent'
                                text_class = 'text-blue-800 font-bold' if is_active else 'text-gray-800 font-bold'
                                
                                with ui.row().classes(f'w-full cursor-pointer p-1.5 rounded border transition-all {bg_class} hover:bg-slate-50') \
                                        .on('click', lambda d=dom: asyncio.create_task(toggle_domain(d))):
                                    ui.label(f"📘 {dom}").classes(f'text-xs {text_class} leading-tight')
                                    
                                with ui.row().classes('w-full gap-1.5 flex-wrap mt-1'):
                                    for svc in sorted(list(svcs)):
                                        ui.label(svc).classes('bg-blue-50 text-blue-700 border border-blue-100 rounded px-2 py-0.5 text-[9px] font-semibold')
            sidebar_blueprint()
            
            # Danger Zone Card (Refreshable)
            @ui.refreshable
            def sidebar_danger_zone():
                curr_exam = app.storage.user.get('current_exam', 'cdl')
                services = fetch_active_services(curr_exam)
                
                with ui.expansion('⚠️ Danger Zone (Reset Priors)', icon='warning').classes('w-full border border-red-200 bg-white rounded-xl shadow-sm text-xs'):
                    with ui.column().classes('w-full p-4 gap-3'):
                        ui.label('Reset Service Mastery').classes('font-bold text-gray-700 text-xs')
                        
                        reset_topic_select = ui.select(options=services, label="Select Topic").classes('w-full text-xs')
                        
                        async def on_reset_topic():
                            val = reset_topic_select.value
                            if not val:
                                ui.notify('Select a service to reset.', type='warning')
                                return
                            await run.io_bound(state_controller.reset_single_node, user_id, val)
                            ui.notify(f'Priors reset for {val}!', type='info')
                            clear_quiz_state()
                            quiz_tab_container.refresh()
                            await run.io_bound(update_mastery_graph, user_id, curr_exam, domain_filter, show_subs, orientation)
                            refresh_graph_view()
                            
                        ui.button('Reset Topic Mastery', on_click=on_reset_topic).classes('bg-red-50 text-red-600 border border-red-200 rounded w-full text-xs py-1.5 font-medium')
                        
                        ui.label('Reset Entire Track').classes('font-bold text-gray-700 text-xs mt-2')
                        
                        async def on_reset_track():
                            await run.io_bound(state_controller.reset_entire_exam, user_id, curr_exam)
                            ui.notify(f'Wiped track parameters!', type='info')
                            exam_lbl = selected_exam_label(curr_exam)
                            app.storage.user['messages'] = [{
                                "role": "assistant",
                                "content": f"Exam priors for **{exam_lbl}** have been cleared. What would you like to review?",
                                "sources": []
                            }]
                            clear_quiz_state()
                            chat_messages_container.refresh()
                            quiz_tab_container.refresh()
                            await run.io_bound(update_mastery_graph, user_id, curr_exam, domain_filter, show_subs, orientation)
                            refresh_graph_view()
                            
                        ui.button('Reset Exam Track', on_click=on_reset_track).classes('bg-red-50 text-red-600 border border-red-200 rounded w-full text-xs py-1.5 font-medium')
                        
                        ui.label('Reset Global Profile').classes('font-bold text-gray-700 text-xs mt-2')
                        
                        async def on_reset_global():
                            await run.io_bound(state_controller.reset_global_profile, user_id)
                            ui.notify('Wiped all exams!', type='info')
                            app.storage.user['messages'] = [{
                                "role": "assistant",
                                "content": "All learning parameters have been set back to default priors.",
                                "sources": []
                            }]
                            clear_quiz_state()
                            chat_messages_container.refresh()
                            quiz_tab_container.refresh()
                            await run.io_bound(update_mastery_graph, user_id, curr_exam, domain_filter, show_subs, orientation)
                            refresh_graph_view()
                            
                        ui.button('Reset Global Exams Profile', on_click=on_reset_global).classes('bg-red-600 text-white rounded w-full text-xs py-1.5 font-medium')
            sidebar_danger_zone()
            
            ui.label('Developed using NiceGUI, Neo4j & Gemini.').classes('text-[10px] text-gray-400 text-center w-full mt-2')

        # COLUMN 2 & 3: MAIN WORKSPACE PANEL
        with ui.column().classes('col-span-1 lg:col-span-2 gap-6 w-full'):
            
            with ui.card().classes('w-full border border-[#dadce0] rounded-xl bg-white shadow-sm p-0 overflow-hidden'):
                with ui.row().classes('bg-[#f8f9fa] border-b border-[#dadce0] px-5 py-3 w-full justify-between items-center'):
                    with ui.row().classes('items-center gap-2'):
                        ui.icon('assignment', size='20px').classes('text-blue-600')
                        ui.label('GCP Study Reasoning Engine').classes('text-sm font-bold text-gray-800')
                    with ui.row().classes('items-center bg-[#e8f0fe] border border-[#d2e3fc] px-3 py-1 rounded-full text-xs font-semibold text-blue-800'):
                        ui.label(selected_exam_label(exam_id))
                        
                with ui.column().classes('p-5 w-full gap-4'):
                    with ui.tabs().classes('w-full border-b border-gray-200') as tabs:
                        chat_tab = ui.tab('💬 Grounded Study Chat')
                        quiz_tab = ui.tab('🧠 Adaptive Practice Quiz')
                        
                    with ui.tab_panels(tabs, value=chat_tab).classes('w-full bg-transparent p-0 mt-3'):
                        
                        # Grounded Study Chat panel
                        with ui.tab_panel(chat_tab).classes('p-0 gap-4 flex flex-col w-full'):
                            chat_messages_container()
                            
                            async def on_send():
                                query = chat_input.value.strip()
                                if not query:
                                    return
                                messages = app.storage.user.get('messages', [])
                                messages.append({
                                    "role": "user",
                                    "content": query,
                                    "sources": []
                                })
                                app.storage.user['messages'] = messages
                                chat_messages_container.refresh()
                                chat_input.value = ''
                                asyncio.create_task(process_chat_response(query))
                                
                            with ui.row().classes('w-full gap-3 mt-2 items-center'):
                                chat_input = ui.input(placeholder='Ask a question about this blueprint...', value='') \
                                    .classes('flex-1 rounded-full border border-gray-300 px-4 text-sm bg-white shadow-sm focus:border-blue-600') \
                                    .on('keydown.enter', on_send)
                                    
                                ui.button(icon='send', on_click=on_send) \
                                    .classes('bg-blue-600 text-white rounded-full p-2 h-10 w-10 flex items-center justify-center shadow-sm')
                                    
                        # Adaptive Practice Quiz panel
                        with ui.tab_panel(quiz_tab).classes('p-0 gap-4 flex flex-col w-full'):
                            quiz_tab_container()
                            
        # COLUMN 4: GRAPH VIEW
        with ui.column().classes('col-span-1 gap-6 w-full'):
            
            with ui.card().classes('w-full border border-[#dadce0] rounded-xl p-5 bg-white shadow-sm gap-4 h-full'):
                with ui.row().classes('items-center gap-2 border-b border-gray-100 pb-2 w-full'):
                    ui.icon('hub', size='20px').classes('text-blue-600')
                    ui.label('📊 Bayesian Mastery Network').classes('text-sm font-bold text-gray-800')
                    
                ui.label('Visually trace how domains link to services and subconcepts. Color scales represent live mastery:').classes('text-xs text-gray-500 leading-normal')
                
                with ui.row().classes('w-full justify-between items-center text-[10px] text-gray-600 bg-gray-50 border border-gray-200 p-2 rounded'):
                    with ui.row().classes('items-center gap-1'):
                        ui.html('<span style="color: #1e8e3e; font-size: 14px;">■</span>')
                        ui.label('Mastered')
                    with ui.row().classes('items-center gap-1'):
                        ui.html('<span style="color: #f9ab00; font-size: 14px;">■</span>')
                        ui.label('Reviewing')
                    with ui.row().classes('items-center gap-1'):
                        ui.html('<span style="color: #d93025; font-size: 14px;">■</span>')
                        ui.label('Struggling')
                        
                        
                draw_graph_view()

# Run server entry point
if __name__ in {"__main__", "nicegui"}:
    # Standard storage secret configuration for user session state storage cookies
    ui.run(
        host="0.0.0.0",
        port=8080,
        title="GCP Exam Reasoning Engine",
        storage_secret="gcp_exam_reasoning_engine_secret_safe_key_987",
        reload=False
    )
