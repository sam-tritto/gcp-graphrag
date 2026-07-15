import os
import hashlib
import re
import requests
import time
import google.api_core.exceptions as gcp_errors
import google.generativeai as genai
from utils.graph_ops import (
    load_environment,
    get_driver,
    verify_connection,
    setup_schema,
    setup_exam_metadata,
    get_stored_hash,
    set_stored_hash,
    delete_source_chunks,
    insert_chunks_batch,
    link_chunks_to_services_batch,
    update_chunk_embeddings_batch,
    build_sequential_relationships,
    ensure_vector_index
)
from data.fetch_docs import parse_html_framework, parse_pdf_case_study

# Initialize environment configuration
load_environment()

# Define the curated registry of Google certifications, domains, services, and docs
GCP_EXAMS = {
    # Foundational Tier
    "cdl": {
        "name": "Cloud Digital Leader",
        "docs": [
            "https://cloud.google.com/compute/docs/concepts",
            "https://cloud.google.com/storage/docs/introduction",
            "https://cloud.google.com/sql/docs/introduction",
            "https://cloud.google.com/iam/docs/overview",
            "https://cloud.google.com/run/docs/overview/what-is-cloud-run",
            "https://cloud.google.com/kubernetes-engine/docs/concepts/what-is-gke"
        ],
        "domains": {
            "Digital Transformation with Google Cloud": [
                "Compute Engine", "Google Kubernetes Engine", "Cloud Run", "App Engine"
            ],
            "Innovating with Data and Google Cloud": [
                "BigQuery", "Cloud Storage", "Cloud Spanner", "Cloud SQL", "Looker"
            ],
            "Infrastructure and Application Modernization": [
                "Compute Engine", "Google Kubernetes Engine", "VPC", "IAM"
            ],
            "Google Cloud Security and Operations": [
                "IAM", "Operations Suite", "Cloud Armor"
            ]
        }
    },
    # Associate Tier
    "ace": {
        "name": "Associate Cloud Engineer",
        "docs": [
            "https://cloud.google.com/compute/docs/concepts",
            "https://cloud.google.com/storage/docs/introduction",
            "https://cloud.google.com/kubernetes-engine/docs/concepts/what-is-gke",
            "https://cloud.google.com/run/docs/overview/what-is-cloud-run",
            "https://cloud.google.com/sql/docs/introduction",
            "https://cloud.google.com/iam/docs/overview",
            "https://cloud.google.com/vpc/docs/vpc"
        ],
        "domains": {
            "Setting Up a Cloud Solution Environment": [
                "IAM", "Compute Engine", "Cloud Storage"
            ],
            "Deploying and Implementing a Cloud Solution": [
                "Compute Engine", "Google Kubernetes Engine", "Cloud Run", "Cloud SQL", "App Engine"
            ],
            "Configuring Access and Security": [
                "IAM", "VPC"
            ]
        }
    },
    "adp": {
        "name": "Associate Data Practitioner",
        "docs": [
            "https://cloud.google.com/bigquery/docs/introduction",
            "https://cloud.google.com/storage/docs/introduction",
            "https://cloud.google.com/pubsub/docs/overview",
            "https://cloud.google.com/dataflow/docs/concepts/overview",
            "https://cloud.google.com/sql/docs/introduction",
            "https://cloud.google.com/looker/docs/intro"
        ],
        "domains": {
            "Data Ingestion and Pipelines": [
                "Cloud Storage", "Pub/Sub", "Dataflow"
            ],
            "Data Storage and Querying": [
                "BigQuery", "Cloud SQL", "Datastore"
            ],
            "Data Visualization and Insights": [
                "Looker", "BigQuery"
            ]
        }
    },
    # Professional Tier
    "pca": {
        "name": "Professional Cloud Architect",
        "docs": [
            "https://cloud.google.com/architecture/framework",
            "https://services.google.com/fh/files/misc/v6.1_pca_altostrat_media_case_study_english.pdf",
            "https://services.google.com/fh/files/misc/v6.1_pca_cymbal_retail_case_study_english.pdf",
            "https://services.google.com/fh/files/misc/v6.1_pca_ehr_healthcare_case_study_english.pdf",
            "https://services.google.com/fh/files/misc/v6.1_pca_knightmotives_automotive_case_study_english.pdf"
        ],
        "domains": {
            "Designing for Security and Compliance": [
                "IAM", "VPC", "Cloud Armor", "Key Management Service", "Identity-Aware Proxy"
            ],
            "Designing for Reliability, Disaster Recovery, and Business Continuity": [
                "Compute Engine", "Google Kubernetes Engine", "Cloud Storage", "Cloud SQL", "Cloud Spanner", "BigQuery"
            ],
            "Designing a Solution Infrastructure": [
                "Compute Engine", "Google Kubernetes Engine", "Cloud Run", "App Engine", "Cloud Storage"
            ]
        }
    },
    "pde": {
        "name": "Professional Data Engineer",
        "docs": [
            "https://cloud.google.com/bigquery/docs/introduction",
            "https://cloud.google.com/pubsub/docs/overview",
            "https://cloud.google.com/dataflow/docs/concepts/overview",
            "https://cloud.google.com/storage/docs/introduction",
            "https://cloud.google.com/bigtable/docs/overview",
            "https://cloud.google.com/dataproc/docs/concepts/overview",
            "https://cloud.google.com/spanner/docs/overview"
        ],
        "domains": {
            "Designing Data Processing Systems": [
                "BigQuery", "Pub/Sub", "Dataflow", "Cloud Storage", "Bigtable"
            ],
            "Building and Operationalizing Data Systems": [
                "Dataproc", "Bigtable", "Cloud Spanner", "Pub/Sub", "Dataflow"
            ],
            "Operationalizing Machine Learning Models": [
                "Vertex AI", "BigQuery"
            ]
        }
    },
    "pmle": {
        "name": "Professional Machine Learning Engineer",
        "docs": [
            "https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform",
            "https://cloud.google.com/storage/docs/introduction",
            "https://cloud.google.com/bigquery/docs/introduction",
            "https://cloud.google.com/compute/docs/concepts",
            "https://cloud.google.com/kubernetes-engine/docs/concepts/what-is-gke",
            "https://cloud.google.com/dataflow/docs/concepts/overview"
        ],
        "domains": {
            "ML Problem Framing and Data Preparation": [
                "Vertex AI", "Cloud Storage", "BigQuery"
            ],
            "ML Model Development and Training": [
                "Vertex AI", "Compute Engine", "Google Kubernetes Engine"
            ],
            "ML Pipeline Creation and Orchestration": [
                "Vertex AI", "Dataflow", "Artifact Registry"
            ],
            "ML Model Deployment and Monitoring": [
                "Vertex AI", "Operations Suite", "Cloud Run"
            ]
        }
    }
}

def compute_content_hash(content_bytes):
    """Computes MD5 hash for content comparison."""
    return hashlib.md5(content_bytes).hexdigest()

def chunk_text_sliding_window(text, chunk_size=1000, overlap=200):
    """Splits raw text into readable semantic chunks with a sliding window."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

def generate_embeddings_in_batches(chunks, batch_size=50):
    """Generates text-embedding-004 embeddings from Gemini API in batches."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key":
        print("Warning: GEMINI_API_KEY is not configured. Skipping embedding generation.")
        return []
        
    genai.configure(api_key=api_key)
    
    total_chunks = len(chunks)
    embedded_chunks = []
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        
        print(f"Generating embeddings for batch {i//batch_size + 1} ({len(batch)} chunks)...")
        
        max_retries = 5
        delay = 2  # Start with a 2-second sleep delay
        embeddings = None
        
        for attempt in range(max_retries):
            try:
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=texts,
                    task_type="retrieval_document"
                )
                embeddings = result.get("embedding", [])
                break
            except gcp_errors.ResourceExhausted:
                print(f"⚠️ Rate limit hit. Sleeping {delay}s (Attempt {attempt+1}/{max_retries})...")
                time.sleep(delay)
                delay *= 2  # Exponentially back off the request frequency
            except Exception as e:
                print(f"⚠️ Error during API call (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(delay)
                delay *= 2
                
        if embeddings is None:
            raise Exception("❌ API request failed: Gemini Free Tier rate limit exceeded or query failed.")
            
        for idx, embedding in enumerate(embeddings):
            chunk_copy = batch[idx].copy()
            chunk_copy["embedding"] = embedding
            embedded_chunks.append(chunk_copy)
            
    return embedded_chunks


def sync_source(driver, source_key, source_name, url, exam_id, all_services, is_pdf=False):
    """Performs sync operations for a single data source under the exam context."""
    print(f"\n--- Checking status of {source_name} for Exam {exam_id.upper()} ---")
    
    # 1. Download content bytes to calculate MD5 hash
    try:
        res = requests.get(url, timeout=15)
        res.raise_for_status()
        content_bytes = res.content
    except Exception as e:
        print(f"Aborting sync for {source_name}: Failed to fetch source from {url}. Error: {e}")
        return False
        
    current_hash = compute_content_hash(content_bytes)
    
    # 2. Check if hash matches stored hash (scoped by exam ID to prevent collision)
    stored_hash_key = f"{exam_id}_{source_key}"
    stored_hash = get_stored_hash(driver, stored_hash_key)
    
    if current_hash == stored_hash:
        print(f"No changes detected for '{source_name}' (Exam: {exam_id}). Database is up to date.")
        return True
        
    print(f"Content delta detected for '{source_name}' (Exam: {exam_id})! Commencing ingestion...")
    
    # 3. Parse chunks
    if is_pdf:
        raw_chunks = parse_pdf_case_study(source_key, url)
    else:
        raw_chunks = parse_html_framework(url, source_name)
        
    if not raw_chunks:
        print(f"No chunks extracted from '{source_name}'. Skipping write phase.")
        return False
        
    # Apply sliding-window text chunking to keep chunk sizes consistent
    chunks = []
    for rc in raw_chunks:
        text_slices = chunk_text_sliding_window(rc["text"], chunk_size=1000, overlap=200)
        if len(text_slices) <= 1:
            chunks.append(rc)
        else:
            for idx, slice_text in enumerate(text_slices):
                sub_chunk = rc.copy()
                sub_chunk["chunk_id"] = f"{rc['chunk_id']}_part{idx}"
                sub_chunk["title"] = f"{rc['title']} (Part {idx + 1})"
                sub_chunk["text"] = slice_text
                chunks.append(sub_chunk)

    # Prefix chunk IDs with exam_id to prevent collision in multi-exam schema
    for chunk in chunks:
        chunk["chunk_id"] = f"{exam_id}_{chunk['chunk_id']}"
        
    print(f"Extracted and chunked into {len(chunks)} elements from '{source_name}'. Writing to Neo4j...")
    
    # 4. Delete stale chunks for this source and exam context
    delete_source_chunks(driver, source_name, exam_id=exam_id)
    
    # 5. Insert raw chunks into Neo4j in batch
    insert_chunks_batch(driver, chunks, source_exam=exam_id)
    link_chunks_to_services_batch(driver, chunks, all_services)
        
    # 6. Establish sequential NEXT relationship chain
    build_sequential_relationships(driver, source_name, exam_id=exam_id)
    
    # 7. Generate embeddings and upload to Neo4j in batch
    embedded_chunks = generate_embeddings_in_batches(chunks)
    if embedded_chunks:
        print(f"Uploading {len(embedded_chunks)} embeddings to Neo4j for '{source_name}'...")
        embeddings_batch = [
            {"uid": chunk["chunk_id"], "embedding": chunk["embedding"]}
            for chunk in embedded_chunks
            if "embedding" in chunk
        ]
        update_chunk_embeddings_batch(driver, embeddings_batch)
                
    # 8. Record new hash configuration
    set_stored_hash(driver, stored_hash_key, current_hash)
    print(f"Successfully synchronized metadata hash state for '{source_name}' (Exam: {exam_id})!")
    return True



def purge_inactive_users(driver):
    """Deletes temporary portfolio user profiles older than 14 days."""
    query = """
    MATCH (u:User)
    // Find users whose last active timestamp is older than 14 days
    WHERE u.last_active < timestamp() - (14 * 24 * 60 * 60 * 1000)
    // Detach deletes the user and all their HAS_MASTERY relationships
    DETACH DELETE u
    """
    with driver.session() as session:
        result = session.run(query)
        summary = result.consume()
        print(f"🧹 Successfully purged {summary.counters.nodes_deleted} inactive test profiles.")

def run_pipeline():
    """Main orchestrator for multi-exam document delta sync checks and Neo4j imports."""
    print("Initializing GCP Exam GraphRAG Ingestion & Sync Pipeline (Multi-Exam Model)...")
    
    try:
        driver = get_driver()
    except Exception as e:
        print(f"Initialization Error: {e}")
        return
        
    if not verify_connection(driver):
        print("Error: Neo4j database is unreachable. Please verify your credentials.")
        return
        
    # Set up database unique constraints
    setup_schema(driver)
    
    # Gather all services defined in the domains of all exams to assist chunk mapping
    all_services = set()
    for exam_config in GCP_EXAMS.values():
        for domain_services in exam_config.get("domains", {}).values():
            all_services.update(domain_services)
            
    # 1. Seed exam blueprints, domains, and service taxonomy
    for exam_id, config in GCP_EXAMS.items():
        print(f"Seeding blueprint metadata for {config['name']} ({exam_id.upper()})...")
        setup_exam_metadata(driver, exam_id, config["name"], config.get("domains", {}))
        
    # 2. Sync documentation chunks for each exam
    for exam_id, config in GCP_EXAMS.items():
        for doc_url in config["docs"]:
            is_pdf = doc_url.lower().endswith(".pdf")
            
            # Determine source names/keys
            if is_pdf:
                # e.g. "altostrat_media_case_study"
                name_match = re.search(r'pca_([a-z0-9_]+)_case_study', doc_url)
                if name_match:
                    source_key = name_match.group(1)
                else:
                    source_key = doc_url.split('/')[-1].replace('.pdf', '')
                source_name = f"Case Study: {source_key.replace('_', ' ').title()}"
            else:
                source_key = "framework" if "framework" in doc_url else doc_url.split('/')[-1]
                source_name = "Architecture Framework" if source_key == "framework" else source_key.replace('-', ' ').title()
                
            sync_source(
                driver,
                source_key=source_key,
                source_name=source_name,
                url=doc_url,
                exam_id=exam_id,
                all_services=all_services,
                is_pdf=is_pdf
            )
            

            
    # 4. Validate and enforce vector search index
    ensure_vector_index(driver)
    
    # 5. Automatically purge inactive users (older than 14 days)
    purge_inactive_users(driver)
    
    driver.close()
    print("\nMulti-Exam database delta-sync processes finished.")

if __name__ == "__main__":
    run_pipeline()
