import os
import hashlib
import requests
import google.generativeai as genai
from utils.graph_ops import (
    load_environment,
    get_driver,
    verify_connection,
    setup_schema,
    get_stored_hash,
    set_stored_hash,
    delete_source_chunks,
    insert_chunk_node,
    update_chunk_embedding,
    build_sequential_relationships,
    ensure_vector_index
)
from data.fetch_docs import TARGET_DOCS, parse_html_framework, parse_pdf_case_study

# Initialize environment configuration
load_environment()

def compute_content_hash(content_bytes):
    """Computes MD5 hash for content comparison."""
    return hashlib.md5(content_bytes).hexdigest()

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
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=texts,
                task_type="retrieval_document"
            )
            embeddings = result.get("embedding", [])
            
            for idx, embedding in enumerate(embeddings):
                chunk_copy = batch[idx].copy()
                chunk_copy["embedding"] = embedding
                embedded_chunks.append(chunk_copy)
        except Exception as e:
            print(f"Error generating embeddings for batch starting at index {i}: {e}")
            
    return embedded_chunks

def sync_source(driver, source_key, source_name, url, is_pdf=False):
    """Performs sync operations for a single data source."""
    print(f"\n--- Checking status of {source_name} ---")
    
    # 1. Download content bytes to calculate MD5 hash
    try:
        res = requests.get(url, timeout=15)
        res.raise_for_status()
        content_bytes = res.content
    except Exception as e:
        print(f"Aborting sync for {source_name}: Failed to fetch source from {url}. Error: {e}")
        return False
        
    current_hash = compute_content_hash(content_bytes)
    
    # 2. Check if hash matches stored hash
    stored_hash = get_stored_hash(driver, source_key)
    
    if current_hash == stored_hash:
        print(f"No changes detected for '{source_name}'. Database is up to date.")
        return True
        
    print(f"Content delta detected for '{source_name}'! Commencing ingestion...")
    
    # 3. Parse chunks
    if is_pdf:
        chunks = parse_pdf_case_study(source_key, url)
    else:
        chunks = parse_html_framework(url)
        
    if not chunks:
        print(f"No chunks extracted from '{source_name}'. Skipping write phase.")
        return False
        
    print(f"Extracted {len(chunks)} chunks from '{source_name}'. Writing to Neo4j...")
    
    # 4. Delete stale chunks for this source
    delete_source_chunks(driver, source_name)
    
    # 5. Insert raw chunks into Neo4j
    for chunk in chunks:
        insert_chunk_node(
            driver,
            chunk_uid=chunk["chunk_id"],
            text=chunk["text"],
            title=chunk["title"],
            source=chunk["source"]
        )
        
    # 6. Establish sequential NEXT relationship chain
    build_sequential_relationships(driver, source_name)
    
    # 7. Generate embeddings and upload to Neo4j
    embedded_chunks = generate_embeddings_in_batches(chunks)
    if embedded_chunks:
        print(f"Uploading {len(embedded_chunks)} embeddings to Neo4j for '{source_name}'...")
        for chunk in embedded_chunks:
            if "embedding" in chunk:
                update_chunk_embedding(driver, chunk["chunk_id"], chunk["embedding"])
                
    # 8. Record new hash configuration
    set_stored_hash(driver, source_key, current_hash)
    print(f"Successfully synchronized metadata hash state for '{source_name}'!")
    return True

def run_pipeline():
    """Main orchestrator for document delta sync checks and Neo4j imports."""
    print("Initializing GCP Exam GraphRAG Ingestion & Sync Pipeline...")
    
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
    
    # 1. Sync Architecture Framework HTML
    sync_source(
        driver,
        source_key="framework",
        source_name="Architecture Framework",
        url=TARGET_DOCS["framework"],
        is_pdf=False
    )
    
    # 2. Sync all PDF Case Studies
    for key, url in TARGET_DOCS["case_studies"].items():
        case_name = f"Case Study: {key.replace('_', ' ').title()}"
        sync_source(
            driver,
            source_key=f"cs_{key}",
            source_name=case_name,
            url=url,
            is_pdf=True
        )
        
    # 3. Validate and enforce vector search index
    ensure_vector_index(driver)
    
    driver.close()
    print("\nIn-place database delta-sync processes finished.")

if __name__ == "__main__":
    run_pipeline()
