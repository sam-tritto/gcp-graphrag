import os
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.indexes import create_vector_index

def load_environment():
    """Loads environment variables with fallback hierarchy for the .env folder."""
    env_paths = [
        Path(".env"),                             # Root file
        Path(".env/.env.development"),            # Subfolder dev file
        Path(".env/.env"),                        # Subfolder default file
        Path("../.env/.env.development"),         # Subfolder dev file (from data/utils directory)
    ]
    
    loaded = False
    for path in env_paths:
        if path.is_file():
            load_dotenv(path)
            loaded = True
            break
            
    if not loaded:
        # Fallback to standard dotenv loading (which traverses up)
        load_dotenv()

# Initialize environment configuration
load_environment()

def get_driver():
    """Initializes and returns a Neo4j driver instance."""
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not uri or not password:
        raise ValueError(
            "NEO4J_URI and NEO4J_PASSWORD environment variables must be set.\n"
            "Please check your environment configuration."
        )
        
    return GraphDatabase.driver(uri, auth=(username, password))

def verify_connection(driver):
    """Verifies that the Neo4j database is reachable."""
    try:
        driver.verify_connectivity()
        return True
    except Exception as e:
        print(f"Neo4j connectivity check failed: {e}")
        return False

def setup_schema(driver):
    """Sets up constraints on nodes to ensure indexing and data integrity."""
    with driver.session() as session:
        # 1. Unique ID constraint on Chunk nodes
        try:
            session.run("CREATE CONSTRAINT chunk_uid IF NOT EXISTS FOR (c:Chunk) REQUIRE c.uid IS UNIQUE")
            print("Verified constraint: Chunk(uid) is unique.")
        except Exception as e:
            print(f"Note creating Chunk constraint: {e}")
            
        # 2. Unique ID constraint on Metadata nodes
        try:
            session.run("CREATE CONSTRAINT metadata_id IF NOT EXISTS FOR (m:Metadata) REQUIRE m.id IS UNIQUE")
            print("Verified constraint: Metadata(id) is unique.")
        except Exception as e:
            print(f"Note creating Metadata constraint: {e}")

        # 3. Unique ID constraint on Exam nodes
        try:
            session.run("CREATE CONSTRAINT exam_id IF NOT EXISTS FOR (e:Exam) REQUIRE e.id IS UNIQUE")
            print("Verified constraint: Exam(id) is unique.")
        except Exception as e:
            print(f"Note creating Exam constraint: {e}")

        # 4. Unique Name constraint on Domain nodes
        try:
            session.run("CREATE CONSTRAINT domain_name IF NOT EXISTS FOR (d:Domain) REQUIRE d.name IS UNIQUE")
            print("Verified constraint: Domain(name) is unique.")
        except Exception as e:
            print(f"Note creating Domain constraint: {e}")

        # 5. Unique Name constraint on Service nodes
        try:
            session.run("CREATE CONSTRAINT service_name IF NOT EXISTS FOR (s:Service) REQUIRE s.name IS UNIQUE")
            print("Verified constraint: Service(name) is unique.")
        except Exception as e:
            print(f"Note creating Service constraint: {e}")

def setup_exam_metadata(driver, exam_id, exam_name, domains_config):
    """Seeds the Exam, Domain, and Service blueprint metadata and links them."""
    with driver.session() as session:
        # Create or update core Exam node
        session.run("""
            MERGE (e:Exam {id: $exam_id})
            SET e.name = $name
        """, exam_id=exam_id, name=exam_name)
        
        # Loop through domains and their services
        for domain_name, services in domains_config.items():
            session.run("""
                MERGE (d:Domain {name: $domain_name})
                WITH d
                MATCH (e:Exam {id: $exam_id})
                MERGE (e)-[:HAS_DOMAIN]->(d)
            """, exam_id=exam_id, domain_name=domain_name)
            
            for service_name in services:
                session.run("""
                    MERGE (s:Service {name: $service_name})
                    WITH s
                    MATCH (e:Exam {id: $exam_id})
                    MERGE (e)-[:REQUIRES_SERVICE]->(s)
                    WITH s
                    MATCH (d:Domain {name: $domain_name})
                    MERGE (d)-[:TESTS_KNOWLEDGE_OF]->(s)
                """, exam_id=exam_id, domain_name=domain_name, service_name=service_name)

def get_stored_hash(driver, source_id):
    """Retrieves the stored content hash for a document sync status."""
    query = "MATCH (m:Metadata {id: $source_id}) RETURN m.hash AS hash"
    with driver.session() as session:
        result = session.run(query, source_id=source_id)
        record = result.single()
        return record["hash"] if record else None

def set_stored_hash(driver, source_id, content_hash):
    """Saves the content hash for a document sync status."""
    query = "MERGE (m:Metadata {id: $source_id}) SET m.hash = $hash, m.last_updated = datetime()"
    with driver.session() as session:
        session.run(query, source_id=source_id, hash=content_hash)

def delete_source_chunks(driver, source_name, exam_id=None):
    """Deletes old chunk nodes and relationships for a specific source, preserving taxonomy."""
    if exam_id:
        query = "MATCH (c:Chunk {source: $source_name, source_exam: $exam_id}) DETACH DELETE c"
    else:
        query = "MATCH (c:Chunk {source: $source_name}) DETACH DELETE c"
    with driver.session() as session:
        result = session.run(query, source_name=source_name, exam_id=exam_id)
        counters = result.consume().counters
        deleted_count = counters.nodes_deleted
        print(f"Cleaned up {deleted_count} stale chunks for source '{source_name}' (Exam: {exam_id}).")

def insert_chunk_node(driver, chunk_uid, text, title, source, source_exam=None):
    """Inserts a single chunk node, setting exam metadata and relationships if provided."""
    query = """
    MERGE (c:Chunk {uid: $uid})
    SET c.text = $text,
        c.title = $title,
        c.source = $source
    """
    if source_exam:
        query += ", c.source_exam = $source_exam"
        
    query += "\nRETURN c"
    
    with driver.session() as session:
        session.run(query, uid=chunk_uid, text=text, title=title, source=source, source_exam=source_exam)
        
        if source_exam:
            # Connect Chunk to Exam node
            session.run("""
                MATCH (c:Chunk {uid: $uid})
                MATCH (e:Exam {id: $exam_id})
                MERGE (c)-[:RELEVANT_TO]->(e)
            """, uid=chunk_uid, exam_id=source_exam)

def link_chunk_to_services(driver, chunk_uid, text, all_services):
    """Scans the text of a chunk and connects it to Service nodes using word-boundary matching."""
    import re
    with driver.session() as session:
        for service_name in all_services:
            # Word boundary matching (case-insensitive)
            # Escaping service_name. Note that for Pub/Sub, the slash is a non-word boundary.
            # We can use a regex that matches on boundary or whitespace/punctuation boundaries.
            pattern = r'(?i)\b' + re.escape(service_name) + r'\b'
            if re.search(pattern, text):
                session.run("""
                    MATCH (c:Chunk {uid: $uid})
                    MERGE (s:Service {name: $service_name})
                    MERGE (c)-[:DISCUSSES]->(s)
                """, uid=chunk_uid, service_name=service_name)

def update_chunk_embedding(driver, chunk_uid, embedding):
    """Updates a chunk's embedding attribute."""
    query = """
    MATCH (c:Chunk {uid: $uid})
    SET c.embedding = $embedding
    """
    with driver.session() as session:
        session.run(query, uid=chunk_uid, embedding=embedding)

def build_sequential_relationships(driver, source_name, exam_id=None):
    """
    Links chunks belonging to the same source in order (c1)-[:NEXT]->(c2).
    This establishes sequential structure in the knowledge graph.
    """
    query = "MATCH (c:Chunk {source: $source"
    if exam_id:
        query += ", source_exam: $exam_id"
    query += "}) RETURN c.uid AS uid ORDER BY c.uid"
    with driver.session() as session:
        result = session.run(query, source=source_name, exam_id=exam_id)
        uids = [record["uid"] for record in result]
        
    if len(uids) < 2:
        return
        
    link_query = """
    MATCH (c1:Chunk {uid: $uid1})
    MATCH (c2:Chunk {uid: $uid2})
    MERGE (c1)-[:NEXT]->(c2)
    """
    with driver.session() as session:
        for idx in range(len(uids) - 1):
            session.run(link_query, uid1=uids[idx], uid2=uids[idx+1])
    print(f"Created sequential NEXT relationships for {len(uids)} chunks in '{source_name}' (Exam: {exam_id}).")

def ensure_vector_index(driver):
    """Guarantees presence of the Neo4j vector search index."""
    try:
        create_vector_index(
            driver,
            name="gcp_exam_embeddings",
            label="Chunk",
            embedding_property="embedding",
            dimensions=768,
            similarity_fn="cosine"
        )
        print("Neo4j Vector index 'gcp_exam_embeddings' verified/created.")
    except Exception as e:
        print(f"Vector index verified (or already exists): {e}")
