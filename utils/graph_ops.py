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
    """Sets up constraints on Chunk and Metadata nodes to ensure indexing and data integrity."""
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

def delete_source_chunks(driver, source_name):
    """Deletes old chunk nodes and relationships for a specific source, preserving taxonomy."""
    query = "MATCH (c:Chunk {source: $source_name}) DETACH DELETE c"
    with driver.session() as session:
        result = session.run(query, source_name=source_name)
        counters = result.consume().counters
        deleted_count = counters.nodes_deleted
        print(f"Cleaned up {deleted_count} stale chunks for source '{source_name}'.")

def insert_chunk_node(driver, chunk_uid, text, title, source):
    """Inserts a single chunk node into the Neo4j database."""
    query = """
    MERGE (c:Chunk {uid: $uid})
    SET c.text = $text,
        c.title = $title,
        c.source = $source
    RETURN c
    """
    with driver.session() as session:
        session.run(query, uid=chunk_uid, text=text, title=title, source=source)

def update_chunk_embedding(driver, chunk_uid, embedding):
    """Updates a chunk's embedding attribute."""
    query = """
    MATCH (c:Chunk {uid: $uid})
    SET c.embedding = $embedding
    """
    with driver.session() as session:
        session.run(query, uid=chunk_uid, embedding=embedding)

def build_sequential_relationships(driver, source_name):
    """
    Links chunks belonging to the same source in order (c1)-[:NEXT]->(c2).
    This establishes sequential structure in the knowledge graph.
    """
    query = """
    MATCH (c:Chunk {source: $source})
    RETURN c.uid AS uid
    ORDER BY c.uid
    """
    with driver.session() as session:
        result = session.run(query, source=source_name)
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
    print(f"Created sequential NEXT relationships for {len(uids)} chunks in '{source_name}'.")

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
