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

        # 6. Unique ID constraint on Question nodes
        try:
            session.run("CREATE CONSTRAINT question_uid IF NOT EXISTS FOR (q:Question) REQUIRE q.uid IS UNIQUE")
            print("Verified constraint: Question(uid) is unique.")
        except Exception as e:
            print(f"Note creating Question constraint: {e}")

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
    """Scans the text of a chunk and connects it to Service nodes using boundary matching."""
    import re
    matched_services = []
    for service_name in all_services:
        # Match boundaries allowing punctuation next to words (e.g. Pub/Sub)
        pattern = r'(?i)(?<![a-zA-Z0-9])' + re.escape(service_name) + r'(?![a-zA-Z0-9])'
        if re.search(pattern, text):
            matched_services.append(service_name)
            
    if not matched_services:
        return
        
    query = """
    MATCH (c:Chunk {uid: $uid})
    UNWIND $services AS service_name
    MERGE (s:Service {name: service_name})
    MERGE (c)-[:DISCUSSES]->(s)
    """
    with driver.session() as session:
        session.run(query, uid=chunk_uid, services=matched_services)

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

class UserStateController:
    def __init__(self, driver):
        self.driver = driver

    def touch_user(self, user_id: str):
        """Ensures the User node exists and updates their last_active timestamp."""
        query = """
        MERGE (u:User {id: $user_id})
        SET u.last_active = timestamp()
        """
        with self.driver.session() as session:
            session.run(query, user_id=user_id)

    def update_mastery(self, user_id: str, node_name: str, passed: bool):
        """Updates Thompson sampling parameters based on quiz results."""
        self.touch_user(user_id)
        query = """
        MATCH (u:User {id: $user_id})
        MATCH (n) WHERE (n:Service OR n:Domain) AND n.name = $node_name
        MERGE (u)-[r:HAS_MASTERY]->(n)
        ON CREATE SET r.alpha = 1, r.beta = 1
        SET r.alpha = r.alpha + CASE WHEN $passed = false THEN 1 ELSE 0 END,
            r.beta = r.beta + CASE WHEN $passed = true THEN 1 ELSE 0 END
        """
        with self.driver.session() as session:
            session.run(query, user_id=user_id, node_name=node_name, passed=passed)

    def reset_single_node(self, user_id: str, node_name: str):
        """Level 1: Reset a single service or domain node back to default priors."""
        self.touch_user(user_id)
        query = """
        MATCH (u:User {id: $user_id})-[r:HAS_MASTERY]->(n)
        WHERE n.name = $node_name
        SET r.alpha = 1, r.beta = 1
        """
        with self.driver.session() as session:
            session.run(query, user_id=user_id, node_name=node_name)

    def reset_entire_exam(self, user_id: str, exam_id: str):
        """Level 2: Reset all nodes mapped to a specific Google certification."""
        self.touch_user(user_id)
        query = """
        MATCH (u:User {id: $user_id})-[r:HAS_MASTERY]->(target)
        WHERE (target:Service OR target:Domain)
        AND (
            EXISTS {
                MATCH (e:Exam {id: $exam_id})-[:HAS_DOMAIN]->(target)
            } OR EXISTS {
                MATCH (e:Exam {id: $exam_id})-[:REQUIRES_SERVICE]->(target)
            }
        )
        SET r.alpha = 1, r.beta = 1
        """
        with self.driver.session() as session:
            session.run(query, user_id=user_id, exam_id=exam_id)

    def reset_global_profile(self, user_id: str):
        """Level 3: Reset the entire student profile back to a blank slate."""
        self.touch_user(user_id)
        query = """
        MATCH (u:User {id: $user_id})-[r:HAS_MASTERY]->()
        DELETE r
        """
        with self.driver.session() as session:
            session.run(query, user_id=user_id)

    def get_user_mastery_stats(self, user_id: str, exam_id: str):
        """Retrieves alpha and beta values for all services and domains of an exam."""
        self.touch_user(user_id)
        query = """
        MATCH (e:Exam {id: $exam_id})
        OPTIONAL MATCH (e)-[:HAS_DOMAIN]->(d:Domain)
        OPTIONAL MATCH (d)-[:TESTS_KNOWLEDGE_OF]->(s:Service)
        WITH DISTINCT d, s
        MATCH (u:User {id: $user_id})
        OPTIONAL MATCH (u)-[rd:HAS_MASTERY]->(d)
        OPTIONAL MATCH (u)-[rs:HAS_MASTERY]->(s)
        RETURN 
            d.name AS domain_name, 
            rd.alpha AS domain_alpha, 
            rd.beta AS domain_beta,
            s.name AS service_name,
            rs.alpha AS service_alpha,
            rs.beta AS service_beta
        """
        with self.driver.session() as session:
            result = session.run(query, user_id=user_id, exam_id=exam_id)
            stats = {
                "domains": {},
                "services": {}
            }
            for record in result:
                d_name = record["domain_name"]
                if d_name:
                    stats["domains"][d_name] = {
                        "alpha": record["domain_alpha"] if record["domain_alpha"] is not None else 1,
                        "beta": record["domain_beta"] if record["domain_beta"] is not None else 1
                    }
                s_name = record["service_name"]
                if s_name:
                    stats["services"][s_name] = {
                        "alpha": record["service_alpha"] if record["service_alpha"] is not None else 1,
                        "beta": record["service_beta"] if record["service_beta"] is not None else 1
                    }
            return stats

def insert_question_node(driver, uid, question_text, options, correct_answer, explanation, exam_id, all_services):
    """Inserts a Question node and connects it to the target Exam and matching Services."""
    import re
    import json
    
    # Convert options list to dict A, B, C, D if list
    if isinstance(options, list):
        opt_labels = ["A", "B", "C", "D"]
        options_dict = {opt_labels[i]: options[i] for i in range(min(len(options), 4))}
    else:
        options_dict = options
        
    query = """
    MERGE (q:Question {uid: $uid})
    SET q.question = $question,
        q.options = $options,
        q.correct_answer = $correct_answer,
        q.explanation = $explanation,
        q.source_exam = $exam_id
    """
    with driver.session() as session:
        session.run(
            query,
            uid=uid,
            question=question_text,
            options=json.dumps(options_dict) if isinstance(options_dict, dict) else options_dict,
            correct_answer=correct_answer,
            explanation=explanation,
            exam_id=exam_id
        )
        
        # Connect Question to Exam
        session.run("""
            MATCH (q:Question {uid: $uid})
            MATCH (e:Exam {id: $exam_id})
            MERGE (q)-[:PART_OF]->(e)
        """, uid=uid, exam_id=exam_id)
        
        # Link to Services based on mentions in text/explanation
        matched_services = []
        combined_text = f"{question_text} {explanation}"
        for service_name in all_services:
            pattern = r'(?i)(?<![a-zA-Z0-9])' + re.escape(service_name) + r'(?![a-zA-Z0-9])'
            if re.search(pattern, combined_text):
                matched_services.append(service_name)
                
        if matched_services:
            session.run("""
                MATCH (q:Question {uid: $uid})
                UNWIND $services AS service_name
                MERGE (s:Service {name: service_name})
                MERGE (q)-[:TESTS]->(s)
            """, uid=uid, services=matched_services)

