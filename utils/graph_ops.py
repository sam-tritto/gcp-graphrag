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
            load_dotenv(path, override=True)
            loaded = True
            break
            
    if not loaded:
        # Fallback to standard dotenv loading (which traverses up)
        load_dotenv(override=True)

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

        # 6. Unique constraints on Third-Tier nodes
        constraints = [
            ("UseCase", "description"),
            ("CLICommand", "syntax"),
            ("FunctionalRole", "description"),
            ("Constraint", "description"),
            ("OptimizationPattern", "description"),
            ("EvaluationMetric", "description"),
            ("HierarchyNode", "name"),
            ("CaseStudy", "name"),
            ("SubConcept", "name"),
            ("AntiPattern", "description")
        ]
        for label, prop in constraints:
            try:
                session.run(f"CREATE CONSTRAINT {label.lower()}_{prop} IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE")
                print(f"Verified constraint: {label}({prop}) is unique.")
            except Exception as e:
                print(f"Note creating {label} constraint: {e}")



def setup_exam_metadata(driver, exam_id, exam_name, domains_config):
    """Seeds the Exam, Domain, Service, and SubConcept blueprint metadata and links them."""
    with driver.session() as session:
        # Create or update core Exam node
        session.run("""
            MERGE (e:Exam {id: $exam_id})
            SET e.name = $name
        """, exam_id=exam_id, name=exam_name)
        
        # Loop through domains and their services
        for domain_name, service_data in domains_config.items():
            session.run("""
                MERGE (d:Domain {name: $domain_name})
                WITH d
                MATCH (e:Exam {id: $exam_id})
                MERGE (e)-[:HAS_DOMAIN]->(d)
            """, exam_id=exam_id, domain_name=domain_name)
            
            # Support both old format (list of services) and new format (dict of services -> subconcepts)
            if isinstance(service_data, list):
                services_dict = {s: [s] for s in service_data}
            else:
                services_dict = service_data
                
            for service_name, subconcepts in services_dict.items():
                session.run("""
                    MERGE (s:Service {name: $service_name})
                    WITH s
                    MATCH (e:Exam {id: $exam_id})
                    MERGE (e)-[:REQUIRES_SERVICE]->(s)
                    WITH s
                    MATCH (d:Domain {name: $domain_name})
                    MERGE (d)-[:TESTS_KNOWLEDGE_OF]->(s)
                """, exam_id=exam_id, domain_name=domain_name, service_name=service_name)
                
                for sub_name in subconcepts:
                    session.run("""
                        MERGE (sub:SubConcept {name: $sub_name})
                        WITH sub
                        MATCH (s:Service {name: $service_name})
                        MERGE (s)-[:HAS_SUBCONCEPT]->(sub)
                    """, service_name=service_name, sub_name=sub_name)

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

def insert_chunks_batch(driver, chunks, source_exam=None):
    """Inserts multiple chunk nodes and links them to the exam context in a single query transaction."""
    if not chunks:
        return
    
    query = """
    UNWIND $batches AS batch
    MERGE (c:Chunk {uid: batch.uid})
    SET c.text = batch.text,
        c.title = batch.title,
        c.source = batch.source
    """
    if source_exam:
        query += ", c.source_exam = $source_exam"
        
    with driver.session() as session:
        session.run(query, batches=[{
            "uid": chunk["chunk_id"],
            "text": chunk["text"],
            "title": chunk["title"],
            "source": chunk["source"]
        } for chunk in chunks], source_exam=source_exam)
        
        if source_exam:
            # Link chunks to the Exam node in batch
            rel_query = """
            UNWIND $uids AS uid
            MATCH (c:Chunk {uid: uid})
            MATCH (e:Exam {id: $exam_id})
            MERGE (c)-[:RELEVANT_TO]->(e)
            """
            session.run(rel_query, uids=[c["chunk_id"] for c in chunks], exam_id=source_exam)

def link_chunks_to_services_batch(driver, chunks, all_services):
    """Scans text of multiple chunks and connects them to Service nodes in a single batch transaction."""
    import re
    batch_data = []
    for chunk in chunks:
        chunk_uid = chunk["chunk_id"]
        text = chunk["text"]
        matched_services = []
        for service_name in all_services:
            pattern = r'(?i)(?<![a-zA-Z0-9])' + re.escape(service_name) + r'(?![a-zA-Z0-9])'
            if re.search(pattern, text):
                matched_services.append(service_name)
        if matched_services:
            batch_data.append({
                "uid": chunk_uid,
                "services": matched_services
            })
            
    if not batch_data:
        return
        
    query = """
    UNWIND $batch_data AS item
    MATCH (c:Chunk {uid: item.uid})
    UNWIND item.services AS service_name
    MERGE (s:Service {name: service_name})
    MERGE (c)-[:DISCUSSES]->(s)
    """
    with driver.session() as session:
        session.run(query, batch_data=batch_data)

def update_chunk_embeddings_batch(driver, embeddings_batch):
    """Updates multiple chunk embeddings in a single batch transaction."""
    if not embeddings_batch:
        return
    query = """
    UNWIND $batch AS item
    MATCH (c:Chunk {uid: item.uid})
    SET c.embedding = item.embedding
    """
    with driver.session() as session:
        session.run(query, batch=embeddings_batch)

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
    UNWIND $pairs AS pair
    MATCH (c1:Chunk {uid: pair.uid1})
    MATCH (c2:Chunk {uid: pair.uid2})
    MERGE (c1)-[:NEXT]->(c2)
    """
    pairs = [{"uid1": uids[idx], "uid2": uids[idx+1]} for idx in range(len(uids) - 1)]
    with driver.session() as session:
        session.run(link_query, pairs=pairs)
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
        MATCH (n) WHERE (n:Service OR n:Domain OR n:SubConcept) AND n.name = $node_name
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
        WHERE (target:Service OR target:Domain OR target:SubConcept)
        AND (
            EXISTS {
                MATCH (e:Exam {id: $exam_id})-[:HAS_DOMAIN]->(target)
            } OR EXISTS {
                MATCH (e:Exam {id: $exam_id})-[:REQUIRES_SERVICE]->(target)
            } OR EXISTS {
                MATCH (e:Exam {id: $exam_id})-[:REQUIRES_SERVICE]->(:Service)-[:HAS_SUBCONCEPT]->(target)
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
        """Retrieves alpha and beta values for all services, domains, and sub-concepts of an exam."""
        self.touch_user(user_id)
        query = """
        MATCH (e:Exam {id: $exam_id})
        OPTIONAL MATCH (e)-[:HAS_DOMAIN]->(d:Domain)
        OPTIONAL MATCH (d)-[:TESTS_KNOWLEDGE_OF]->(s:Service)
        OPTIONAL MATCH (s)-[:HAS_SUBCONCEPT]->(sub:SubConcept)
        WITH DISTINCT d, s, sub
        MATCH (u:User {id: $user_id})
        OPTIONAL MATCH (u)-[rd:HAS_MASTERY]->(d)
        OPTIONAL MATCH (u)-[rs:HAS_MASTERY]->(s)
        OPTIONAL MATCH (u)-[rsub:HAS_MASTERY]->(sub)
        RETURN 
            d.name AS domain_name, 
            rd.alpha AS domain_alpha, 
            rd.beta AS domain_beta,
            s.name AS service_name,
            rs.alpha AS service_alpha,
            rs.beta AS service_beta,
            sub.name AS sub_name,
            rsub.alpha AS sub_alpha,
            rsub.beta AS sub_beta
        """
        with self.driver.session() as session:
            result = session.run(query, user_id=user_id, exam_id=exam_id)
            stats = {
                "domains": {},
                "services": {},
                "subconcepts": {}
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
                sub_name = record["sub_name"]
                if sub_name:
                    stats["subconcepts"][sub_name] = {
                        "alpha": record["sub_alpha"] if record["sub_alpha"] is not None else 1,
                        "beta": record["sub_beta"] if record["sub_beta"] is not None else 1
                    }
            return stats

def setup_third_tier_metadata(driver):
    """Seeds the third-tier leaf nodes and dynamic connections for all certifications."""
    with driver.session() as session:
        # --- CDL UseCases ---
        use_cases = [
            ("Compute Engine", "Lifting and Shifting Virtual Machines with Full OS Control"),
            ("Google Kubernetes Engine", "Managing and Scaling Containerized Microservices using Kubernetes"),
            ("Cloud Run", "Serverless Container Execution with Scaling to Zero"),
            ("App Engine", "PaaS Platform for Hosting Web Applications without Managing Infrastructure"),
            ("Cloud Storage", "Unstructured Data & Cost-Effective Archiving"),
            ("Cloud Spanner", "Global Scalability with SQL/Strong Consistency"),
            ("BigQuery", "Serverless Enterprise Data Warehousing & Analytics"),
            ("Cloud SQL", "Fully Managed Relational Database for MySQL, PostgreSQL, and SQL Server"),
            ("Looker", "Governed Business Intelligence, Data Modeling, and Enterprise Reporting"),
            ("Vertex AI", "Building, Training, and Deploying Custom Machine Learning Models"),
            ("AutoML", "No-Code Custom Machine Learning Model Training"),
            ("Pre-trained APIs", "Out-of-the-box ML Capabilities for Vision, Translation, and Natural Language"),
            ("Apigee", "API Management, Governance, Security, and Analytics"),
            ("Anthos", "Hybrid and Multi-Cloud Container Management Platform"),
            ("VPC", "Private Cloud Networking, Subnets, and Security Controls"),
            ("IAM", "Resource Authorization, Fine-Grained Permissions, and Access Management"),
            ("Cloud Armor", "DDoS Protection and Web Application Firewall (WAF)"),
            ("Site Reliability Engineering (SRE) principles", "Improving Service Reliability, Automation, and Incident Response"),
            ("Service Level Objectives (SLOs/SLAs)", "Defining and Measuring Service Quality and Reliability Targets"),
            ("Cloud Logging", "Centralized Log Storage, Auditing, and Real-Time Log Analysis"),
            ("Cloud Monitoring", "Performance Dashboards, Infrastructure Metrics, and Alerting")
        ]
        for service, usecase in use_cases:
            session.run("""
                MERGE (s:Service {name: $service})
                MERGE (u:UseCase {description: $usecase})
                MERGE (s)-[:BEST_FOR]->(u)
            """, service=service, usecase=usecase)

        # --- ACE Operations & CLI Commands ---
        cli_commands = [
            ("Cloud Storage", "CONFIGURED_BY", "gsutil mb -c coldline -l us-east1 gs://bucket"),
            ("Google Kubernetes Engine", "SCALED_BY", "kubectl scale deployment --replicas=5"),
            ("Compute Engine", "CREATED_VIA", "gcloud compute instances create"),
            ("Cloud Run", "DEPLOYED_VIA", "gcloud run deploy"),
            ("Cloud SQL", "BACKED_UP_VIA", "gcloud sql backups create"),
            ("App Engine", "DEPLOYED_VIA", "gcloud app deploy"),
            ("IAM", "ROLES_ASSIGNED_VIA", "gcloud projects add-iam-policy-binding"),
            ("VPC", "SUBNET_CREATED_VIA", "gcloud compute networks subnets create"),
            ("Artifact Registry", "IMAGE_TAGGED_VIA", "docker tag & gcloud auth configure-docker"),
            ("Node Pools", "CREATED_VIA", "gcloud container node-pools create")
        ]
        for service, rel, command in cli_commands:
            session.run(f"""
                MERGE (s:Service {{name: $service}})
                MERGE (c:CLICommand {{syntax: $command}})
                MERGE (s)-[:{rel}]->(c)
            """, service=service, command=command)

        # GKE specific links for ACE
        session.run("""
            MERGE (gke:Service {name: 'Google Kubernetes Engine'})
            MERGE (ar:Service {name: 'Artifact Registry'})
            MERGE (np:Service {name: 'Node Pools'})
            MERGE (gke)-[:MAPS_TO]->(ar)
            MERGE (gke)-[:CONFIGURES]->(np)
        """)

        # Resource Hierarchy Links for ACE
        session.run("""
            MERGE (org:HierarchyNode {name: 'Organization'})
            MERGE (fld:HierarchyNode {name: 'Folder'})
            MERGE (prj:HierarchyNode {name: 'Project'})
            MERGE (res:HierarchyNode {name: 'Resource'})
            MERGE (org)-[:PARENT_OF]->(fld)
            MERGE (fld)-[:PARENT_OF]->(prj)
            MERGE (prj)-[:PARENT_OF]->(res)
        """)

        # --- ADP Roles & Ingestion Formats ---
        adp_roles = [
            ("Looker Studio", "USED_FOR", "Building Interactive Business Reports & Dashboards"),
            ("Cloud Storage", "ACTS_AS", "Raw Staging Zone for Pipeline Ingestion"),
            ("Pub/Sub", "ACTS_AS", "Real-Time Message Ingestion & Event Bus"),
            ("Dataflow", "USED_FOR", "ETL Pipeline Ingestion and Stream/Batch Processing"),
            ("Cloud SQL", "USED_FOR", "Transactional Database for Structured Application Data"),
            ("BigQuery", "USED_FOR", "Serverless Enterprise Data Warehousing & SQL Analytics"),
            ("Cloud Composer", "USED_FOR", "Orchestrating Complex Data Pipelines using Apache Airflow DAGs"),
            ("Looker Enterprise", "USED_FOR", "Governed Data Modeling and Single Source of Truth Reporting via LookML")
        ]
        for service, rel, role in adp_roles:
            session.run(f"""
                MERGE (s:Service {{name: $service}})
                MERGE (f:FunctionalRole {{description: $role}})
                MERGE (s)-[:{rel}]->(f)
            """, service=service, role=role)

        # ADP Ingestion Formats mapping
        ingestion_mappings = [
            ("CSV", "INGESTED_TO", "Cloud Storage"),
            ("JSON", "INGESTED_TO", "Cloud Storage"),
            ("Parquet", "STORED_AS", "Cloud Storage"),
            ("Avro", "STORED_AS", "Cloud Storage"),
            ("CSV", "LOADED_INTO", "BigQuery"),
            ("JSON", "LOADED_INTO", "BigQuery"),
            ("Parquet", "LOADED_INTO", "BigQuery"),
            ("Avro", "LOADED_INTO", "BigQuery")
        ]
        for fmt, rel, target in ingestion_mappings:
            session.run(f"""
                MERGE (f:HierarchyNode {{name: $fmt}})
                MERGE (t:Service {{name: $target}})
                MERGE (f)-[:{rel}]->(t)
            """, fmt=fmt, target=target)

        # --- PCA SLA & Constraints & Case Studies ---
        pca_constraints = [
            ("Cloud Spanner", "GUARANTEES", "99.999% Availability Multi-Region SLA"),
            ("Cloud SQL", "LIMITED_TO", "64 TB Storage Capacity"),
            ("Cloud Spanner", "ARCHITECTED_AS", "Globally Scalable Relational Database with Strong Consistency"),
            ("Cloud SQL", "ARCHITECTED_AS", "Regional SQL with Read Replicas"),
            ("Compute Engine", "LIMITED_TO", "Single-Zone Virtual Machine Availability SLA (99.9% / 99.99% with HA)"),
            ("VPC", "CONNECTS_VIA", "Shared VPC for Multi-Project Network Isolation"),
            ("Cloud Interconnect", "PROVIDES", "Direct Dedicated Network Connection up to 100 Gbps"),
            ("Cloud VPN", "PROVIDES", "IPsec VPN Connection over Public Internet up to 3 Gbps per tunnel"),
            ("SRE principles", "DEFINES", "Error Budgets as the Threshold for Deploying New Features"),
            ("Service Level Agreements (SLAs)", "MEASURED_BY", "Recovery Time Objective (RTO) & Recovery Point Objective (RPO)")
        ]
        for service, rel, constraint in pca_constraints:
            session.run(f"""
                MERGE (s:Service {{name: $service}})
                MERGE (c:Constraint {{description: $constraint}})
                MERGE (s)-[:{rel}]->(c)
            """, service=service, constraint=constraint)

        # PCA Case Studies
        case_studies = [
            ("EHR Healthcare", "Cloud Spanner"),
            ("EHR Healthcare", "Cloud SQL"),
            ("Mountkirk Games", "Google Kubernetes Engine"),
            ("Mountkirk Games", "Cloud Spanner"),
            ("Altostrat Media", "Compute Engine"),
            ("Cymbal Retail", "Cloud Run"),
            ("Cymbal Retail", "App Engine"),
            ("Knightmotives Automotive", "Pub/Sub"),
            ("Knightmotives Automotive", "BigQuery")
        ]
        for case_name, service in case_studies:
            session.run("""
                MERGE (c:CaseStudy {name: $case_name})
                MERGE (s:Service {name: $service})
                MERGE (c)-[:SOLVED_BY]->(s)
            """, case_name=case_name, service=service)

        # --- PDE Optimization Patterns ---
        pde_patterns = [
            ("BigQuery", "ACCESS_PATTERN", "Analytical & Cold Analytics"),
            ("Cloud Bigtable", "ACCESS_PATTERN", "Low-latency & High-throughput NoSQL"),
            ("Cloud Spanner", "ACCESS_PATTERN", "Global Transactional"),
            ("Dataproc", "USED_FOR", "Hadoop & Spark Lift-and-Shift"),
            ("Dataflow", "USED_FOR", "Serverless Stream & Batch Pipeline (Apache Beam) Pattern"),
            ("BigQuery", "OPTIMIZED_BY", "Partitioning on Date Fields + Clustering on High-Cardinality Columns"),
            ("Cloud Bigtable", "DESIGNED_USING", "Lexicographically Ordered Row Keys to Prevent Node Hotspotting"),
            ("Cloud Spanner", "OPTIMIZED_BY", "Interleaved Tables to Physically Co-locate Parent and Child Rows"),
            ("Cloud DLP", "USED_FOR", "De-identifying Sensitive PII Data before Processing"),
            ("Dataplex", "USED_FOR", "Governing Data Lakes and Data Warehouses with Unified Security"),
            ("Data Catalog", "USED_FOR", "Metadata Management and Data Lineage Tracing"),
            ("BigQuery", "SECURED_BY", "Authorized Views"),
            ("BigQuery", "SECURED_BY", "Row/Column-level Security")
        ]
        for service, rel, pattern in pde_patterns:
            session.run(f"""
                MERGE (s:Service {{name: $service}})
                MERGE (p:OptimizationPattern {{description: $pattern}})
                MERGE (s)-[:{rel}]->(p)
            """, service=service, pattern=pattern)

        # --- PMLE Metrics & Observability ---
        pmle_metrics = [
            ("Vertex AI Model Monitoring", "TRIGGERS_ON", "Data Drift threshold violations via K-S test"),
            ("Gemini Model Evaluation", "MEASURED_BY", "ROUGE and BLEU scores for text summarization"),
            ("Vertex AI Pipelines", "ORCHESTRATES", "Reproducible MLOps Workflows & Model Artifact Lineage"),
            ("AutoML", "EVALUATED_BY", "Area Under Precision-Recall Curve (AUPRC) & Confusion Matrix"),
            ("BigQuery ML", "EVALUATED_BY", "Mean Absolute Error (MAE) and R-Squared for Regressions"),
            ("TPU", "REQUIRED_FOR", "Large-Scale Distributed Deep Learning Model Training (Transformer/LLM)"),
            ("GPU", "SUITED_FOR", "Parallel Matrix Computations and Model Training/Inference"),
            ("CPU", "SUITED_FOR", "General Purpose Compute and Lightweight Model Inference"),
            ("Feature Attribution Drift", "MONITORED_BY", "Vertex AI Model Monitoring"),
            ("Training-Serving Skew", "MONITORED_BY", "Vertex AI Model Monitoring")
        ]
        for service, rel, metric in pmle_metrics:
            session.run(f"""
                MERGE (s:Service {{name: $service}})
                MERGE (m:EvaluationMetric {{description: $metric}})
                MERGE (s)-[:{rel}]->(m)
            """, service=service, metric=metric)

        # Model Development Options grouping
        groupings = [
            ("Custom Containers", "Model Training Method"),
            ("AutoML", "Model Training Method"),
            ("BigQuery ML", "Model Training Method")
        ]
        for service, parent in groupings:
            session.run("""
                MERGE (s:Service {name: $service})
                MERGE (p:HierarchyNode {name: $parent})
                MERGE (s)-[:CLASSIFIED_UNDER]->(p)
            """, service=service, parent=parent)

        # ML Lifecycle vs GenAI
        lifecycles = [
            ("Vertex AI Pipelines", "MLOps Lifecycle"),
            ("Feature Store", "MLOps Lifecycle"),
            ("Experiment Tracking", "MLOps Lifecycle"),
            ("Model Garden", "Generative AI Platform"),
            ("Vertex AI Agent Builder", "Generative AI Platform"),
            ("Vector Search", "Generative AI Platform")
        ]
        for service, parent in lifecycles:
            session.run("""
                MERGE (s:Service {name: $service})
                MERGE (p:HierarchyNode {name: $parent})
                MERGE (s)-[:PART_OF]->(p)
            """, service=service, parent=parent)

        # --- Comparative Trade-offs (Tier 4) ---
        comparisons = [
            ("Cloud Spanner", "Cloud SQL", "Global write scalability with strong SQL consistency", 
             "Cloud Spanner scales horizontally across regions while maintaining transactional consistency, whereas Cloud SQL is limited to a single primary instance for writes."),
            ("Cloud Run", "Google Kubernetes Engine", "Web applications and microservices with minimal operational overhead", 
             "Cloud Run is serverless and scales to zero, whereas GKE requires managing node pools, cluster control planes, and Kubernetes workloads."),
            ("BigQuery", "Cloud SQL", "Analytical queries over petabyte-scale historical datasets", 
             "BigQuery is a serverless column-store data warehouse optimized for OLAP, whereas Cloud SQL is optimized for OLTP transaction workloads."),
            ("BigQuery", "Cloud Spanner", "Analytical queries over petabyte-scale historical datasets", 
             "BigQuery is serverless and optimized for OLAP analytics, whereas Cloud Spanner is optimized for high-throughput OLTP transactions."),
            ("Cloud Storage", "Cloud SQL", "Storing massive amounts of unstructured files cost-effectively", 
             "Cloud Storage is an object store with multiple tiered archival classes, whereas databases like Cloud SQL are far more expensive for large binary file storage."),
            ("Vertex AI", "Compute Engine", "Unified machine learning platform with built-in MLOps pipelines", 
             "Vertex AI manages the entire ML lifecycle (training, deploying, monitoring) out-of-the-box, whereas manual VM setups require building pipelines from scratch."),
            ("AutoML", "Vertex AI", "Training custom models without writing machine learning code", 
             "AutoML automatically handles feature engineering and model selection, whereas custom training on Vertex AI requires writing code and selecting hyperparameters."),
            ("Cloud VPN", "Cloud Interconnect", "Establishing secure network connectivity quickly and at low cost", 
             "Cloud VPN runs over public internet and configures in minutes, whereas Cloud Interconnect requires dedicated physical links and high monthly costs."),
            ("VPC Peering", "Cloud VPN", "Connecting two virtual networks with low latency and no gateway bottlenecks", 
             "VPC Peering routes traffic directly using internal IPs, whereas VPN tunnels require gateways that cap bandwidth and add encryption overhead.")
        ]
        for s1, s2, cond, reason in comparisons:
            session.run("""
                MERGE (sa:Service {name: $s1})
                MERGE (sb:Service {name: $s2})
                MERGE (sa)-[r:PREFERABLE_OVER]->(sb)
                SET r.condition = $cond, r.reason = $reason
            """, s1=s1, s2=s2, cond=cond, reason=reason)

        # --- Prerequisites (Tier 7) ---
        prereqs = [
            ("Google Kubernetes Engine", "Compute Engine"),
            ("Google Kubernetes Engine", "Artifact Registry"),
            ("Cloud Run", "Artifact Registry"),
            ("Looker", "BigQuery"),
            ("Looker Studio", "BigQuery"),
            ("Dataflow", "Cloud Storage"),
            ("Dataproc", "Compute Engine"),
            ("Vertex AI", "BigQuery"),
            ("AutoML", "Cloud Storage"),
            ("Apigee", "Cloud Run"),
            ("Anthos", "Google Kubernetes Engine"),
            ("Shared VPC", "VPC")
        ]
        for s1, s2 in prereqs:
            session.run("""
                MERGE (sa:Service {name: $s1})
                MERGE (sb:Service {name: $s2})
                MERGE (sa)-[:REQUIRES_PREREQUISITE_KNOWLEDGE_OF]->(sb)
            """, s1=s1, s2=s2)

        # --- Anti-Patterns (Tier 5) ---
        antipatterns = [
            ("Cloud Storage", "Storing highly active daily transactional state", "Cloud SQL"),
            ("Compute Engine", "Hardcoding database credentials inside application virtual machine images", "Secret Manager"),
            ("IAM", "Assigning broad primitive Owner roles directly to developers in production", "IAM Custom Roles"),
            ("Cloud VPN", "Using standard VPN tunnels for transferring petabyte-scale migration data under short deadlines", "Transfer Appliance"),
            ("Vertex AI", "Manually copying model artifacts between projects for promotion to production", "Vertex AI Model Registry"),
            ("BigQuery", "Querying massive tables without partitioning or clustering, causing high scan costs", "BigQuery Partitioning & Clustering"),
            ("Compute Engine", "Running steady-state non-spiky workloads on expensive manual virtual machines without autoscaling", "Compute Engine Managed Instance Groups"),
            ("Google Kubernetes Engine", "Storing container configuration variables inside application code rather than mounting environment details", "ConfigMaps & Secrets"),
            ("VPC", "Exposing VM internal service ports directly to the public internet for administrator access", "Identity-Aware Proxy")
        ]
        for service, desc, resolution in antipatterns:
            session.run("""
                MERGE (s:Service {name: $service})
                MERGE (ap:AntiPattern {description: $desc})
                MERGE (res:Service {name: $resolution})
                MERGE (s)-[:COMMON_PITFALL]->(ap)
                MERGE (ap)-[:RESOLVED_BY]->(res)
            """, service=service, desc=desc, resolution=resolution)
            
    print("Third-tier blueprint leaf nodes and relationships successfully seeded.")



