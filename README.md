# GCP Exam Knowledge Graph & GraphRAG Chatbot

An automated learning assistant designed to index official Google Cloud documentation, Architecture Framework pillars, and business case studies into a Neo4j Graph database, enabling a localized Streamlit Chat interface using Gemini 1.5 Flash to provide grounded, scenario-based architecture guidance across all main Google Cloud certification paths.

This project uses native graph-traversal vector queries rather than flat vector databases, ensuring the LLM understands contextual design patterns and system thresholds.

---

## Supported Certifications & Blueprint Paths

The system seeds exam blueprint metadata and maps documentation to **six distinct Google Cloud certifications**:

| Tier | Certification Name | Exam Code | Target Documentation Scraped |
|---|---|---|---|
| **Foundational** | [Cloud Digital Leader](https://cloud.google.com/learn/certifications/cloud-digital-leader) | `cdl` | Official GCP Docs |
| **Associate** | [Associate Cloud Engineer](https://cloud.google.com/learn/certifications/associate-cloud-engineer) | `ace` | Official GCP Docs |
| **Associate** | [Associate Data Practitioner](https://cloud.google.com/learn/certifications/associate-data-practitioner) | `adp` | BigQuery Docs |
| **Professional** | [Professional Cloud Architect](https://cloud.google.com/learn/certifications/cloud-architect) | `pca` | GCP Architecture Framework, Case Study PDFs (Altostrat, Cymbal Retail, EHR Healthcare, Knightmotives) |
| **Professional** | [Professional Data Engineer](https://cloud.google.com/learn/certifications/data-engineer) | `pde` | BigQuery & Pub/Sub Docs |
| **Professional** | [Professional Machine Learning Engineer](https://cloud.google.com/learn/certifications/machine-learning-engineer) | `pmle` | Vertex AI Docs |

---

## System Architecture

```
gcp_exam_graphrag/
├── .env/
│   └── .env.example            # Environment configurations template
├── .github/
│   └── workflows/
│       └── update_pipeline.yml  # Automated cron sync loop
├── data/
│   └── fetch_docs.py            # Local & CI script to scrape docs & PDFs
├── utils/
│   ├── __init__.py
│   └── graph_ops.py             # Schema definitions and Cypher queries
├── Dockerfile                   # Lean container definition for Cloud Run
├── pyproject.toml              # UV dependency management
├── app.py                      # Interactive Streamlit Chat UI
└── sync_pipeline.py            # Ingestion and indexing orchestrator
```

---

## GraphRAG & Ingestion Optimization Features

To ensure production-grade robustness and search accuracy, the ingestion pipeline implements three key strategies:

1. **Robust Rate-Limiting & Backoff**: Wrapping embedding generation API calls to Google AI Studio in an exponential backoff retry loop to gracefully handle free-tier rate limits (`ResourceExhausted` errors).
2. **Shared Service Taxonomy**: Chunks across all certifications are mapped to single global service nodes via a `[:DISCUSSES]` relationship. This merges documentation insights across exams, allowing GraphRAG to retrieve valuable cross-exam security and architectural insights.
3. **Sliding-Window Chunking**: Splits large document sections and PDF pages into semantic chunks of 1000 characters with a 200-character overlap, guaranteeing consistent text lengths for optimal similarity matching.

---

## Knowledge Graph Multi-Tier Architecture

To support deep architectural reasoning and highly granular diagnostics, the knowledge graph is designed with a multi-tier structure:

1. **Tier 4: Comparative Decision Boundaries & Trade-Offs**
   Helps the chatbot resolve trade-off questions (e.g., *"When should I use Cloud Spanner vs. Cloud SQL?"*) by storing explicit conditional boundaries.
   * **Relationship Pattern**: `(Service A) -[:PREFERABLE_OVER {condition: "...", reason: "..."}]-> (Service B)`
   * **Example**: `(Cloud Spanner) -[:PREFERABLE_OVER {condition: "Global write scale & strong consistency", reason: "Scales horizontally across regions while maintaining transactional consistency"}]-> (Cloud SQL)`

2. **Tier 5: Common Anti-Patterns & Pitfalls**
   Assists in identifying architectural misconfigurations (common on Professional exams) and suggesting the correct resolutions.
   * **Relationship Pattern**: `(Service) -[:COMMON_PITFALL]-> (AntiPattern) -[:RESOLVED_BY]-> (Service/BestPractice)`
   * **Example**: `(Cloud Storage) -[:COMMON_PITFALL]-> (Storing highly active daily transactional state) -[:RESOLVED_BY]-> (Cloud SQL)`

3. **Bayesian Belief Network (BBN) Sub-Service Concepts**
   Upgrades the diagnostic granularity of the adaptive practice quiz beyond services to specific sub-service concepts.
   * **Structure**: `Domain -> Service -> SubConcept -> Question`
   * **Example**: `Data Storage & Querying -> BigQuery -> Partitioning & Clustering -> BQ_Optimization_Question`
   * This allows the Active Learning engine to pinpoint the exact feature of a service a student struggles with.

4. **Tier 7: Core Prerequisite Learning Paths**
   Maps learning dependencies between services to construct logical educational tracks.
   * **Relationship Pattern**: `(Service A) -[:REQUIRES_PREREQUISITE_KNOWLEDGE_OF]-> (Service B)`
   * **Example**: `(Google Kubernetes Engine) -[:REQUIRES_PREREQUISITE_KNOWLEDGE_OF]-> (Compute Engine)`

---

## Technical Stack & Costs

* **Frontend UI:** Streamlit (Runs completely free on your local machine)
* **Knowledge Graph & Vector Index:** Neo4j Aura Free Tier (Supports up to 200,000 nodes, hosting the entire GCP service taxonomy)
* **LLM & Embedding Generator:** Google AI Studio (Gemini 1.5 Flash) (The free tier handles extraction, embedding generation, and response generation)
* **Package Management:** `uv` (For near-instant dependency compilation and deterministic environment locking)
* **Automation:** GitHub Actions (Weekly cron pipeline)
* **Hosting:** Google Cloud Run (Containerized serverless hosting scaling to zero, remaining within the free tier allowance)

---

## Step-by-Step Setup Guides

### 1. Neo4j Aura Instance Provisioning

To set up a free Neo4j Aura instance:
1. Navigate to [Neo4j Aura Console](https://aura.neo4j.io/) and register for a free account.
2. Select **Create Instance**.
3. Choose the **AuraDB Free** tier.
4. Once created, a PDF/text file containing your credentials will automatically download. This file includes:
   * `NEO4J_URI` (looks like `neo4j+s://xxxxxx.databases.neo4j.io`)
   * `NEO4J_USERNAME` (default is `neo4j`)
   * `NEO4J_PASSWORD` (automatically generated string)
5. Keep these credentials secure; you will paste them into your environment configuration.

### 2. Google AI Studio (Gemini API) Key Setup

1. Go to [Google AI Studio](https://aistudio.google.com/).
2. Log in with your Google account.
3. Click **Get API key** in the sidebar.
4. Click **Create API key** and select a project or create a new one.
5. Copy the generated key.

### 3. Local Environment Configuration

1. Locate the `.env/` folder in the root directory.
2. Copy `.env/.env.example` to create `.env/.env.development`:
   ```bash
   cp .env/.env.example .env/.env.development
   ```
3. Edit `.env/.env.development` and replace the placeholder values with your Neo4j Aura credentials and Gemini API Key:
   ```env
   NEO4J_URI=neo4j+s://<YOUR_INSTANCE_ID>.databases.neo4j.io
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=<YOUR_PASSWORD>
   GEMINI_API_KEY=<YOUR_GEMINI_API_KEY>
   ```

---

## Local Execution Instructions

### 1. Prerequisites
Ensure you have `uv` installed. If you do not have it, install it using the official script:
* **macOS/Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
* **Windows:**
  ```powershell
  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

### 2. Synchronization Pipeline Run
To run the sync script which scrapes Google Cloud Architecture Framework documentation and case study PDFs, splits them using sliding-window chunking (1000 character size / 200 character overlap), generates text embeddings using Gemini (`text-embedding-004`) with rate-limiting retries, links chunks to global service nodes via `[:DISCUSSES]`, builds sequential `NEXT` relationships, and creates the Neo4j vector index:
```bash
uv run sync_pipeline.py
```

### 3. Launch Streamlit UI
To start the chat application locally:
```bash
uv run streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Serverless Deployment Sequence

To deploy the Streamlit chat application to Google Cloud Run:

1. **Authenticate with Google Cloud:**
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Build the container image via Cloud Build:**
   ```bash
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/gcp-exam-bot:latest
   ```

3. **Deploy to Cloud Run:**
   Configure max-instances to 1 and scale limits to ensure the application stays fully within GCP's 2-million-free-requests-per-month allowance:
   ```bash
   gcloud run deploy gcp-exam-bot \
       --image gcr.io/YOUR_PROJECT_ID/gcp-exam-bot:latest \
       --platform managed \
       --region us-central1 \
       --allow-unauthenticated \
       --max-instances 1 \
       --memory 1Gi \
       --set-secrets="NEO4J_URI=NEO4J_URI:latest,NEO4J_PASSWORD=NEO4J_PASSWORD:latest,GEMINI_API_KEY=GEMINI_API_KEY:latest"
   ```
