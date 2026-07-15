# GCP Exam Knowledge Graph & GraphRAG Chatbot

An automated learning assistant designed to index official Google Cloud Architecture Framework pillars and Professional Cloud Architect (PCA) business case studies into a Neo4j Graph database, enabling a localized Streamlit Chat interface using Gemini 1.5 Flash to provide grounded, scenario-based architecture guidance.

This project uses native graph-traversal vector queries rather than flat vector databases, ensuring the LLM understands contextual design patterns and system thresholds.

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
To run the sync script which scrapes Google Cloud Architecture Framework one-page docs, downloads the 5 official case study PDFs, chunks them, generates text embeddings using Gemini (`text-embedding-004`), builds sequential `NEXT` relationships, and creates the Neo4j vector index:
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
