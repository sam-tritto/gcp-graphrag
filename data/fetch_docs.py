import io
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

# The target GCP study documents and official business case studies
TARGET_DOCS = {
    "framework": "https://cloud.google.com/architecture/framework",
    "case_studies": {
        "altostrat": "https://services.google.com/fh/files/misc/v6.1_pca_altostrat_media_case_study_english.pdf",
        "cymbal_retail": "https://services.google.com/fh/files/misc/v6.1_pca_cymbal_retail_case_study_english.pdf",
        "ehr_healthcare": "https://services.google.com/fh/files/misc/v6.1_pca_ehr_healthcare_case_study_english.pdf",
        "knightmotives": "https://services.google.com/fh/files/misc/v6.1_pca_knightmotives_automotive_case_study_english.pdf",
    }
}

def parse_html_framework(url):
    """Scrapes the Google Cloud Architecture Framework one-page documentation."""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching Architecture Framework from {url}: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Locate article content. Streamlined to avoid header/footer/side navigation noise.
    articles = soup.find_all('article') or soup.find_all('main') or [soup]
    
    chunks = []
    chunk_idx = 0
    for article in articles:
        # Divide article into sections or paragraphs to make semantic chunks
        sections = article.find_all(['section', 'h2', 'h3'])
        if not sections:
            # Fallback to simple paragraph division if no logical sections
            paragraphs = article.find_all('p')
            temp_text = []
            for p in paragraphs:
                text = p.get_text(separator=' ').strip()
                if len(text) > 100:
                    temp_text.append(text)
                if len(' '.join(temp_text)) > 1500:
                    chunks.append({
                        "source": "Architecture Framework",
                        "chunk_id": f"fw_p_{chunk_idx}",
                        "title": "Architecture Framework Overview",
                        "text": ' '.join(temp_text)
                    })
                    temp_text = []
                    chunk_idx += 1
            if temp_text:
                chunks.append({
                    "source": "Architecture Framework",
                    "chunk_id": f"fw_p_{chunk_idx}",
                    "title": "Architecture Framework Overview",
                    "text": ' '.join(temp_text)
                })
        else:
            for section in sections:
                # Find sibling text until next heading or section
                title = section.get_text().strip()
                text_blocks = []
                
                # Gather content under this heading/section
                for sibling in section.next_siblings:
                    if sibling.name in ['h1', 'h2', 'h3', 'section']:
                        break
                    if sibling.name in ['p', 'ul', 'ol', 'div']:
                        txt = sibling.get_text(separator=' ').strip()
                        if txt:
                            text_blocks.append(txt)
                            
                combined_text = "\n".join(text_blocks).strip()
                if len(combined_text) > 150:
                    chunks.append({
                        "source": "Architecture Framework",
                        "chunk_id": f"fw_sec_{chunk_idx}",
                        "title": title,
                        "text": f"[{title}]\n{combined_text}"
                    })
                    chunk_idx += 1
                    
    # If no chunks were extracted, fallback to a single block
    if not chunks:
        text = soup.get_text(separator=' ').strip()
        chunks.append({
            "source": "Architecture Framework",
            "chunk_id": "fw_fallback",
            "title": "Architecture Framework (Full Scrape)",
            "text": text[:10000] # Limit size of full text dump
        })
        
    return chunks

def parse_pdf_case_study(name, url):
    """Downloads and extracts text chunks page-by-page from a case study PDF."""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching PDF case study {name} from {url}: {e}")
        return []

    chunks = []
    try:
        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)
        
        # Read each page and treat it as a semantic chunk
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and len(text.strip()) > 100:
                chunks.append({
                    "source": f"Case Study: {name.replace('_', ' ').title()}",
                    "chunk_id": f"cs_{name}_{page_idx}",
                    "title": f"{name.replace('_', ' ').title()} - Page {page_idx + 1}",
                    "text": text.strip()
                })
    except Exception as e:
        print(f"Error parsing PDF case study {name}: {e}")
        
    return chunks

def fetch_all_chunks():
    """Fetches and compiles all raw text chunks from the sources."""
    all_chunks = []
    
    # 1. Fetch framework
    print("Fetching and parsing Google Cloud Architecture Framework...")
    fw_chunks = parse_html_framework(TARGET_DOCS["framework"])
    print(f"Extracted {len(fw_chunks)} chunks from Architecture Framework.")
    all_chunks.extend(fw_chunks)
    
    # 2. Fetch all case studies
    for name, url in TARGET_DOCS["case_studies"].items():
        print(f"Fetching and parsing case study: {name}...")
        cs_chunks = parse_pdf_case_study(name, url)
        print(f"Extracted {len(cs_chunks)} chunks from {name} case study.")
        all_chunks.extend(cs_chunks)
        
    return all_chunks
