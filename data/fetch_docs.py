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

def parse_html_framework(url, source_name=None):
    """Scrapes the Google Cloud HTML documentation pages, stripping DevSite wrappers."""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching docs from {url}: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Strip DevSite specific navigation, header, footer, and feedback elements
    for el in soup.find_all([
        'nav', 'header', 'footer', 'script', 'style',
        'devsite-header', 'devsite-footer', 'devsite-footer-utility',
        'devsite-book-nav', 'devsite-nav', 'devsite-rating-section',
        'devsite-feedback', 'devsite-select', 'devsite-page-rating'
    ]):
        el.decompose()
        
    import re
    # Strip feedback/widgets by class or ID patterns
    for el in soup.find_all(class_=re.compile(r'devsite-nav|devsite-metadata|devsite-feedback|devsite-rating|feedback-widget|header-wrapper|footer-wrapper')):
        el.decompose()
    
    # Dynamically resolve source name/title if not provided
    if not source_name:
        if soup.title and soup.title.string:
            source_name = soup.title.string.strip()
            # Clean common Google Cloud page title suffixes
            source_name = re.sub(r'\s*\|\s*Google\s*Cloud.*$', '', source_name, flags=re.IGNORECASE)
        else:
            source_name = url.split('/')[-1].replace('-', ' ').title()
            if not source_name:
                source_name = "GCP Documentation"
                
    slug = re.sub(r'[^a-z0-9_]', '', source_name.lower().replace(' ', '_'))
    if not slug:
        slug = "doc"
        
    # Locate main content container inside DevSite layout
    content_area = (
        soup.find('div', class_='devsite-article-body') 
        or soup.find('article') 
        or soup.find('main') 
        or soup
    )
    
    chunks = []
    chunk_idx = 0
    
    headings = content_area.find_all(['h2', 'h3'])
    if not headings:
        paragraphs = content_area.find_all('p')
        text_blocks = [p.get_text(separator=' ').strip() for p in paragraphs if len(p.get_text().strip()) > 30]
        full_text = "\n".join(text_blocks)
        if full_text:
            chunks.append({
                "source": source_name,
                "chunk_id": f"{slug}_full",
                "title": f"{source_name} Overview",
                "text": full_text
            })
    else:
        for heading in headings:
            title = heading.get_text().strip()
            text_blocks = []
            curr = heading.next_sibling
            while curr and curr not in headings:
                if hasattr(curr, 'get_text'):
                    txt = curr.get_text(separator=' ').strip()
                    if txt:
                        text_blocks.append(txt)
                curr = curr.next_sibling
                
            combined_text = "\n".join(text_blocks).strip()
            if len(combined_text) > 100:
                chunks.append({
                    "source": source_name,
                    "chunk_id": f"{slug}_sec_{chunk_idx}",
                    "title": f"{source_name} - {title}",
                    "text": f"[{title}]\n{combined_text}"
                })
                chunk_idx += 1
                
    if not chunks:
        text = content_area.get_text(separator='\n').strip()
        text = re.sub(r'\n+', '\n', text)
        if len(text) > 100:
            chunks.append({
                "source": source_name,
                "chunk_id": f"{slug}_fallback",
                "title": f"{source_name} General",
                "text": text[:20000]
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
