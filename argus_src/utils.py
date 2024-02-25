import requests
from bs4 import BeautifulSoup
#from cleantext import clean  # requirements: pip install clean-text, unidecode
from markdownify import markdownify as md
from pypdf import PdfReader

import time
from pathlib import Path


def crawl_website(url: str) -> str :
    DEFAULT_TARGET_CONTENT = ['article', 'div', 'main', 'p']

    try:
        print(f'- Crawling: {url}')
        response = requests.get(url)
    except requests.exceptions.RequestException as e:
        print(f'❌ Request error for {url}: {e}')
        return None
    
    content_type = response.headers.get('Content-Type', '')

    if 'text/html' in content_type:
        strip_elements = ['a']

        # Create BS4 instance for parsing
        soup = BeautifulSoup(response.text, 'html.parser')

        # Strip unwanted tags
        for script in soup(['script', 'style']):
            script.decompose()

        max_text_length = 0
        main_content = ""
        for tag in soup.find_all(DEFAULT_TARGET_CONTENT):
            text_length = len(tag.get_text())
            if text_length > max_text_length:
                max_text_length = text_length
                main_content = tag

        content = str(main_content)

        # Return if text > 0
        if len(content) == 0:
            return None
        
        # Parse markdown
        output = md(
            content,
            keep_inline_images_in=['td', 'th', 'a', 'figure'],
            strip=strip_elements
        )
    
        print(f'✅ Success')

        return output

    elif 'application/pdf'in content_type:
        
        try:
            ext = '.pdf'
            # Create temp folder for pdf files
            Path("tmp").mkdir(parents=True, exist_ok=True)

            # Get timestamp as filename and save pdf file
            timestamp = int(time.time())
            pdf_path = f"tmp/{timestamp}{ext}"

            chunk_size = 2000  # Load 2000 byte at once
            with open(pdf_path, 'wb') as fd:
                for chunk in response.iter_content(chunk_size):
                    fd.write(chunk)

            # Read in saved PDF
            pdfreader = PdfReader(pdf_path)
            pdftext = ""
            for page in pdfreader.pages:
                pdftext += page.extract_text() + "\n"

            print(f'✅ Successful read PDF file {pdf_path}')
            return pdftext
        
        except Exception as e:
            print(f'❌ Error Reading PDF file: {e}')
            return None
    
    elif 'text/html' not in content_type:
        print(f'❌ Content not text/html for {url}')
        return None
    