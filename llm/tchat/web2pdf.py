import os
from bs4 import BeautifulSoup
import requests
PDF_OUTPUT_DIR = '/Users/maxxoto/Study/Personal/Machine-Learning/stephen-langchain/tchat'
WKHTMLTOPDF_PATH = '/path/to/wkhtmltopdf'
def scrape_and_save_pdf(url): # Send a GET request to the URL
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'lxml')
        # Create a temporary HTML file for wkhtmltopdf
        html_file_path = f'{PDF_OUTPUT_DIR}/temp_{url}.html'
        with open(html_file_path, 'w') as f:
            f.write(str(soup))
        # Use pdfkit to
        # the HTML to PDF
        command = f'wkhtmltopdf --load-error-handling ignore {WKHTMLTOPDF_PATH} {html_file_path} {PDF_OUTPUT_DIR}/{url}.pdf'
        os.system(command)
def main():
    urls_to_scrape = [ 'https://www.example.com/page1', 'https://www.example.com/page2',]
    for url in urls_to_scrape:
        scrape_and_save_pdf(url)
        print(f'Saved {url} to PDF')

main()
