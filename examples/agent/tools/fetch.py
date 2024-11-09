import logging
from docling.document_converter import DocumentConverter


def fetch(url: str) -> str:
    '''
        Fetch a document at the provided URL and convert it to Markdown.
    '''

    logging.debug(f'[fetch] Fetching %s', url)
    converter = DocumentConverter()
    result = converter.convert(url)
    return result.document.export_to_markdown()
