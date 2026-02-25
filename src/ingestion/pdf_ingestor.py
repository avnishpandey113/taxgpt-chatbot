"""
PDF Ingestor — Financial reports and converted PPT slides.

Strategy:
- Page-level extraction via PyMuPDF (fitz)
- Text is split into overlapping chunks (512 tokens, 50 overlap)
- Very short pages (< 100 chars) are merged with adjacent pages
- Metadata tracks: source file, page number, chunk index
"""

import os
import re
import logging
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


def _split_into_chunks(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping word-level chunks.
    Each chunk is ~chunk_size words with overlap words shared with the next chunk.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 80:  # skip near-empty chunks
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def _clean_text(text: str) -> str:
    """Remove excessive whitespace, form feeds, and control characters."""
    text = re.sub(r'\f', '\n', text)
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def extract_chunks_from_pdf(filepath: str, chunk_size: int = 512, overlap: int = 50) -> list[dict]:
    """
    Extract text chunks from a PDF file.
    Returns a list of dicts: {id, text, metadata}
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")

    doc = fitz.open(filepath)
    filename = Path(filepath).name
    source_type = _detect_source_type(filename)
    total_pages = len(doc)

    chunks = []
    page_buffer = ""

    for page_num in range(total_pages):
        page = doc[page_num]
        raw_text = page.get_text("text")
        page_text = _clean_text(raw_text)

        if len(page_text) < 100:
            page_buffer += f"\n{page_text}"
            continue

        if page_buffer:
            page_text = page_buffer + "\n" + page_text
            page_buffer = ""

        page_chunks = _split_into_chunks(page_text, chunk_size, overlap)

        for chunk_idx, chunk_text in enumerate(page_chunks):
            chunk_id = f"{filename}_p{page_num}_c{chunk_idx}"
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "source": filename,
                    "source_type": source_type,
                    "page": page_num + 1,
                    "chunk_index": chunk_idx,
                    "total_pages": total_pages,
                },
            })

    # Flush remaining buffer
    if page_buffer.strip():
        for chunk_idx, chunk_text in enumerate(_split_into_chunks(page_buffer, chunk_size, overlap)):
            chunks.append({
                "id": f"{filename}_buffer_c{chunk_idx}",
                "text": chunk_text,
                "metadata": {
                    "source": filename,
                    "source_type": source_type,
                    "page": "buffer",
                    "chunk_index": chunk_idx,
                    "total_pages": total_pages,
                },
            })

    doc.close()
    logger.info(f"Extracted {len(chunks)} chunks from {filename} ({total_pages} pages)")
    return chunks


def _detect_source_type(filename: str) -> str:
    """Heuristic: classify PDFs by filename."""
    name = filename.lower()
    if "presentation" in name or "ppt" in name or "slide" in name:
        return "presentation"
    if "1040" in name or "form" in name or "instruction" in name:
        return "irs_form"
    if "usc" in name or "statute" in name or "law" in name:
        return "tax_law"
    return "financial_report"


def extract_all_pdfs(data_dir: str, chunk_size: int = 512, overlap: int = 50) -> list[dict]:
    """Extract chunks from all PDF files in the data directory."""
    pdf_files = list(Path(data_dir).glob("*.pdf"))
    all_chunks = []
    for pdf_path in pdf_files:
        try:
            chunks = extract_chunks_from_pdf(str(pdf_path), chunk_size, overlap)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Failed to parse {pdf_path}: {e}")
    logger.info(f"Total PDF chunks extracted: {len(all_chunks)}")
    return all_chunks
