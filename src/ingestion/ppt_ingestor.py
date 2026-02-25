"""
PPT Ingestor — Handles native .pptx files.

If the PPT has already been converted to PDF, use pdf_ingestor instead.
This module handles raw .pptx files using python-pptx.

Strategy:
- Extract text from each slide's text frames
- Extract speaker notes separately (often contain key explanations)
- Each slide becomes one or more chunks
- Slide titles are prepended to all chunks from that slide for context
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_chunks_from_pptx(filepath: str) -> list[dict]:
    """
    Extract text chunks from a .pptx file.
    Returns a list of dicts: {id, text, metadata}
    """
    try:
        from pptx import Presentation
        from pptx.util import Pt
    except ImportError:
        raise ImportError("python-pptx not installed. Run: pip install python-pptx")

    prs = Presentation(filepath)
    filename = Path(filepath).name
    chunks = []

    for slide_num, slide in enumerate(prs.slides, start=1):
        slide_texts = []
        slide_title = ""
        notes_text = ""

        # ── Extract slide content ───────────────────────────────────────────
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for para in shape.text_frame.paragraphs:
                line = para.text.strip()
                if not line:
                    continue
                # Heuristic: large font or placeholder name = title
                if shape.name.lower().startswith("title") and not slide_title:
                    slide_title = line
                else:
                    slide_texts.append(line)

        # ── Extract speaker notes ───────────────────────────────────────────
        if slide.has_notes_slide:
            notes_frame = slide.notes_slide.notes_text_frame
            if notes_frame:
                notes_text = notes_frame.text.strip()

        # Build the main slide chunk
        slide_body = "\n".join(slide_texts)
        if slide_title or slide_body:
            chunk_text = f"Slide {slide_num}"
            if slide_title:
                chunk_text += f" — {slide_title}"
            if slide_body:
                chunk_text += f":\n{slide_body}"

            if len(chunk_text.strip()) > 40:
                chunks.append({
                    "id": f"{filename}_slide{slide_num}_content",
                    "text": chunk_text,
                    "metadata": {
                        "source": filename,
                        "source_type": "presentation",
                        "slide": slide_num,
                        "section": "content",
                        "title": slide_title,
                    },
                })

        # Notes as a separate chunk (often contain the real explanation)
        if notes_text and len(notes_text) > 40:
            notes_chunk = f"Speaker notes for slide {slide_num}"
            if slide_title:
                notes_chunk += f" ({slide_title})"
            notes_chunk += f":\n{notes_text}"

            chunks.append({
                "id": f"{filename}_slide{slide_num}_notes",
                "text": notes_chunk,
                "metadata": {
                    "source": filename,
                    "source_type": "presentation",
                    "slide": slide_num,
                    "section": "notes",
                    "title": slide_title,
                },
            })

    logger.info(f"Extracted {len(chunks)} chunks from {filename} ({len(prs.slides)} slides)")
    return chunks


def extract_all_pptx(data_dir: str) -> list[dict]:
    """Extract chunks from all .pptx files in the data directory."""
    pptx_files = list(Path(data_dir).glob("*.pptx"))
    all_chunks = []
    for pptx_path in pptx_files:
        try:
            chunks = extract_chunks_from_pptx(str(pptx_path))
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Failed to parse {pptx_path}: {e}")
    logger.info(f"Total PPTX chunks extracted: {len(all_chunks)}")
    return all_chunks
