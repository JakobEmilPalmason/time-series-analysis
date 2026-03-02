"""Convert all PDF files in the 03-assignments directory to Markdown."""

import sys
from pathlib import Path

import pymupdf  # PyMuPDF


def pdf_to_markdown(pdf_path: Path) -> str:
    """Extract text from a PDF and return it as markdown."""
    doc = pymupdf.open(pdf_path)
    sections = []

    for page_num, page in enumerate(doc, 1):
        text = page.get_text("text")
        if text.strip():
            sections.append(f"<!-- Page {page_num} -->\n\n{text.strip()}")

    doc.close()
    return "\n\n---\n\n".join(sections)


def convert_all_pdfs(base_dir: Path, overwrite: bool = False):
    """Find all PDFs under 03-assignments and convert each to a .md file alongside it."""
    assignments_dir = base_dir / "03-assignments"
    pdf_files = sorted(assignments_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in 03-assignments/.")
        return

    print(f"Found {len(pdf_files)} PDF files.\n")

    for pdf_path in pdf_files:
        md_path = pdf_path.with_suffix(".md")

        if md_path.exists() and not overwrite:
            print(f"  SKIP (exists): {md_path.relative_to(base_dir)}")
            continue

        print(f"  Converting: {pdf_path.relative_to(base_dir)}")
        try:
            markdown_text = pdf_to_markdown(pdf_path)
            md_path.write_text(markdown_text, encoding="utf-8")
            print(f"       -> {md_path.relative_to(base_dir)}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone.")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    overwrite = "--overwrite" in sys.argv
    convert_all_pdfs(repo_root, overwrite=overwrite)
