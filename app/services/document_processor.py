"""
Document Processor Service.

Wraps docling functionality to convert and chunk PDF documents into smaller text
segments suitable for vector embeddings.
"""
from pathlib import Path

from docling.chunking import HybridChunker
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from loguru import logger

class DocumentProcessor:
    """Handles parsing, OCR (disabled by default), and chunking of documents."""
    def __init__(self):
        """Initialize the Docling DocumentConverter and HybridChunker."""
        pipeline_options = PdfPipelineOptions(do_ocr=False)
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=2, device=AcceleratorDevice.CPU
        )
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        self.chunker = HybridChunker(max_tokens=512)

    def process_document(self, file_path: str) -> list[dict]:
        """
        Process a PDF document into a list of chunk metadata dicts.
        
        Args:
            file_path: Absolute path to the local PDF file.
            
        Returns:
            A list of dictionaries, each containing chunk 'text', 'source', and 'page_number'.
        """
        result = self.converter.convert(file_path)
        doc = result.document
        chunk_iter = self.chunker.chunk(doc)

        chunks = []
        source_name = Path(file_path).name

        for chunk in chunk_iter:
            meta = {"text": chunk.text, "source": source_name}
            if hasattr(chunk, "meta") and hasattr(chunk.meta, "doc_items"):
                items = chunk.meta.doc_items
                if items and hasattr(items[0], "prov") and items[0].prov:
                    meta["page_number"] = items[0].prov[0].page_no
            chunks.append(meta)
        logger.info("Processed {} chunks from {}", len(chunks), file_path)
        return chunks

                