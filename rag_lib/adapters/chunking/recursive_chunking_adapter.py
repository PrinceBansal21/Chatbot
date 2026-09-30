from typing import Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_lib.interfaces.chunking_manager import ChunkingManager, Chunk
from rag_lib.utils.page_markers import extract_slide_type_map
import re

class RecursiveChunkingAdapter(ChunkingManager):
    """Chunking strategy using Langchain's RecursiveCharacterTextSplitter."""

    def __init__(self, max_tokens: int = 1000, overlap_tokens: int = 200, **kwargs):
        # Note: RecursiveCharacterTextSplitter uses characters by default, but can use token counters.
        # For simplicity, we'll assume approx 4 chars per token if no tokenizer is provided,
        # or we can use the chunk_size as characters directly if the user intends that.
        # However, the interface says 'max_tokens'. 
        # Let's assume 1 token ~= 4 chars for the default character splitter.
        self.chunk_size = max_tokens * 4
        self.chunk_overlap = overlap_tokens * 4
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk(self, file_id: str, markdown: str, base_metadata: Dict[str, str]) -> List[Chunk]:
        # Captured from the raw markdown before _split_by_page strips the
        # markers — PPTX-only, maps page_number -> "title" | "content".
        slide_type_map = extract_slide_type_map(markdown)
        page_segments = self._split_by_page(markdown)
        chunks: List[Chunk] = []
        chunk_counter = 0
        _heading_re = re.compile(r"^#{1,6}\s+(.+)", re.MULTILINE)

        for content, page_num in page_segments:
            docs = self.splitter.create_documents([content])
            for doc in docs:
                chunk_id = f"{file_id}-{chunk_counter:05d}"
                meta = base_metadata.copy()
                if page_num:
                    meta["page_number"] = str(page_num)
                    if page_num in slide_type_map:
                        meta["slide_type"] = slide_type_map[page_num]
                
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        file_id=file_id,
                        content=doc.page_content,
                        order=chunk_counter,
                        metadata=meta,
                    )
                )
                chunk_counter += 1
                
        return chunks
