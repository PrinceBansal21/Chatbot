import re
from typing import Dict, List, Optional
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from rag_lib.interfaces.chunking_manager import ChunkingManager, Chunk
from rag_lib.utils.page_markers import extract_slide_type_map

class SemanticChunkingAdapter(ChunkingManager):
    """
    Chunker that respects document structure (headings) for AI search.
    Implemented using LangChain's MarkdownHeaderTextSplitter and RecursiveCharacterTextSplitter.
    """

    def __init__(
        self,
        max_tokens: int = 650,
        min_tokens: int = 250,
        overlap_tokens: int = 120,
        **kwargs
    ) -> None:
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens

        # Define headers to split on
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
            ("######", "Header 6"),
        ]

        # Splitter for structure (headers)
        # strip_headers=False ensures headers remain in the content, matching original behavior
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False 
        )

        # Splitter for content size (tokens)
        # Using tiktoken encoder for more accurate token counting than the previous heuristic
        self.recursive_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=max_tokens,
            chunk_overlap=overlap_tokens,
            model_name="gpt-3.5-turbo",
            disallowed_special=()
        )

    def chunk(self, file_id: str, markdown: str, base_metadata: Dict[str, str]) -> List[Chunk]:
        # 1. Split by Structure (Headers)
        header_docs = self.header_splitter.split_text(markdown)
        
        chunks: List[Chunk] = []
        chunk_counter = 0

        # Captured from the raw markdown before page markers are stripped —
        # PPTX-only, maps page_number -> "title" | "content" (any page position).
        slide_type_map = extract_slide_type_map(markdown)
        
        # 2. Split large sections recursively based on token count
        for doc in header_docs:
            # Split by page first
            page_segments = self._split_by_page(doc.page_content)
            
            for content, page_num in page_segments:
                sub_docs = self.recursive_splitter.split_documents(
                    self.recursive_splitter.create_documents([content])
                )
                
                for sub_doc in sub_docs:
                    chunk_id = f"{file_id}-{chunk_counter:05d}"
                    metadata = base_metadata.copy()
                    
                    # Extract the deepest header to use as 'section_heading'
                    # from the original doc metadata (as sub_doc is new and lacks header metadata unless we pass it)
                    # wait, split_documents preserves metadata if passed, but create_documents([content]) creates fresh docs
                    # We need to use doc.metadata
                    
                    current_heading = None
                    for h in ["Header 6", "Header 5", "Header 4", "Header 3", "Header 2", "Header 1"]:
                        if doc.metadata.get(h):
                            current_heading = doc.metadata[h]
                            break
                    
                    if current_heading:
                        metadata["section_heading"] = current_heading
                        
                    if page_num:
                        metadata["page_number"] = page_num
                    if page_num in slide_type_map:
                        metadata["slide_type"] = slide_type_map[page_num]
                    
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            file_id=file_id,
                            content=sub_doc.page_content.strip(),
                            order=chunk_counter,
                            metadata=metadata,
                        )
                    )
                    chunk_counter += 1
                
        return chunks
