"""Contextual Retrieval Chunking Adapter.

Implements Anthropic's Contextual Retrieval approach:
https://www.anthropic.com/news/contextual-retrieval

For each chunk, an LLM generates a short context explaining how the chunk
relates to the overall document. This context is prepended to the chunk,
dramatically improving retrieval accuracy.

Enhancements beyond vanilla Contextual Retrieval:
  - **Heading/section hierarchy**: Extracts markdown headings to provide the
    LLM with the section path (e.g. "Benefits > 401(k) Plan > Matching").
  - **Previous chunk summary**: Passes truncated text of the preceding chunk
    so the LLM understands continuation and cross-chunk references.
  - **Document title**: Extracted from first H1 heading or metadata, included
    in the prompt for better document-level awareness.

Example:
  Original chunk: "The company matches 100% of the first 3%..."
  
  With context: "This chunk is from the Employee Benefits Guide, specifically 
  the 401(k) Plan section under Retirement Benefits. It describes the employer 
  matching policy. The company matches 100% of the first 3%..."
"""

from typing import Dict, List, Optional, Any, Set, Tuple
import re
import hashlib
import json
import logging
import os
import time
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_lib.interfaces.chunking_manager import ChunkingManager, Chunk
from rag_lib.utils.page_markers import extract_slide_type_map

logger = logging.getLogger(__name__)


class ContextualRetrievalAdapter(ChunkingManager):
    """
    Contextual Retrieval chunking strategy (Anthropic's approach).
    
    This strategy:
    1. Splits document into chunks using recursive splitting
    2. For each chunk, uses an LLM to generate situating context
    3. Prepends the context to each chunk for better retrieval
    
    The context explains:
    - Where this chunk appears in the document
    - What section/topic it belongs to
    - How it relates to surrounding content
    
    Args:
        max_tokens: Maximum tokens per chunk (default: 500)
        overlap_tokens: Overlap between chunks (default: 50)
        llm_config: LLM configuration for context generation
        max_context_tokens: Maximum tokens for generated context (default: 100)
        parallel_requests: Number of parallel LLM requests (default: 5)
        context_cache: Whether to cache generated contexts (default: True)
        include_doc_context_in_chunks: When True, prepend a compact document
            header (title + summary) to every chunk's embedded content so the
            embedding captures document-level semantics (default: True)
    """

    # Regex for markdown headings
    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)

    # Regex for markdown table rows: lines starting and ending with |
    _TABLE_ROW_RE = re.compile(r"^\|.+\|\s*$")
    # Regex for table separator rows: | --- | --- | (with optional alignment colons)
    _TABLE_SEP_RE = re.compile(r"^\|\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|\s*$")

    # Prompt template for generating chunk context (enhanced with structural signals)
    CONTEXT_PROMPT = """<document>
{document}
</document>
{title_block}{section_block}{previous_block}
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Consider the document title, the section hierarchy this chunk belongs to, and how this chunk relates to the preceding content. Answer only with the context, nothing else."""

    def __init__(
        self,
        max_tokens: int = 500,
        min_tokens: int = 50,
        overlap_tokens: int = 50,
        llm_config: Optional[Dict[str, Any]] = None,
        max_context_tokens: int = 100,
        parallel_requests: int = 5,
        context_cache: bool = True,
        context_cache_path: Optional[str] = None,
        request_delay: float = 0.0,
        batch_delay: float = 1.0,
        max_doc_context_chars: int = 50000,
        include_doc_context_in_chunks: bool = True,
        **kwargs
    ):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        self.llm_config = llm_config or {}
        self.max_context_tokens = max_context_tokens
        self.parallel_requests = parallel_requests
        self.context_cache = context_cache
        self.context_cache_path = context_cache_path  # Path for persistent cache file
        self.request_delay = request_delay    # Delay (seconds) between individual LLM calls
        self.batch_delay = batch_delay        # Delay (seconds) between batches of parallel calls
        self.max_doc_context_chars = max_doc_context_chars  # Max chars for document context in LLM prompt
        self.include_doc_context_in_chunks = include_doc_context_in_chunks  # Prepend doc title + summary to chunk content
        
        # Cache for generated contexts (chunk_hash -> context)
        self._context_cache: Dict[str, str] = {}
        
        # Auto-load persistent cache if path is set and file exists
        if self.context_cache and self.context_cache_path:
            self.load_context_cache()
        
        # LLM manager (lazy loaded)
        self._llm_manager = None
        
        # Approximate chars per token
        self.chars_per_token = 4
        
        # Chunk splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_tokens * self.chars_per_token,
            chunk_overlap=overlap_tokens * self.chars_per_token,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        )

    @property
    def llm_manager(self):
        """Lazy-load LLM manager for context generation."""
        if self._llm_manager is None:
            from rag_lib.factory import RAGFactory
            from rag_lib.model_config import (
                ModelConfigError,
                require_model_spec,
                resolve_config_provider_key,
            )

            provider = self.llm_config.get("provider") or RAGFactory.get_default_llm_provider()

            # Resolve {model_id, region} strictly: explicit llm_config first,
            # then model_defaults.llm.<provider> - raises if neither is configured.
            model_id = self.llm_config.get("model_id")
            region = self.llm_config.get("region")
            if not model_id or not region:
                spec = require_model_spec("llm", resolve_config_provider_key(provider))
                model_id = model_id or spec["model_id"]
                region = region or spec["region"]

            self._llm_manager = RAGFactory.get_llm_manager(
                provider=provider,
                model_id=model_id,
                region=region,
            )
        return self._llm_manager

    def _get_chunk_hash(self, chunk_text: str, document_hash: str) -> str:
        """Generate a hash for caching chunk context."""
        combined = f"{document_hash}:{chunk_text[:200]}"
        return hashlib.sha256(combined.encode()).hexdigest()

    # =========================================================================
    # PERSISTENT CACHE I/O
    # =========================================================================

    def save_context_cache(self, path: Optional[str] = None) -> str:
        """
        Save the in-memory context cache to a JSON file.
        
        Args:
            path: File path to save to. Defaults to self.context_cache_path.
            
        Returns:
            The path the cache was saved to.
            
        Raises:
            ValueError: If no path is provided and context_cache_path is not set.
        """
        save_path = path or self.context_cache_path
        if not save_path:
            raise ValueError(
                "No cache path specified. Pass 'path' or set 'context_cache_path' in config."
            )
        
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        cache_data = {
            "version": 1,
            "num_entries": len(self._context_cache),
            "config": {
                "max_tokens": self.max_tokens,
                "overlap_tokens": self.overlap_tokens,
                "max_context_tokens": self.max_context_tokens,
                "max_doc_context_chars": self.max_doc_context_chars,
                "model_id": self.llm_config.get("model_id", "unknown"),
            },
            "contexts": self._context_cache,
        }
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Context cache saved: {len(self._context_cache)} entries → {save_path}")
        return save_path

    def load_context_cache(self, path: Optional[str] = None) -> int:
        """
        Load a previously saved context cache from a JSON file.
        
        Merges loaded entries into the current in-memory cache (does not
        overwrite entries already present).
        
        Args:
            path: File path to load from. Defaults to self.context_cache_path.
            
        Returns:
            Number of entries loaded.
        """
        load_path = path or self.context_cache_path
        if not load_path or not os.path.exists(load_path):
            if load_path:
                logger.info(f"No cache file found at {load_path}, starting fresh.")
            return 0
        
        try:
            with open(load_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            
            contexts = cache_data.get("contexts", {})
            
            # Merge: existing in-memory entries take priority
            loaded = 0
            for key, value in contexts.items():
                if key not in self._context_cache:
                    self._context_cache[key] = value
                    loaded += 1
            
            logger.info(
                f"Context cache loaded: {loaded} new entries from {load_path} "
                f"(total in-memory: {len(self._context_cache)})"
            )
            return loaded
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load context cache from {load_path}: {e}")
            return 0

    # =========================================================================
    # TABLE DETECTION & TABLE-AWARE SPLITTING
    # =========================================================================

    def _extract_tables(self, content: str) -> List[Dict[str, Any]]:
        """
        Detect and extract markdown tables from content.

        A valid markdown table has:
          - A header row (pipe-delimited)
          - A separator row (e.g. |---|---|)
          - At least one data row

        Returns:
            List of dicts with keys:
              - markdown: Full table text
              - row_count: Number of data rows (excl. header + separator)
              - col_count: Number of columns
              - start_offset: Character offset where the table starts
              - end_offset: Character offset where the table ends
        """
        tables: List[Dict[str, Any]] = []
        lines = content.split("\n")
        table_lines: List[str] = []
        start_line_idx: Optional[int] = None
        has_separator = False

        def _flush_table():
            nonlocal table_lines, start_line_idx, has_separator
            if has_separator and len(table_lines) >= 3:
                table_md = "\n".join(table_lines)
                # Count data rows (total lines minus header and separator)
                data_rows = len(table_lines) - 2
                # Column count from header
                header_cells = [c.strip() for c in table_lines[0].split("|") if c.strip()]
                col_count = len(header_cells)
                # Calculate character offsets
                offset = sum(len(lines[j]) + 1 for j in range(start_line_idx))
                end_offset = offset + len(table_md)
                tables.append({
                    "markdown": table_md,
                    "row_count": data_rows,
                    "col_count": col_count,
                    "start_offset": offset,
                    "end_offset": end_offset,
                })
            table_lines = []
            start_line_idx = None
            has_separator = False

        for i, line in enumerate(lines):
            stripped = line.strip()
            if self._TABLE_ROW_RE.match(stripped):
                if not table_lines:
                    start_line_idx = i
                table_lines.append(line)
                # Check if this line is a separator row
                if not has_separator and self._TABLE_SEP_RE.match(stripped):
                    has_separator = True
            else:
                _flush_table()

        # Flush any table at end of content
        _flush_table()
        return tables

    def _merge_cross_page_tables(
        self, page_segments: List[Tuple[str, Optional[str]]]
    ) -> List[Tuple[str, Optional[str]]]:
        """
        Detect and merge tables that span across page boundaries.

        If a page ends with table rows and the next page starts with table
        rows (continuation without a separator row = already part of the
        same table), the two pages are merged into one segment.

        Args:
            page_segments: Output of _split_by_page()

        Returns:
            Merged list of (content, page_number) tuples.  Page numbers
            for merged segments are hyphenated, e.g. "3-4".
        """
        if len(page_segments) <= 1:
            return page_segments

        merged: List[Tuple[str, Optional[str]]] = [page_segments[0]]

        for content, page_num in page_segments[1:]:
            prev_content, prev_page = merged[-1]
            # Check if previous page ends mid-table
            prev_lines = prev_content.rstrip().split("\n")
            prev_trailing: List[str] = []
            for ln in reversed(prev_lines):
                if self._TABLE_ROW_RE.match(ln.strip()):
                    prev_trailing.append(ln)
                else:
                    break

            # Check if current page starts mid-table
            curr_lines = content.lstrip().split("\n")
            curr_leading: List[str] = []
            for ln in curr_lines:
                if self._TABLE_ROW_RE.match(ln.strip()):
                    curr_leading.append(ln)
                else:
                    break

            # If both sides have table rows and the continuation does NOT
            # contain its own separator (i.e. it's a continuation, not a
            # brand-new table), merge the segments.
            has_new_separator = any(
                self._TABLE_SEP_RE.match(ln.strip()) for ln in curr_leading
            )
            if prev_trailing and curr_leading and not has_new_separator:
                merged_page = (
                    f"{prev_page}-{page_num}" if prev_page and page_num
                    else prev_page or page_num
                )
                merged[-1] = (prev_content.rstrip() + "\n" + content.lstrip(), merged_page)
                logger.info(
                    f"Merged cross-page table: pages {prev_page} + {page_num} "
                    f"({len(prev_trailing)} + {len(curr_leading)} table rows)"
                )
            else:
                merged.append((content, page_num))

        return merged

    # Table placeholder sentinel (will never appear in real markdown)
    _TABLE_PLACEHOLDER = "\n\n__TABLE_PLACEHOLDER_{idx}__\n\n"

    def _table_aware_split(
        self, content: str, page_num: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Split page content into chunks while keeping tables atomic.

        Strategy:
          1. Extract all markdown tables from the content.
          2. Replace each table with a unique placeholder.
          3. Run the standard RecursiveCharacterTextSplitter on the
             placeholder-bearing text.
          4. For each resulting chunk, restore placeholders back to the
             original table markdown.
          5. If a table alone exceeds max chunk size, it becomes its own
             chunk (never split mid-row).

        Returns:
            List of dicts: {"text": str, "page_num": str, "tables": list}
        """
        tables = self._extract_tables(content)

        if not tables:
            # No tables — standard splitting
            docs = self.splitter.create_documents([content])
            return [
                {"text": doc.page_content, "page_num": page_num, "tables": []}
                for doc in docs
                if doc.page_content.strip()
            ]

        # Replace tables with placeholders (process in reverse to keep offsets valid)
        modified_content = content
        placeholder_map: Dict[str, Dict[str, Any]] = {}
        for idx, table in enumerate(reversed(tables)):
            ph = f"__TABLE_PLACEHOLDER_{len(tables) - 1 - idx}__"
            actual_idx = len(tables) - 1 - idx
            placeholder_map[ph] = tables[actual_idx]
            # Find the table markdown in the content and replace it
            modified_content = modified_content.replace(
                table["markdown"], f"\n\n{ph}\n\n", 1
            )

        # Split the modified content (tables are now short placeholders)
        docs = self.splitter.create_documents([modified_content])

        raw_chunks: List[Dict[str, Any]] = []
        used_placeholders: set = set()

        for doc in docs:
            text = doc.page_content
            if not text.strip():
                continue

            chunk_tables: List[Dict[str, Any]] = []

            # Restore any placeholders in this chunk
            for ph, table_info in placeholder_map.items():
                if ph in text:
                    text = text.replace(ph, table_info["markdown"])
                    chunk_tables.append(table_info)
                    used_placeholders.add(ph)

            raw_chunks.append({
                "text": text,
                "page_num": page_num,
                "tables": chunk_tables,
            })

        # Any table whose placeholder was split across chunks (shouldn't happen,
        # but defensive) — emit it as its own chunk
        for ph, table_info in placeholder_map.items():
            if ph not in used_placeholders:
                raw_chunks.append({
                    "text": table_info["markdown"],
                    "page_num": page_num,
                    "tables": [table_info],
                })
                logger.info(
                    f"Table emitted as standalone chunk on page {page_num} "
                    f"({table_info['row_count']} rows, {table_info['col_count']} cols)"
                )

        return raw_chunks

    # =========================================================================
    # STRUCTURAL CONTEXT EXTRACTION (no LLM calls)
    # =========================================================================

    def _extract_document_title(self, markdown: str, base_metadata: Dict[str, str]) -> str:
        """
        Extract document title from metadata or first H1 heading.
        
        Priority:
          1. base_metadata["title"] (set during ingestion)
          2. First H1 heading in markdown
          3. Empty string (omitted from prompt)
        """
        title = base_metadata.get("title", "").strip()
        if title:
            return title
        
        # Fall back to first H1
        match = re.search(r"^#\s+(.+)", markdown, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return ""

    def _build_heading_map(
        self, markdown: str, raw_chunks: List[Dict]
    ) -> List[str]:
        """
        Build a heading-chain (section path) for every raw chunk.
        
        Scans the full markdown for headings, tracks a heading stack,
        and maps each chunk to the heading hierarchy active at its position.
        
        Args:
            markdown: Full markdown text
            raw_chunks: List of dicts with at least 'text' key
            
        Returns:
            List[str] of heading chains, one per chunk, e.g.
            "Benefits > 401(k) Plan > Employer Match"
        """
        # 1. Collect all heading positions
        headings: List[Tuple[int, int, str]] = []  # (position, level, text)
        for m in self._HEADING_RE.finditer(markdown):
            level = len(m.group(1))  # number of '#' characters
            text = m.group(2).strip()
            headings.append((m.start(), level, text))
        
        if not headings:
            return [""] * len(raw_chunks)
        
        # 2. Find each chunk's start position in the markdown
        chunk_positions: List[int] = []
        search_start = 0
        for rc in raw_chunks:
            # Use first 200 chars for matching to avoid cost of full search
            snippet = rc["text"][:200].strip()
            pos = markdown.find(snippet, search_start)
            if pos == -1:
                # Fallback: search from beginning (overlapping chunks)
                pos = markdown.find(snippet)
            chunk_positions.append(pos if pos != -1 else search_start)
            if pos != -1:
                search_start = pos  # advance for next chunk
        
        # 3. For each chunk, find the active heading stack at that position
        heading_chains: List[str] = []
        for chunk_pos in chunk_positions:
            # Gather all headings that appear before this chunk's position
            heading_stack: List[Tuple[int, str]] = []  # (level, text)
            for h_pos, h_level, h_text in headings:
                if h_pos > chunk_pos:
                    break
                # Pop headings at same or deeper level (maintain hierarchy)
                while heading_stack and heading_stack[-1][0] >= h_level:
                    heading_stack.pop()
                heading_stack.append((h_level, h_text))
            
            chain = " > ".join(text for _, text in heading_stack)
            heading_chains.append(chain)
        
        return heading_chains

    @staticmethod
    def _get_previous_chunk_summary(raw_chunks: List[Dict], idx: int, max_chars: int = 300) -> str:
        """
        Get a truncated summary of the previous chunk's text.
        
        This gives the LLM continuity cues without an extra call.
        
        Args:
            raw_chunks: All raw chunks in order
            idx: Current chunk index
            max_chars: Maximum characters to include from previous chunk
            
        Returns:
            Truncated previous chunk text, or empty string for the first chunk
        """
        if idx <= 0:
            return ""
        prev_text = raw_chunks[idx - 1]["text"].strip()
        if len(prev_text) <= max_chars:
            return prev_text
        return prev_text[:max_chars] + "..."

    # =========================================================================
    # KEYWORD EXTRACTION (no LLM calls — pure text analysis)
    # =========================================================================

    # Common English stop words to filter out
    _STOP_WORDS: Set[str] = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "need", "must",
        "it", "its", "this", "that", "these", "those", "he", "she", "they",
        "we", "you", "i", "me", "him", "her", "us", "them", "my", "your",
        "his", "our", "their", "what", "which", "who", "whom", "whose",
        "when", "where", "why", "how", "all", "each", "every", "both",
        "few", "more", "most", "other", "some", "such", "no", "not", "only",
        "own", "same", "so", "than", "too", "very", "just", "also", "if",
        "then", "else", "about", "up", "out", "into", "over", "after",
        "before", "between", "under", "above", "below", "through", "during",
        "without", "within", "along", "following", "across", "behind",
        "beyond", "plus", "except", "since", "until", "against", "among",
        "throughout", "despite", "towards", "upon", "however", "therefore",
        "although", "though", "while", "because", "unless", "whether",
        "as", "like", "once", "here", "there", "any", "many", "much",
        "well", "back", "still", "even", "new", "one", "two", "first",
        "last", "long", "great", "little", "right", "old", "big", "high",
        "different", "next", "early", "young", "important", "public",
        "make", "made", "get", "got", "go", "going", "see", "say", "said",
        "come", "take", "know", "think", "give", "use", "used", "using",
        "find", "tell", "ask", "work", "seem", "feel", "try", "leave",
        "call", "keep", "let", "begin", "show", "hear", "play", "run",
        "move", "live", "believe", "hold", "bring", "happen", "write",
        "provide", "sit", "stand", "lose", "pay", "meet", "include",
        "continue", "set", "learn", "change", "lead", "understand",
        "watch", "follow", "stop", "create", "speak", "read", "allow",
        "add", "spend", "grow", "open", "walk", "win", "offer", "remember",
        "consider", "appear", "buy", "wait", "serve", "die", "send",
        "build", "stay", "fall", "cut", "reach", "kill", "remain",
        "etc", "e.g", "i.e", "eg", "ie", "per", "via", "vs",
    }

    # Pattern for extracting meaningful tokens (words, numbers, hyphenated terms)
    _KEYWORD_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]*(?:'[A-Za-z]+)?|\d+(?:\.\d+)?%?")

    # Pattern for multi-word proper nouns / capitalized phrases
    _PROPER_NOUN_RE = re.compile(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)")

    def _extract_chunk_keywords(
        self, chunk_text: str, max_keywords: int = 10
    ) -> List[str]:
        """
        Extract important keywords from a single chunk using text analysis.
        
        No LLM call — uses frequency analysis, proper-noun detection, and
        markdown heading extraction to identify the most relevant terms.
        
        Strategy:
          1. Extract multi-word proper nouns (e.g. "United States", "John Smith")
          2. Extract single tokens, filter stop words & short words
          3. Give bonus weight to:
             - Words that appear in markdown headings within the chunk
             - Capitalized words (likely proper nouns / entities)
             - Longer words (more specific / informative)
             - Numbers with % (statistics / metrics)
          4. Return top-N by weighted frequency
        
        Args:
            chunk_text: The raw chunk text
            max_keywords: Maximum keywords to return
            
        Returns:
            List of keyword strings, ordered by relevance
        """
        keywords: List[str] = []
        seen_lower: Set[str] = set()
        
        # 1. Multi-word proper nouns (high value)
        for match in self._PROPER_NOUN_RE.finditer(chunk_text):
            phrase = match.group().strip()
            if phrase.lower() not in seen_lower and len(phrase.split()) <= 4:
                keywords.append(phrase)
                seen_lower.add(phrase.lower())
        
        # 2. Heading terms (boosted weight)
        heading_words: Set[str] = set()
        for m in self._HEADING_RE.finditer(chunk_text):
            for word in self._KEYWORD_TOKEN_RE.findall(m.group(2)):
                if word.lower() not in self._STOP_WORDS and len(word) >= 3:
                    heading_words.add(word.lower())
        
        # 3. Token frequency analysis
        tokens = self._KEYWORD_TOKEN_RE.findall(chunk_text)
        counter: Counter = Counter()
        for token in tokens:
            low = token.lower()
            # Skip stop words and very short tokens
            if low in self._STOP_WORDS or len(token) < 3:
                continue
            # Weight: base 1 + bonuses
            weight = 1
            if low in heading_words:
                weight += 3  # heading terms are highly relevant
            if token[0].isupper():
                weight += 1  # likely a proper noun / entity
            if len(token) >= 8:
                weight += 1  # longer = more specific
            if re.match(r"\d+(?:\.\d+)?%", token):
                weight += 2  # statistics / metrics
            counter[low] += weight
        
        # 4. Build final keyword list (proper nouns first, then by frequency)
        remaining_slots = max_keywords - len(keywords)
        if remaining_slots > 0:
            for word, _ in counter.most_common(remaining_slots + 10):
                if word not in seen_lower:
                    # Use original casing from first occurrence
                    original = next(
                        (t for t in tokens if t.lower() == word), word
                    )
                    keywords.append(original)
                    seen_lower.add(word)
                if len(keywords) >= max_keywords:
                    break
        
        return keywords[:max_keywords]

    def _generate_context(
        self,
        chunk_text: str,
        document_text: str,
        document_hash: str,
        document_title: str = "",
        section_path: str = "",
        previous_chunk_summary: str = "",
    ) -> Optional[str]:
        """
        Generate situating context for a chunk using LLM.
        
        Args:
            chunk_text: The chunk to contextualize
            document_text: The full document (or truncated version)
            document_hash: Hash of document for caching
            document_title: Extracted document title
            section_path: Heading hierarchy for this chunk (e.g. "Benefits > 401(k)")
            previous_chunk_summary: Truncated text from the preceding chunk
            
        Returns:
            Generated context string or None if failed
        """
        # Compute cache key (needed for both lookup and store)
        cache_key = self._get_chunk_hash(chunk_text, document_hash)
        
        # Check cache first
        if self.context_cache and cache_key in self._context_cache:
            return self._context_cache[cache_key]
        
        # Build optional prompt blocks (only included when data is available)
        title_block = f"\n<document_title>\n{document_title}\n</document_title>\n" if document_title else ""
        section_block = f"\n<section_path>\n{section_path}\n</section_path>\n" if section_path else ""
        previous_block = f"\n<previous_chunk>\n{previous_chunk_summary}\n</previous_chunk>\n" if previous_chunk_summary else ""
        
        prompt = self.CONTEXT_PROMPT.format(
            document=document_text,
            chunk=chunk_text,
            title_block=title_block,
            section_block=section_block,
            previous_block=previous_block,
        )
        
        # Retry with exponential backoff + jitter for throttling errors
        max_retries = 10
        for attempt in range(max_retries):
            try:
                # Small delay before each request to avoid bursting
                if self.request_delay > 0 and attempt == 0:
                    time.sleep(self.request_delay * random.uniform(0.5, 1.5))
                
                response = self.llm_manager.generate(
                    prompt=prompt,
                    system_prompt="You are a helpful assistant that provides concise context for document chunks to improve search retrieval.",
                    max_tokens=self.max_context_tokens,
                    temperature=0.0,  # Deterministic for consistency
                )
                
                context = response.text.strip()
                
                # Cache the result
                if self.context_cache:
                    self._context_cache[cache_key] = context
                
                return context
                
            except Exception as e:
                error_str = str(e)
                is_throttle = "ThrottlingException" in error_str or "Too many tokens" in error_str or "rate" in error_str.lower()
                
                if is_throttle and attempt < max_retries - 1:
                    # Exponential backoff with jitter: ~5s, ~10s, ~20s, ~40s, ~80s, ...
                    base_wait = (2 ** attempt) * 5
                    jitter = random.uniform(0.5, 1.5)
                    wait_time = min(base_wait * jitter, 120)  # Cap at 2 minutes
                    logger.info(f"Throttled on chunk context generation (attempt {attempt + 1}/{max_retries}), waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                
                logger.warning(f"Failed to generate context for chunk (attempt {attempt + 1}/{max_retries}): {e}")
                return None

    def _generate_contexts_parallel(
        self,
        raw_chunks: List[Dict],
        document_text: str,
        document_hash: str,
        document_title: str = "",
        heading_chains: Optional[List[str]] = None,
    ) -> List[Optional[str]]:
        """
        Generate contexts for chunks in batches with throttle control.
        
        Processes chunks in batches of `parallel_requests` size, with a
        configurable delay between batches to respect provider throughput limits.
        
        Each chunk receives structural signals (title, section path, previous
        chunk text) in addition to the full document.
        
        Args:
            raw_chunks: List of raw chunk dicts with 'text' key
            document_text: Full document text
            document_hash: Document hash for caching
            document_title: Extracted document title
            heading_chains: Per-chunk heading hierarchy strings
            
        Returns:
            List of contexts (same order as chunks)
        """
        num_chunks = len(raw_chunks)
        contexts: List[Optional[str]] = [None] * num_chunks
        batch_size = max(self.parallel_requests, 1)
        total_batches = (num_chunks + batch_size - 1) // batch_size
        
        # Pre-compute per-chunk signals (zero LLM cost)
        _heading_chains = heading_chains or [""] * num_chunks
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, num_chunks)
            
            logger.info(
                f"Processing context batch {batch_num + 1}/{total_batches} "
                f"(chunks {start_idx + 1}-{end_idx} of {num_chunks})"
            )
            
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                future_to_idx = {}
                for i in range(start_idx, end_idx):
                    prev_summary = self._get_previous_chunk_summary(raw_chunks, i)
                    future = executor.submit(
                        self._generate_context,
                        raw_chunks[i]["text"],
                        document_text,
                        document_hash,
                        document_title=document_title,
                        section_path=_heading_chains[i],
                        previous_chunk_summary=prev_summary,
                    )
                    future_to_idx[future] = i
                
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        contexts[idx] = future.result()
                    except Exception as e:
                        logger.warning(f"Context generation failed for chunk {idx}: {e}")
                        contexts[idx] = None
            
            # Delay between batches to avoid hitting TPM limits
            if batch_num < total_batches - 1 and self.batch_delay > 0:
                logger.info(f"Batch delay: waiting {self.batch_delay}s before next batch...")
                time.sleep(self.batch_delay)
        
        return contexts

    # Regex to match markdown image references: ![alt](url) or ![alt](url "title")
    _IMAGE_REF_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")

    def _truncate_document_for_context(self, document: str, max_chars: int = None) -> str:
        """
        Truncate document for LLM context window.
        
        Steps:
          1. Strip markdown image references (waste tokens, no semantic value)
          2. Truncate to max_chars, keeping beginning and end
        
        Args:
            document: Full document text
            max_chars: Maximum characters (defaults to self.max_doc_context_chars)
        """
        max_chars = max_chars or self.max_doc_context_chars
        
        # Strip image references to save token budget
        document = self._IMAGE_REF_RE.sub("", document)
        
        if len(document) <= max_chars:
            return document
        
        # Keep 40% from start, 40% from end, truncate middle
        keep_chars = max_chars // 2
        start = document[:keep_chars]
        end = document[-keep_chars:]
        
        return f"{start}\n\n[... middle of document truncated for context generation ...]\n\n{end}"

    def _prepend_context(
        self,
        context: Optional[str],
        chunk_text: str,
        document_title: str = "",
        document_summary: str = "",
    ) -> str:
        """Prepend document context and chunk context to chunk text.

        When ``include_doc_context_in_chunks`` is enabled, a compact
        document-level header (title + summary) is prepended before the
        per-chunk LLM context.  This lets the embedding model capture
        document-level semantics alongside the chunk content.

        Resulting format:
            [Document: <title>] <summary>\n\n<chunk_context>\n\n<chunk_text>
        """
        parts: list[str] = []

        # 1. Optional document-level header
        if self.include_doc_context_in_chunks and (document_title or document_summary):
            doc_header_parts: list[str] = []
            if document_title:
                doc_header_parts.append(f"[Document: {document_title}]")
            if document_summary:
                doc_header_parts.append(document_summary)
            parts.append(" ".join(doc_header_parts))

        # 2. Per-chunk LLM context
        if context:
            parts.append(context)

        # 3. Original chunk text
        parts.append(chunk_text)

        return "\n\n".join(parts)

    def chunk(self, file_id: str, markdown: str, base_metadata: Dict[str, str]) -> List[Chunk]:
        """
        Chunk markdown content with contextual retrieval.
        
        Each chunk gets LLM-generated context prepended that explains
        its position and relevance within the document.
        
        Args:
            file_id: Unique identifier for the file
            markdown: Markdown content to chunk
            base_metadata: Base metadata to attach to each chunk
            
        Returns:
            List of Chunk objects with contextualized content
        """
        chunks: List[Chunk] = []
        
        # Generate document hash for caching
        document_hash = hashlib.sha256(markdown[:5000].encode()).hexdigest()
        
        # Truncate document for context generation (LLM context limits)
        truncated_doc = self._truncate_document_for_context(markdown)
        
        # Captured from the raw markdown before splitting strips the markers —
        # PPTX-only, maps page_number -> "title" | "content" (any page position).
        slide_type_map = extract_slide_type_map(markdown)

        # Split by page first, then merge cross-page tables
        page_segments = self._split_by_page(markdown)
        page_segments = self._merge_cross_page_tables(page_segments)
        
        # Build page-level table map (for page_tables metadata)
        page_tables_map: Dict[Optional[str], List[Dict[str, Any]]] = {}
        for content, page_num in page_segments:
            page_tbls = self._extract_tables(content)
            if page_tbls:
                page_tables_map[page_num] = page_tbls
        
        # Collect all raw chunks using table-aware splitting
        raw_chunks: List[Dict] = []
        
        for content, page_num in page_segments:
            split_results = self._table_aware_split(content, page_num)
            raw_chunks.extend(split_results)
        
        if not raw_chunks:
            return chunks
        
        # Extract structural signals (zero LLM cost)
        document_title = self._extract_document_title(markdown, base_metadata)
        document_summary = base_metadata.get("document_summary", "")
        heading_chains = self._build_heading_map(markdown, raw_chunks)
        
        logger.info(
            f"Generating contexts for {len(raw_chunks)} chunks in {file_id} "
            f"(title='{document_title[:50]}', headings_found={sum(1 for h in heading_chains if h)})"
        )
        
        # Generate contexts for all chunks (in parallel, with structural signals)
        contexts = self._generate_contexts_parallel(
            raw_chunks,
            truncated_doc,
            document_hash,
            document_title=document_title,
            heading_chains=heading_chains,
        )
        
        # Create final chunks with context prepended
        for idx, (raw_chunk, context) in enumerate(zip(raw_chunks, contexts)):
            chunk_id = f"{file_id}-{idx:05d}"
            
            # Prepend context to chunk (with optional doc-level header)
            contextualized_content = self._prepend_context(
                context,
                raw_chunk["text"],
                document_title=document_title,
                document_summary=document_summary,
            )
            
            # Extract keywords from the raw chunk text (no LLM call)
            chunk_keywords = self._extract_chunk_keywords(raw_chunk["text"])
            
            metadata = base_metadata.copy()
            metadata["has_context"] = "true" if context else "false"
            
            # Store document title for document-aware retrieval / filtering
            if document_title:
                metadata["document_title"] = document_title
            
            # Store chunk context summary separately for retrieval/display
            if context:
                metadata["chunk_context"] = context
            
            # Store heading chain for downstream retrieval/filtering
            if heading_chains[idx]:
                metadata["heading_chain"] = heading_chains[idx]
            
            if raw_chunk["page_num"]:
                metadata["page_number"] = raw_chunk["page_num"]
            if raw_chunk["page_num"] in slide_type_map:
                metadata["slide_type"] = slide_type_map[raw_chunk["page_num"]]
            
            # ---- Table metadata ----
            chunk_tables = raw_chunk.get("tables", [])
            if not chunk_tables:
                # Fallback: detect tables in chunk text (for non-table-aware paths)
                chunk_tables = self._extract_tables(raw_chunk["text"])
            
            if chunk_tables:
                metadata["has_table"] = "true"
                metadata["table_content"] = "\n\n".join(
                    t["markdown"] for t in chunk_tables
                )
                metadata["table_row_count"] = str(
                    sum(t["row_count"] for t in chunk_tables)
                )
                metadata["table_col_count"] = str(
                    max(t["col_count"] for t in chunk_tables)
                )
            else:
                metadata["has_table"] = "false"
            
            # Append table column headers as keywords
            if chunk_tables:
                seen_kw = {k.lower() for k in chunk_keywords}
                for tbl in chunk_tables:
                    header_line = tbl["markdown"].split("\n", 1)[0]
                    for cell in header_line.split("|"):
                        col = cell.strip()
                        if col and col.lower() not in seen_kw and col.lower() not in self._STOP_WORDS:
                            chunk_keywords.append(col)
                            seen_kw.add(col.lower())
            
            # Store chunk keywords for hybrid search / filtering
            if chunk_keywords:
                metadata["chunk_keywords"] = ",".join(chunk_keywords)
            
            # Attach full page-level tables for sibling-chunk context
            page_key = raw_chunk["page_num"]
            if page_key and page_key in page_tables_map:
                all_page_tables = page_tables_map[page_key]
                # Only add if the page has tables not already in this chunk
                page_table_md = "\n\n".join(
                    t["markdown"] for t in all_page_tables
                )
                if page_table_md:
                    metadata["page_tables"] = page_table_md
            
            # ---- Structured data envelope (has_data / data) ----
            _data_tables = [
                {
                    "markdown": t["markdown"],
                    "row_count": t["row_count"],
                    "col_count": t["col_count"],
                    "page": raw_chunk["page_num"] or "",
                }
                for t in chunk_tables
            ]
            # Placeholder arrays for future enrichment
            _data_attachments: list = []
            _data_links: list = []
            
            if _data_tables or _data_attachments or _data_links:
                metadata["has_data"] = "true"
                metadata["data"] = json.dumps({
                    "tables": _data_tables,
                    "attachments": _data_attachments,
                    "links": _data_links,
                })
            else:
                metadata["has_data"] = "false"
            
            chunk = Chunk(
                chunk_id=chunk_id,
                file_id=file_id,
                content=contextualized_content,
                order=idx,
                metadata=metadata,
                parent_id=None,
                level="chunk"
            )
            chunks.append(chunk)
        
        # Log statistics
        contexts_generated = sum(1 for c in contexts if c is not None)
        logger.info(
            f"Contextual chunking complete for {file_id}: "
            f"{len(chunks)} chunks, {contexts_generated}/{len(chunks)} contextualized"
        )
        
        # Auto-save persistent cache if path is configured
        if self.context_cache and self.context_cache_path:
            try:
                self.save_context_cache()
            except Exception as e:
                logger.warning(f"Failed to auto-save context cache: {e}")
        
        return chunks


__all__ = ["ContextualRetrievalAdapter"]
