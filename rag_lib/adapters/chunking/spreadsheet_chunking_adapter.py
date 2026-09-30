"""Spreadsheet Chunking Adapter — Token-Based Table Chunking.

Specialized chunking strategy for XLS, XLSX, and CSV files.
Preserves full table structure by slicing the markdown table into
token-budgeted chunks (default 1000 tokens, 150 overlap).

Each chunk is a proper markdown table with headers repeated.  A parent
summary chunk (LLM-generated overview of the full sheet) is emitted
first, and per-chunk summaries + data records are stored in child
chunk metadata.

Strategy: 1 sheet = multiple chunks (parent summary + N child table slices).

Parent chunk (level="parent"):
  - LLM-generated summary for semantic/FAQ queries
  - chunking_strategy = "semantic_row_grouping"
  - has_data = false (data lives in children)
  - num_clusters = total number of child chunks

Child chunk (level="child"):
  - Token-budgeted markdown table slice with headers repeated
  - chunking_strategy = "semantic_row"
  - has_data = true with JSON records for that slice
  - chunk_index, total_chunks, summary in metadata
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any, Tuple

from rag_lib.interfaces.chunking_manager import ChunkingManager, Chunk

logger = logging.getLogger(__name__)

# ── Token estimation ─────────────────────────────────────────────
# Try tiktoken for accurate counts; fall back to char/4 heuristic.
try:
    import tiktoken as _tiktoken
    _ENCODER = _tiktoken.get_encoding("cl100k_base")

    def _estimate_tokens(text: str) -> int:
        return len(_ENCODER.encode(text))
except Exception:
    _ENCODER = None

    def _estimate_tokens(text: str) -> int:  # type: ignore[misc]
        return max(1, len(text) // 4)


class SpreadsheetChunkingAdapter(ChunkingManager):
    """
    Token-based chunking adapter for spreadsheet files (XLS, XLSX, CSV).

    Each sheet produces:
      1. A **parent summary** chunk (``level="parent"``) — an LLM-generated
         overview of the full sheet, used for high-level semantic search.
      2. **N child table chunks** (``level="child"``) — token-budgeted slices
         of the markdown table.  Headers are repeated in every chunk so each
         slice is a self-contained markdown table.  Per-chunk summaries are
         stored in ``metadata["chunk_summary"]``.

    Args:
        llm_manager: LLM manager for summary generation (optional).
        max_rows_in_data: Maximum rows to include in ``has_data`` JSON (default 10000).
        sample_rows_for_summary: Rows sampled for the LLM summary prompt (default 5).
        generate_column_schema: Whether to infer column types (default True).
        include_full_markdown: Include full markdown table in parent metadata (default True).
        max_tokens: Token budget per child chunk (default 1000).
        overlap_tokens: Overlap between consecutive child chunks in tokens (default 150).
    """

    # Regex patterns for sheet detection
    _SHEET_MARKER_RE = re.compile(
        r"(?:^|\n)(?:#{1,3}\s+(?:Sheet|Worksheet|Tab):\s*(.+?)(?:\n|$))|"
        r"(?:<!--\s*sheet:\s*(.+?)\s*-->)",
        re.IGNORECASE | re.MULTILINE,
    )

    # Regex for markdown table detection
    _TABLE_START_RE = re.compile(r"^\|(.+)\|$", re.MULTILINE)
    _TABLE_SEP_RE = re.compile(r"^\|[\s:|-]+\|$", re.MULTILINE)

    # ── LLM prompts ──────────────────────────────────────────────
    SUMMARY_PROMPT = """Analyze this spreadsheet data and generate a comprehensive summary for semantic search.

Sheet Name: {sheet_name}
File Name: {file_name}
Columns ({col_count} total): {columns}

Sample Data (first {sample_count} rows):
{sample_rows}

Statistics:
- Total rows: {row_count}
- Total columns: {col_count}
{additional_stats}

Generate a summary that:
1. Describes what this data represents (domain/purpose)
2. Explains key columns and their meaning
3. Notes data characteristics (categories, value ranges, time periods if applicable)
4. Suggests potential questions users might ask

Format your response as:
[Sheet: {sheet_name}]
<2-3 sentence description of the data>

Columns:
- <column_name>: <brief description and data type>
(list all columns)

Key Characteristics:
- <relevant statistics and patterns>"""

    CHUNK_SUMMARY_PROMPT = """Summarize the following table chunk in 1-2 sentences.
Focus on what data rows are present and any notable values.

Sheet: {sheet_name}
Rows {start_row}-{end_row} of {total_rows}

{table_chunk}

Summary:"""

    def __init__(
        self,
        llm_manager: Optional[Any] = None,
        max_rows_in_data: int = 10000,
        sample_rows_for_summary: int = 5,
        generate_column_schema: bool = True,
        include_full_markdown: bool = True,
        max_tokens: int = 1000,
        overlap_tokens: int = 150,
        **kwargs,
    ):
        self.llm_manager = llm_manager
        self.max_rows_in_data = max_rows_in_data
        self.sample_rows_for_summary = sample_rows_for_summary
        self.generate_column_schema = generate_column_schema
        self.include_full_markdown = include_full_markdown
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

        # LLM will be lazily initialized if needed
        self._llm = None
    
    def _get_llm(self):
        """Lazily initialize LLM manager."""
        if self._llm is None:
            if self.llm_manager:
                self._llm = self.llm_manager
            else:
                try:
                    from rag_lib.factory import RAGFactory
                    self._llm = RAGFactory.get_llm_manager()
                except Exception as e:
                    logger.warning(f"Could not initialize LLM manager: {e}")
                    self._llm = None
        return self._llm
    
    def chunk(
        self,
        file_id: str,
        markdown: str,
        base_metadata: Dict[str, str],
    ) -> List[Chunk]:
        """
        Convert markdown (with sheet content) into token-budgeted table chunks.

        For every sheet the method emits:
          * 1 parent summary chunk  (``level="parent"``)
          * N child table chunks    (``level="child"``)

        Args:
            file_id: Unique identifier for the source file.
            markdown: Markdown content with table(s), possibly with sheet markers.
            base_metadata: Base metadata to attach to each chunk.

        Returns:
            List of Chunk objects (parent summaries + child table slices).
        """
        logger.info(f"Starting spreadsheet chunking for {file_id}")

        # Parse sheets from markdown
        sheets = self._parse_sheets_from_markdown(markdown, base_metadata)

        # Fallback: if only 1 sheet found, try reading Excel directly
        if len(sheets) <= 1:
            source_uri = base_metadata.get("source_uri") or base_metadata.get("storage_location")
            if source_uri and self._is_spreadsheet_file(source_uri):
                logger.info(f"Single sheet found, attempting direct Excel parse from {source_uri}")
                excel_sheets = self._parse_excel_directly(source_uri, base_metadata)
                if excel_sheets:
                    logger.info(f"Found {len(excel_sheets)} sheets from Excel file")
                    sheets = excel_sheets

        if not sheets:
            logger.warning(f"No sheets found in {file_id}, treating content as single sheet")
            sheets = [{
                "name": base_metadata.get("file_name", "Sheet1").rsplit(".", 1)[0],
                "content": markdown,
                "index": 0,
            }]

        chunks: List[Chunk] = []
        for idx, sheet in enumerate(sheets):
            try:
                sheet_chunks = self._process_sheet(file_id, sheet, idx, base_metadata)
                chunks.extend(sheet_chunks)
            except Exception as e:
                logger.error(f"Error processing sheet {sheet.get('name', idx)}: {e}")
                chunks.append(self._create_fallback_chunk(file_id, sheet, idx, base_metadata))

        logger.info(f"Spreadsheet chunking complete for {file_id}: {len(chunks)} chunks")
        return chunks
    
    def _is_spreadsheet_file(self, uri: str) -> bool:
        """Check if the URI points to a spreadsheet file."""
        lower_uri = uri.lower()
        return any(lower_uri.endswith(ext) for ext in ['.xlsx', '.xls', '.xlsm', '.csv'])
    
    def _parse_excel_directly(
        self,
        source_uri: str,
        base_metadata: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Parse an Excel file from object storage to extract sheets.
        
        This is a fallback when Docling doesn't add sheet markers.
        """
        try:
            import pandas as pd
            import io
            
            file_bytes = self._fetch_file_from_uri(source_uri)
            if not file_bytes:
                logger.warning(f"Could not fetch file from {source_uri}")
                return []
            
            # Determine file type
            lower_uri = source_uri.lower()
            
            sheets = []
            if lower_uri.endswith('.csv'):
                # CSV is single sheet
                df = pd.read_csv(io.BytesIO(file_bytes))
                markdown_table = self._dataframe_to_markdown(df)
                sheets.append({
                    "name": base_metadata.get("file_name", "Sheet1").rsplit(".", 1)[0],
                    "content": markdown_table,
                    "index": 0,
                    "dataframe": df
                })
            else:
                # Excel file - read all sheets
                excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
                
                for idx, sheet_name in enumerate(excel_file.sheet_names):
                    try:
                        df = pd.read_excel(excel_file, sheet_name=sheet_name)
                        if df.empty:
                            logger.debug(f"Skipping empty sheet: {sheet_name}")
                            continue
                        
                        markdown_table = self._dataframe_to_markdown(df)
                        sheets.append({
                            "name": sheet_name,
                            "content": markdown_table,
                            "index": idx,
                            "dataframe": df
                        })
                        logger.debug(f"Parsed sheet '{sheet_name}': {len(df)} rows x {len(df.columns)} cols")
                    except Exception as e:
                        logger.warning(f"Error parsing sheet '{sheet_name}': {e}")
                        continue
            
            return sheets
            
        except Exception as e:
            logger.error(f"Error parsing Excel directly: {e}")
            return []
    
    def _fetch_file_from_uri(self, uri: str) -> Optional[bytes]:
        """Fetch file bytes from a cloud URI or local path."""
        try:
            if "://" in uri:
                from cloud_lib.factory import CloudFactory

                file_mgr = CloudFactory.get_file_manager()
                return file_mgr.read_bytes(uri)
            else:
                # Local file
                with open(uri, 'rb') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Error fetching file from {uri}: {e}")
            return None
    
    def _dataframe_to_markdown(self, df) -> str:
        """Convert pandas DataFrame to markdown table."""
        try:
            # Clean column names
            df.columns = [str(col).strip() for col in df.columns]
            
            # Convert to markdown
            markdown = df.to_markdown(index=False)
            return markdown if markdown else ""
        except Exception as e:
            logger.warning(f"Error converting DataFrame to markdown: {e}")
            return ""
    
    def _parse_sheets_from_markdown(
        self,
        markdown: str,
        base_metadata: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Parse markdown content to extract individual sheets.
        
        Looks for sheet markers like:
        - ## Sheet: SheetName
        - <!-- sheet: SheetName -->
        
        If no markers found, treats entire content as single sheet.
        """
        sheets = []
        
        # Find all sheet markers
        markers = list(self._SHEET_MARKER_RE.finditer(markdown))
        
        if not markers:
            # No sheet markers - single sheet
            return [{
                "name": self._extract_sheet_name_from_filename(base_metadata),
                "content": markdown,
                "index": 0
            }]
        
        # Extract content between markers
        for i, match in enumerate(markers):
            sheet_name = match.group(1) or match.group(2) or f"Sheet{i + 1}"
            sheet_name = sheet_name.strip()
            
            start_pos = match.end()
            end_pos = markers[i + 1].start() if i + 1 < len(markers) else len(markdown)
            
            content = markdown[start_pos:end_pos].strip()
            
            if content:  # Only add non-empty sheets
                sheets.append({
                    "name": sheet_name,
                    "content": content,
                    "index": i
                })
        
        return sheets
    
    def _extract_sheet_name_from_filename(self, base_metadata: Dict[str, str]) -> str:
        """Extract a reasonable sheet name from file metadata."""
        file_name = base_metadata.get("file_name", "")
        if file_name:
            # Remove extension and use as sheet name
            name = file_name.rsplit(".", 1)[0]
            return name if name else "Sheet1"
        return "Sheet1"
    
    def _process_sheet(
        self,
        file_id: str,
        sheet: Dict[str, Any],
        idx: int,
        base_metadata: Dict[str, str],
    ) -> List[Chunk]:
        """
        Process a single sheet into a parent summary + child table chunks.

        Steps:
          1. Parse markdown table → structured data (headers + rows).
          2. Infer column schema.
          3. Generate parent LLM summary (**parent chunk**).
          4. Slice rows into token-budgeted groups (with overlap).
          5. Emit one child chunk per slice, each a valid markdown table.
          6. Generate per-chunk summaries in metadata.
        """
        sheet_name = sheet["name"]
        content = sheet["content"]
        logger.debug(f"Processing sheet: {sheet_name}")

        # Parse markdown table
        table_data = self._parse_markdown_table(content)

        if not table_data or not table_data.get("rows"):
            logger.warning(f"No table data in sheet {sheet_name}")
            return [self._create_text_chunk(file_id, sheet, idx, base_metadata)]

        headers = table_data["headers"]
        rows = table_data["rows"]
        row_count = table_data["row_count"]

        # Infer column schema
        schema: Dict[str, Dict[str, Any]] = {}
        if self.generate_column_schema:
            schema = self._infer_column_schema(table_data)

        # ── 1. Parent summary chunk ──────────────────────────────
        parent_id = f"{file_id}-sheet-{idx:03d}"
        summary = self._generate_sheet_summary(
            sheet_name=sheet_name,
            table_data=table_data,
            schema=schema,
            file_name=base_metadata.get("file_name", ""),
        )

        # ── 2. Token-based table slicing ─────────────────────────
        slices = self.slice_sheet_rows(headers, rows)
        total_chunks = len(slices)

        # Build parent metadata (needs num_clusters = total_chunks)
        parent_metadata = self._build_parent_metadata(
            sheet=sheet,
            table_data=table_data,
            schema=schema,
            base_metadata=base_metadata,
            markdown_content=content,
            num_clusters=total_chunks,
        )

        chunks: List[Chunk] = []
        chunks.append(Chunk(
            chunk_id=parent_id,
            file_id=file_id,
            content=summary,
            order=0,
            metadata=parent_metadata,
            parent_id=None,
            level="parent",
            embedding_content=summary,
        ))

        # ── 3. Emit child chunks ─────────────────────────────────
        for child_idx, (start, end) in enumerate(slices):
            slice_rows = rows[start:end]
            table_md = self._rows_to_markdown_table(headers, slice_rows)

            chunk_id = f"{file_id}-sheet-{idx:03d}-chunk-{child_idx:03d}"

            # Per-chunk summary
            chunk_summary = self._generate_chunk_summary(
                sheet_name=sheet_name,
                table_md=table_md,
                start_row=start + 1,
                end_row=end,
                total_rows=row_count,
            )

            child_metadata = self._build_child_metadata(
                sheet=sheet,
                headers=headers,
                slice_rows=slice_rows,
                chunk_index=child_idx,
                total_chunks=total_chunks,
                chunk_summary=chunk_summary,
                base_metadata=base_metadata,
            )

            # Embedding content: parent summary context + table chunk
            embedding_text = (
                f"[Sheet: {sheet_name}] {chunk_summary}\n\n{table_md}"
            )

            chunks.append(Chunk(
                chunk_id=chunk_id,
                file_id=file_id,
                content=table_md,
                order=child_idx + 1,  # 0 is parent
                metadata=child_metadata,
                parent_id=parent_id,
                level="child",
                embedding_content=embedding_text,
            ))

        return chunks

    # ── Row-slicing logic ────────────────────────────────────────

    def estimate_rows_per_chunk(
        self,
        headers: List[str],
        rows: List[List[str]],
        max_tokens: Optional[int] = None,
        sample_size: int = 20,
    ) -> Tuple[int, int, float]:
        """Estimate how many rows fit in a single chunk's token budget.

        Samples up to ``sample_size`` rows (evenly spaced through the
        table) to compute an average tokens-per-row, then calculates
        how many rows fit in the remaining budget after headers.

        Args:
            headers: Column header strings.
            rows: All data rows (list of cell lists).
            max_tokens: Token budget per chunk (defaults to ``self.max_tokens``).
            sample_size: Number of rows to sample for the estimate.

        Returns:
            Tuple of ``(rows_per_chunk, header_tokens, avg_tokens_per_row)``.
        """
        if max_tokens is None:
            max_tokens = self.max_tokens

        # Header cost (repeated in every chunk)
        header_line = "| " + " | ".join(headers) + " |"
        separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        header_block = header_line + "\n" + separator_line
        header_tokens = _estimate_tokens(header_block)

        row_budget = max_tokens - header_tokens
        if row_budget < 50:
            row_budget = max(50, max_tokens)

        # Sample rows evenly across the table
        n = len(rows)
        if n == 0:
            return (0, header_tokens, 0.0)

        sample_count = min(sample_size, n)
        step = max(1, n // sample_count)
        sampled_indices = list(range(0, n, step))[:sample_count]

        total_sample_tokens = 0
        for i in sampled_indices:
            row_line = "| " + " | ".join(rows[i]) + " |"
            total_sample_tokens += _estimate_tokens(row_line)

        avg_tokens_per_row = total_sample_tokens / len(sampled_indices)

        # Rows that fit in the budget (at least 1)
        rows_per_chunk = max(1, int(row_budget / avg_tokens_per_row))

        logger.debug(
            f"Token estimate: header={header_tokens}, "
            f"avg_row={avg_tokens_per_row:.1f}, "
            f"rows_per_chunk={rows_per_chunk} "
            f"(budget {max_tokens})"
        )

        return (rows_per_chunk, header_tokens, avg_tokens_per_row)

    def slice_sheet_rows(
        self,
        headers: List[str],
        rows: List[List[str]],
        max_tokens: Optional[int] = None,
        overlap_tokens: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """Break sheet rows into slices that each fit within the token budget.

        Uses ``estimate_rows_per_chunk`` to determine rows-per-slice, then
        partitions the full row list into ``(start, end)`` ranges with an
        overlap computed from the token budget.

        Args:
            headers: Column header strings.
            rows: All data rows.
            max_tokens: Token budget per chunk (defaults to ``self.max_tokens``).
            overlap_tokens: Overlap budget in tokens (defaults to ``self.overlap_tokens``).

        Returns:
            List of ``(start_row_inclusive, end_row_exclusive)`` tuples.
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
        if overlap_tokens is None:
            overlap_tokens = self.overlap_tokens

        n = len(rows)
        if n == 0:
            return []

        rows_per_chunk, header_tokens, avg_tokens_per_row = (
            self.estimate_rows_per_chunk(headers, rows, max_tokens)
        )

        if rows_per_chunk >= n:
            return [(0, n)]

        # Overlap expressed as number of rows
        overlap_rows = 0
        if overlap_tokens > 0 and avg_tokens_per_row > 0:
            overlap_rows = max(0, int(overlap_tokens / avg_tokens_per_row))

        # Step forward = rows_per_chunk minus overlap
        step = max(1, rows_per_chunk - overlap_rows)

        slices: List[Tuple[int, int]] = []
        pos = 0
        while pos < n:
            end = min(pos + rows_per_chunk, n)
            slices.append((pos, end))
            pos += step

        # Avoid a tiny trailing slice by merging into the last full one
        if len(slices) > 1:
            last_start, last_end = slices[-1]
            if (last_end - last_start) < max(1, rows_per_chunk // 3):
                slices.pop()
                prev_start, _ = slices[-1]
                slices[-1] = (prev_start, last_end)

        logger.info(
            f"slice_sheet_rows: {n} rows → {len(slices)} slices "
            f"(rows_per_chunk={rows_per_chunk}, overlap_rows={overlap_rows}, "
            f"avg_tok/row={avg_tokens_per_row:.1f}, budget={max_tokens})"
        )

        return slices
    
    def _parse_markdown_table(self, content: str) -> Dict[str, Any]:
        """
        Parse a markdown table into structured data.
        
        Scans all tables in the content (separated by empty or non-table
        lines) and returns the **largest** one.  This handles Docling output
        for spreadsheets where a small summary table may precede the real
        data table.
        
        Returns:
            {
                "headers": ["col1", "col2", ...],
                "rows": [["val1", "val2"], ...],
                "row_count": int,
                "col_count": int,
                "markdown": str  # Original markdown
            }
        """
        lines = content.strip().split("\n")
        
        # Collect ALL tables in the content
        all_tables: List[List[str]] = []
        current_table: List[str] = []
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                current_table.append(stripped)
            else:
                if current_table:
                    all_tables.append(current_table)
                    current_table = []
        if current_table:
            all_tables.append(current_table)
        
        if not all_tables:
            return {"headers": [], "rows": [], "row_count": 0, "col_count": 0, "markdown": content}
        
        # Pick the table with the most lines (the actual data table)
        table_lines = max(all_tables, key=len)
        
        if len(table_lines) < 2:
            return {"headers": [], "rows": [], "row_count": 0, "col_count": 0, "markdown": content}
        
        # Parse headers (first line)
        headers = self._parse_table_row(table_lines[0])
        
        # Skip separator line (second line with ---)
        data_start = 1
        if len(table_lines) > 1 and self._TABLE_SEP_RE.match(table_lines[1]):
            data_start = 2
        
        # Parse data rows
        rows = []
        for line in table_lines[data_start:]:
            row = self._parse_table_row(line)
            if row:
                # Pad or trim to match header count
                while len(row) < len(headers):
                    row.append("")
                row = row[:len(headers)]
                rows.append(row)
        
        # Reconstruct clean markdown
        markdown_table = "\n".join(table_lines)
        
        logger.debug(
            f"_parse_markdown_table: found {len(all_tables)} table(s), "
            f"selected largest with {len(rows)} rows x {len(headers)} cols"
        )
        
        return {
            "headers": headers,
            "rows": rows,
            "row_count": len(rows),
            "col_count": len(headers),
            "markdown": markdown_table
        }
    
    def _parse_table_row(self, line: str) -> List[str]:
        """Parse a single markdown table row into cells."""
        # Remove leading/trailing pipes and split
        line = line.strip()
        if line.startswith("|"):
            line = line[1:]
        if line.endswith("|"):
            line = line[:-1]
        
        cells = [cell.strip() for cell in line.split("|")]
        return cells
    
    def _infer_column_schema(self, table_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Infer column types and statistics from table data.
        
        Returns schema like:
        {
            "column_name": {
                "type": "numeric|string|date|categorical|boolean",
                "sample": "example value",
                "unique_count": int,
                "null_count": int,
                "min": value (for numeric),
                "max": value (for numeric),
                "unique_values": [...] (for categorical with few values)
            }
        }
        """
        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])
        
        if not headers or not rows:
            return {}
        
        schema = {}
        
        for col_idx, header in enumerate(headers):
            if not header:
                continue
                
            # Collect all values for this column
            values = []
            for row in rows:
                if col_idx < len(row):
                    val = row[col_idx]
                    if val and val.strip():
                        values.append(val.strip())
            
            if not values:
                schema[header] = {"type": "unknown", "null_count": len(rows)}
                continue
            
            # Infer type
            col_schema = self._infer_column_type(values)
            col_schema["sample"] = values[0] if values else ""
            col_schema["null_count"] = len(rows) - len(values)
            
            schema[header] = col_schema
        
        return schema
    
    def _infer_column_type(self, values: List[str]) -> Dict[str, Any]:
        """Infer the type of a column from its values."""
        if not values:
            return {"type": "unknown"}
        
        # Sample for analysis (max 1000 values)
        sample = values[:1000]
        
        # Check for boolean
        bool_values = {"true", "false", "yes", "no", "1", "0", "y", "n"}
        if all(v.lower() in bool_values for v in sample):
            return {"type": "boolean"}
        
        # Check for numeric
        numeric_values = []
        numeric_count = 0
        for v in sample:
            try:
                # Handle currency symbols, commas, percentages
                cleaned = re.sub(r"[$,€£%]", "", v).strip()
                if cleaned:
                    num = float(cleaned)
                    numeric_values.append(num)
                    numeric_count += 1
            except ValueError:
                pass
        
        if numeric_count >= len(sample) * 0.8:  # 80% numeric
            return {
                "type": "numeric",
                "min": min(numeric_values) if numeric_values else None,
                "max": max(numeric_values) if numeric_values else None,
                "unique_count": len(set(numeric_values))
            }
        
        # Check for date patterns
        date_patterns = [
            r"\d{4}-\d{2}-\d{2}",  # YYYY-MM-DD
            r"\d{2}/\d{2}/\d{4}",  # MM/DD/YYYY
            r"\d{2}-\d{2}-\d{4}",  # DD-MM-YYYY
            r"\d{1,2}/\d{1,2}/\d{2,4}",  # M/D/YY or MM/DD/YYYY
        ]
        date_count = 0
        for v in sample:
            for pattern in date_patterns:
                if re.match(pattern, v.strip()):
                    date_count += 1
                    break
        
        if date_count >= len(sample) * 0.8:
            return {"type": "date"}
        
        # Check for categorical (few unique values)
        unique_values = list(set(sample))
        if len(unique_values) <= 20 and len(unique_values) <= len(sample) * 0.5:
            return {
                "type": "categorical",
                "unique_count": len(unique_values),
                "unique_values": unique_values[:10]  # Max 10 values
            }
        
        # Default to string
        return {
            "type": "string",
            "unique_count": len(set(sample))
        }
    
    def _generate_sheet_summary(
        self,
        sheet_name: str,
        table_data: Dict[str, Any],
        schema: Dict[str, Dict[str, Any]],
        file_name: str = ""
    ) -> str:
        """
        Generate a comprehensive summary for semantic search.
        
        Uses LLM if available, otherwise generates a structured summary.
        """
        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])
        row_count = table_data.get("row_count", 0)
        col_count = table_data.get("col_count", 0)
        
        # Build sample rows for prompt
        sample_rows = rows[:self.sample_rows_for_summary]
        sample_md = self._rows_to_markdown_table(headers, sample_rows)
        
        # Build additional stats from schema
        stats_lines = []
        for col, col_schema in schema.items():
            col_type = col_schema.get("type", "unknown")
            if col_type == "numeric":
                min_val = col_schema.get("min")
                max_val = col_schema.get("max")
                if min_val is not None and max_val is not None:
                    stats_lines.append(f"- {col}: range {min_val} to {max_val}")
            elif col_type == "categorical":
                unique_vals = col_schema.get("unique_values", [])
                if unique_vals:
                    stats_lines.append(f"- {col}: categories include {', '.join(str(v) for v in unique_vals[:5])}")
            elif col_type == "date":
                stats_lines.append(f"- {col}: date field")
        
        additional_stats = "\n".join(stats_lines) if stats_lines else "- No additional statistics available"
        
        # Try LLM generation
        llm = self._get_llm()
        if llm:
            try:
                prompt = self.SUMMARY_PROMPT.format(
                    sheet_name=sheet_name,
                    file_name=file_name,
                    columns=", ".join(headers),
                    col_count=col_count,
                    sample_rows=sample_md,
                    sample_count=len(sample_rows),
                    row_count=row_count,
                    additional_stats=additional_stats
                )
                
                response = llm.generate(prompt, max_tokens=500)
                if response:
                    summary = response.text if hasattr(response, 'text') else str(response)
                    logger.debug(f"Generated LLM summary for sheet {sheet_name}")
                    return summary.strip()
            except Exception as e:
                logger.warning(f"LLM summary generation failed: {e}")
        
        # Fallback: Generate structured summary without LLM
        return self._generate_fallback_summary(
            sheet_name=sheet_name,
            headers=headers,
            row_count=row_count,
            col_count=col_count,
            schema=schema,
            file_name=file_name
        )
    
    def _generate_fallback_summary(
        self,
        sheet_name: str,
        headers: List[str],
        row_count: int,
        col_count: int,
        schema: Dict[str, Dict[str, Any]],
        file_name: str = ""
    ) -> str:
        """Generate a structured summary without LLM."""
        lines = [f"[Sheet: {sheet_name}]"]
        
        if file_name:
            lines.append(f"From file: {file_name}")
        
        lines.append(f"This spreadsheet contains {row_count} rows and {col_count} columns of data.")
        lines.append("")
        lines.append("Columns:")
        
        for header in headers:
            col_schema = schema.get(header, {})
            col_type = col_schema.get("type", "string")
            sample = col_schema.get("sample", "")
            
            type_desc = col_type
            if col_type == "categorical":
                unique_vals = col_schema.get("unique_values", [])
                if unique_vals:
                    type_desc = f"categorical ({', '.join(str(v) for v in unique_vals[:3])}...)"
            elif col_type == "numeric":
                min_val = col_schema.get("min")
                max_val = col_schema.get("max")
                if min_val is not None:
                    type_desc = f"numeric (range: {min_val} - {max_val})"
            
            lines.append(f"- {header}: {type_desc}")
        
        lines.append("")
        lines.append(f"Total Records: {row_count}")
        
        return "\n".join(lines)
    
    def _rows_to_markdown_table(
        self,
        headers: List[str],
        rows: List[List[str]]
    ) -> str:
        """Convert headers and rows back to markdown table format."""
        if not headers:
            return ""
        
        lines = []
        
        # Header row
        lines.append("| " + " | ".join(headers) + " |")
        
        # Separator row
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        # Data rows
        for row in rows:
            # Pad row to match header count
            padded = row + [""] * (len(headers) - len(row))
            lines.append("| " + " | ".join(padded[:len(headers)]) + " |")
        
        return "\n".join(lines)
    
    # ── Metadata builders ──────────────────────────────────────────

    def _build_parent_metadata(
        self,
        sheet: Dict[str, Any],
        table_data: Dict[str, Any],
        schema: Dict[str, Dict[str, Any]],
        base_metadata: Dict[str, str],
        markdown_content: str,
        num_clusters: int = 1,
    ) -> Dict[str, str]:
        """Build metadata for the parent summary chunk.

        The parent chunk is a pure summary for semantic search.
        Data records are stored in child chunks instead.
        """
        metadata = dict(base_metadata)

        sheet_name = sheet.get("name", "Sheet1")
        headers = table_data.get("headers", [])
        row_count = table_data.get("row_count", 0)
        col_count = table_data.get("col_count", 0)

        metadata["sheet_name"] = sheet_name
        metadata["sheet_index"] = str(sheet.get("index", 0))
        metadata["content_type"] = "spreadsheet"
        metadata["chunking_strategy"] = "semantic_row_grouping"

        # ── has_table (markdown-first) ──
        metadata["has_table"] = "true"
        metadata["table_row_count"] = str(row_count)
        metadata["table_col_count"] = str(col_count)

        if headers:
            metadata["chunk_keywords"] = ",".join(headers)

        # Parent does not carry data — data lives in child chunks
        metadata["has_data"] = "false"
        metadata["num_clusters"] = str(num_clusters)

        if schema:
            metadata["column_schema"] = json.dumps(schema)

        return metadata

    def _build_child_metadata(
        self,
        sheet: Dict[str, Any],
        headers: List[str],
        slice_rows: List[List[str]],
        chunk_index: int,
        total_chunks: int,
        chunk_summary: str,
        base_metadata: Dict[str, str],
    ) -> Dict[str, str]:
        """Build metadata for a child table-slice chunk."""
        metadata = dict(base_metadata)

        sheet_name = sheet.get("name", "Sheet1")
        metadata["sheet_name"] = sheet_name
        metadata["sheet_index"] = str(sheet.get("index", 0))
        metadata["content_type"] = "spreadsheet"
        metadata["chunking_strategy"] = "semantic_row"
        metadata["has_table"] = "true"
        metadata["table_row_count"] = str(len(slice_rows))
        metadata["table_col_count"] = str(len(headers))
        metadata["chunk_index"] = str(chunk_index)
        metadata["total_chunks"] = str(total_chunks)
        metadata["summary"] = chunk_summary

        if headers:
            metadata["chunk_keywords"] = ",".join(headers)

        # ── has_data (JSON records for this chunk) ──
        records = []
        for row in slice_rows:
            record = {}
            for i, header in enumerate(headers):
                record[header] = row[i] if i < len(row) else ""
            records.append(record)

        data_envelope = {
            "tables": [{
                "sheet_name": sheet_name,
                "chunk_index": chunk_index,
                "format": "records",
                "headers": headers,
                "records": records,
            }],
        }
        metadata["has_data"] = "true"
        metadata["data"] = json.dumps(data_envelope)

        return metadata

    # ── Per-chunk summary generation ─────────────────────────────

    def _generate_chunk_summary(
        self,
        sheet_name: str,
        table_md: str,
        start_row: int,
        end_row: int,
        total_rows: int,
    ) -> str:
        """Generate a short summary for a child table chunk via LLM (or fallback)."""
        llm = self._get_llm()
        if llm:
            try:
                prompt = self.CHUNK_SUMMARY_PROMPT.format(
                    sheet_name=sheet_name,
                    start_row=start_row,
                    end_row=end_row,
                    total_rows=total_rows,
                    table_chunk=table_md[:3000],  # cap prompt length
                )
                response = llm.generate(prompt, max_tokens=120)
                if response:
                    text = response.text if hasattr(response, 'text') else str(response)
                    return text.strip()
            except Exception as e:
                logger.warning(f"Chunk summary LLM call failed: {e}")

        # Fallback
        return (
            f"[Sheet: {sheet_name}] Rows {start_row}-{end_row} of {total_rows}."
        )
    
    def _create_text_chunk(
        self,
        file_id: str,
        sheet: Dict[str, Any],
        idx: int,
        base_metadata: Dict[str, str]
    ) -> Chunk:
        """Create a basic text chunk when no table is detected."""
        sheet_name = sheet.get("name", f"Sheet{idx + 1}")
        content = sheet.get("content", "")
        
        metadata = dict(base_metadata)
        metadata["sheet_name"] = sheet_name
        metadata["sheet_index"] = str(idx)
        metadata["content_type"] = "spreadsheet_text"
        metadata["has_table"] = "false"
        metadata["has_data"] = "false"
        
        summary = f"[Sheet: {sheet_name}]\n{content[:500]}"
        
        return Chunk(
            chunk_id=f"{file_id}-sheet-{idx:03d}",
            file_id=file_id,
            content=summary,
            order=idx,
            metadata=metadata,
            parent_id=None,
            level="chunk"
        )
    
    def _create_fallback_chunk(
        self,
        file_id: str,
        sheet: Dict[str, Any],
        idx: int,
        base_metadata: Dict[str, str]
    ) -> Chunk:
        """Create a fallback chunk when processing fails."""
        sheet_name = sheet.get("name", f"Sheet{idx + 1}")
        content = sheet.get("content", "")[:2000]  # Truncate for safety
        
        metadata = dict(base_metadata)
        metadata["sheet_name"] = sheet_name
        metadata["sheet_index"] = str(idx)
        metadata["content_type"] = "spreadsheet"
        metadata["has_table"] = "true"
        metadata["has_data"] = "false"
        metadata["processing_error"] = "true"
        
        summary = f"[Sheet: {sheet_name}]\nSpreadsheet data (processing incomplete)\n\n{content}"
        
        return Chunk(
            chunk_id=f"{file_id}-sheet-{idx:03d}",
            file_id=file_id,
            content=summary,
            order=idx,
            metadata=metadata,
            parent_id=None,
            level="chunk"
        )
