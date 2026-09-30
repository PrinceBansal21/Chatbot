import logging
from typing import List, Optional
from rag_lib.interfaces.pii_manager import PIIManager

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

class PresidioAdapter(PIIManager):
    def __init__(self, config: dict = None):
        if not PRESIDIO_AVAILABLE:
            raise ImportError("Microsoft Presidio is not installed. Please install 'presidio-analyzer' and 'presidio-anonymizer'.")
        
        self.config = config or {}
        # Initialize engines
        # Note: AnalyzerEngine loads the NLP model (e.g., en_core_web_lg) which takes time and memory.
        # It should be initialized once.
        try:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
        except Exception as e:
            LOGGER.error(f"Failed to initialize Presidio engines: {e}")
            raise

        self.default_entities = self.config.get("entities", None) # None means all
        self.language = self.config.get("language", "en")

    def redact(self, text: str, **kwargs) -> str:
        if not text:
            return ""

        entities = kwargs.get("entities", self.default_entities)
        language = kwargs.get("language", self.language)

        try:
            results = self.analyzer.analyze(text=text, entities=entities, language=language)
            anonymized_result = self.anonymizer.anonymize(text=text, analyzer_results=results)
            return anonymized_result.text
        except Exception as e:
            LOGGER.error(f"Error during PII redaction: {e}")
            # Fail safe: return original text or raise? 
            # Usually better to fail safe in a pipeline if PII isn't critical blocker, 
            # but for PII compliance, maybe we should raise. 
            # Given the user request, I'll log and return original text to avoid crashing the pipeline, 
            # but ideally this should be configurable.
            return text
