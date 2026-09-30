# Use a public container registry base image
# Use full python:3.12 (not slim) because LibreOffice requires additional dependencies
# that are stripped from the slim image and would cause "exec format error" at runtime
FROM python:3.12

WORKDIR /app

# Build argument for cloud provider (aws or azure)
ARG CLOUD_PROVIDER=aws

# Pin apt package resolution to a Debian snapshot timestamp for reproducible
# dependency versions and reduced supply-chain drift during rebuilds.
ARG DEBIAN_SNAPSHOT=20260901T000000Z

# Build argument for GitHub Personal Access Token (for private repos)
# This is passed securely from the build service via --build-arg
ARG GITHUB_TOKEN=""

# Enable BuildKit inline cache for --cache-from support
ARG BUILDKIT_INLINE_CACHE=1

# Build version stamp (e.g. dev-20251110123045-abc123def456) - injected by CI
# so the image carries provenance metadata for traceability.
ARG BUILD_VERSION="local"
ENV BUILD_VERSION=${BUILD_VERSION}
LABEL org.opencontainers.image.version="${BUILD_VERSION}" \
      org.opencontainers.image.source="https://github.com/BPS-SmartOps/win-application-suite" \
      org.opencontainers.image.title="win-data-ingestion-worker"

# Install system dependencies including build tools for compiling Python packages
# This layer is cached unless the base image or these packages change
RUN set -eux; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i "s|http://deb.debian.org/debian-security|http://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}|g" /etc/apt/sources.list.d/debian.sources; \
        sed -i "s|http://deb.debian.org/debian|http://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}|g" /etc/apt/sources.list.d/debian.sources; \
        sed -i "s|http://security.debian.org/debian-security|http://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}|g" /etc/apt/sources.list.d/debian.sources; \
    fi; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i "s|http://deb.debian.org/debian-security|http://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}|g" /etc/apt/sources.list; \
        sed -i "s|http://deb.debian.org/debian|http://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}|g" /etc/apt/sources.list; \
        sed -i "s|http://security.debian.org/debian-security|http://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}|g" /etc/apt/sources.list; \
    fi; \
    printf 'Acquire::Check-Valid-Until "false";\n' > /etc/apt/apt.conf.d/99snapshot; \
    apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libreoffice-impress \
    libreoffice-writer \
    fontconfig \
    fonts-crosextra-carlito \
    fonts-crosextra-caladea \
    fonts-liberation \
    fonts-noto-core \
    fonts-noto-cjk \
    gcc \
    g++ \
    build-essential \
    git \
    && fc-cache -f -v \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files (separate COPY per file for better cache invalidation)
COPY requirements.txt .
COPY requirements-aws.txt requirements-azure.txt ./

# Configure git to use the GitHub token for private repos (if provided)
# This allows pip to clone private GitHub repositories
RUN if [ -n "$GITHUB_TOKEN" ]; then \
        git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"; \
    fi

# Install base requirements and exactly one cloud provider distribution.
# This layer is cached as long as requirements*.txt files don't change
RUN case "$CLOUD_PROVIDER" in \
        aws|azure) ;; \
        *) echo "Unsupported CLOUD_PROVIDER=$CLOUD_PROVIDER; expected aws or azure"; exit 1 ;; \
    esac && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r "requirements-${CLOUD_PROVIDER}.txt"

# Clear the git config after pip install to avoid leaking token in image layers
RUN git config --global --unset-all url."https://${GITHUB_TOKEN}@github.com/".insteadOf 2>/dev/null || true

# Download spacy model only if spacy is installed (used for PII detection)
RUN python -c "import spacy; exit(0)" 2>/dev/null && python -m spacy download en_core_web_lg || echo "Spacy not installed, skipping model download"

# Copy and install rag_lib first (changes less often than app code)
COPY rag_lib ./rag_lib
RUN pip install --no-cache-dir --no-deps ./rag_lib

# Pre-download Docling ML models (layout, tableformer, code_formula, picture_classifier, rapidocr, easyocr)
# so they are baked into the image and NOT fetched from HuggingFace at runtime.
# DOCLING_CACHE_DIR sets the Docling settings.cache_dir (via pydantic_settings env_prefix="DOCLING_")
# DOCLING_ARTIFACTS_PATH tells Docling's pipeline to use pre-downloaded models instead of HuggingFace Hub
ENV DOCLING_CACHE_DIR=/app/.cache/docling
ENV DOCLING_ARTIFACTS_PATH=/app/.cache/docling/models
RUN docling-tools models download -o "$DOCLING_ARTIFACTS_PATH" layout tableformer code_formula picture_classifier rapidocr easyocr

# Pre-download RapidOCR ONNX models (det, cls, rec) so they are NOT fetched from modelscope.cn at runtime.
# Uses the package's built-in download_models command with its default config.
RUN python -c "\
from rapidocr.utils.download_models import download_models; \
from pathlib import Path; \
import rapidocr; \
pkg_dir = Path(rapidocr.__file__).parent; \
download_models(pkg_dir / 'config.yaml')" \
    && echo "RapidOCR models pre-downloaded successfully"

# Copy application code last (changes most often, invalidates fewest layers)
COPY app.py .
COPY utils ./utils
COPY services ./services
COPY models ./models
COPY worker.py .

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Disable Docling torch model compilation — very slow on CPU-only containers.
# Models still work fine without compilation, just skips the JIT optimization step.
ENV DOCLING_INFERENCE_COMPILE_TORCH_MODELS=false

# Cloud provider configuration - automatically set from build argument
# This determines which cloud services (storage, queue, secrets, vectordb, database) are used
ENV CLOUD_PROVIDER=${CLOUD_PROVIDER}

# APP_ENV and CLOUD_REGION must be supplied by the runtime task definition.

USER appuser

CMD ["python", "worker.py"]
