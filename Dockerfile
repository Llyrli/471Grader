# JN Grader — containerized grading pipeline.
#
# Running student notebooks inside a container provides the execution isolation
# that architecture.md §12 lists as a roadmap item: untrusted student code runs
# with the container's (restricted) privileges, not the host's.
#
# Build:  docker build -t jn-grader .
# Run:    docker run --rm -v "$PWD/workspace:/app/workspace" jn-grader \
#             python scripts/preprocess.py workspace/raw --output workspace/processed
FROM python:3.11-slim

# Non-interactive matplotlib / Jupyter
ENV MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    JUPYTER_PLATFORM_DIRS=1

WORKDIR /app

# Install Python deps first for better layer caching
COPY scripts/requirements.txt scripts/requirements.txt
RUN pip install --no-cache-dir -r scripts/requirements.txt \
    && python -m ipykernel install --name python3 2>/dev/null || true

# Copy the grader (workspace is mounted as a volume at run time)
COPY scripts/ scripts/
COPY tests/ tests/
COPY templates/ templates/

# Run untrusted student code as a non-root user
RUN useradd --create-home grader && chown -R grader:grader /app
USER grader

CMD ["python", "scripts/run_tests.py", "--help"]
