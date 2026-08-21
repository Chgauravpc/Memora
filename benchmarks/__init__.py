"""
LoCoMo benchmark harness for Memora.

Self-contained: every path this package reads or writes is anchored under the repository
root by benchmarks.paths, including the HuggingFace model cache and the Redis/Qdrant
volumes, so the whole operation stays inside /home/kenton/projects/memora on the server.

Entry points:
    python -m benchmarks.download          # fetch locomo10.json
    python -m benchmarks.dataset --inspect # verify dataset schema
    python -m benchmarks.preflight         # check deps, backends, keys, paths
    python -m benchmarks.estimate          # cost/time projection
    python run_locomo.py --workers 12      # run
    python -m benchmarks.report            # scorecard
"""

__all__ = ["paths", "dataset", "download", "llm", "qa", "worker", "runner", "report"]
