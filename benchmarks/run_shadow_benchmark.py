"""python benchmarks/run_shadow_benchmark.py [--json] [--quick]  (== shadow-learn benchmark)"""

import sys

from shadow_learning.benchmark import run_shadow_benchmark

if __name__ == "__main__":
    sys.exit(run_shadow_benchmark(as_json="--json" in sys.argv, quick="--quick" in sys.argv))
