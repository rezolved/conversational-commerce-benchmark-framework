from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from retrieval.txtai_index import hashing_transform


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DeterministicHashingTests(unittest.TestCase):
    def test_same_input_is_stable_in_process(self) -> None:
        texts = ["Black straight leg jeans", "Blue relaxed jeans"]
        first = hashing_transform(texts, dimensions=32)
        second = hashing_transform(texts, dimensions=32)
        self.assertTrue((first == second).all())

    def test_vectors_match_across_python_hash_seeds(self) -> None:
        program = (
            "import json; "
            "from retrieval.txtai_index import hashing_transform; "
            "print(json.dumps(hashing_transform("
            "['Black straight leg jeans', 'Blue relaxed jeans'], "
            "dimensions=32).tolist()))"
        )

        outputs: list[list[list[float]]] = []
        for seed in ("1", "8675309"):
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = seed
            env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
            raw = subprocess.check_output(
                [sys.executable, "-c", program],
                cwd=PROJECT_ROOT,
                env=env,
                text=True,
            )
            outputs.append(json.loads(raw))

        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
