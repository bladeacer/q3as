"""One-off: re-pack generated JSONLs to full-project generated_solution.

Old packer wrote generated_solution = {basename: code}; ada-eval's BUILD and
PROVE evals run gprbuild/gnatformat/gnatprove inside the unpacked
generated_solution, which needs the whole project (main.gpr, src/, main.adc).
This rewrites every outputs/generated_solutions/*/*.jsonl as

    generated_solution = sources | {location.path: model_code}

Backs up each original file to <name>.jsonl.bak before rewriting.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("outputs/generated_solutions")


def main() -> None:
    for f in sorted(ROOT.glob("*/*.jsonl")):
        bak = f.with_suffix(".jsonl.bak")
        if not bak.exists():
            bak.write_bytes(f.read_bytes())
        out: list[str] = []
        for line in bak.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sample = json.loads(line)
            gen = sample.pop("generated_solution", {})
            code = next(iter(gen.values())) if gen else None
            if code is not None:
                sample["generated_solution"] = {
                    **sample["sources"],
                    sample["location"]["path"]: code,
                }
                out.append(json.dumps(sample))
        f.write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"{f}: repacked {len(out)} samples")


if __name__ == "__main__":
    main()
