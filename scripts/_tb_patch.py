"""Import for side effect: make HF-`datasets` fingerprinting cheap.

TurnBench's `resolve_dataset` wraps a concatenated Arrow table in a HF `Dataset`
(constructor, not `load_dataset`, precisely to avoid `combine_chunks()`). But
`datasets>=5` computes the dataset fingerprint in `Dataset.__init__` by pickling
the table, and its dill hook calls `combine_chunks()` — which overflows the 32-bit
offsets of the embedded `binary` audio column on the full ~13 GB test split
(`ArrowInvalid: offset overflow while concatenating arrays`). The dev split is
small enough to slip under 2 GB, so this only bites on test.

We only ever read the table (predict / score) — the fingerprint's cache-identity
role is irrelevant here — so replace `generate_fingerprint` with a deterministic
schema+row-count hash that never materializes the audio. `cast_column` afterwards
uses `update_fingerprint` on this string (no table hashing), so it's unaffected.
"""
import hashlib

import datasets.arrow_dataset as _ad


def _cheap_fingerprint(dataset) -> str:
    table = dataset._data
    key = f"{table.schema}|{table.num_rows}".encode()
    return hashlib.sha1(key).hexdigest()[:16]


_ad.generate_fingerprint = _cheap_fingerprint
