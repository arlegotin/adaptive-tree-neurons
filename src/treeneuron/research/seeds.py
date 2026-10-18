"""Canonical v2 namespaces, independent of the legacy generator's inner seeds."""
import hashlib
import json


def seed64(root: int, *labels: str) -> int:
    if type(root) is not int or not 0 <= root < 2**63 or any(type(v) is not str for v in labels):
        raise ValueError('Seed root must be a63-bit integer and labels strings')
    payload = json.dumps([root, *labels], separators=(',', ':'), ensure_ascii=True).encode('ascii')
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') & ((1 << 63)-1)


def seed32(root: int, *labels: str, used: set[int]) -> tuple[int, int]:
    """Allocate an independent32-bit seed; intentional pairing reuses its record."""
    counter = 0
    while True:
        value = seed64(root, *labels, 'library32', str(counter)) & ((1 << 32)-1)
        if value not in used:
            used.add(value)
            return value, counter
        counter += 1
