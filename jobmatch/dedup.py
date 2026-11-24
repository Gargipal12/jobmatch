"""Near-duplicate detection.

The same job often appears several times: re-posted, or copied by a staffing
agency with a few words changed. Exact-match comparison misses these, so we
compare overlap instead.

Method: turn each description into a set of word 3-grams ("shingles") and
compare sets with Jaccard similarity = |A and B| / |A or B|. Identical text
scores 1.0; one changed word in a 20-word text still scores about 0.75.

This is O(n^2) pairs, fine for thousands of jobs. For millions you would use
MinHash + locality-sensitive hashing to avoid comparing every pair.
"""
import re

from .data import Job

_WORD = re.compile(r"[a-z0-9]+")


def shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = _WORD.findall(text.lower())
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_duplicate_groups(jobs: list[Job], threshold: float = 0.6) -> list[list[Job]]:
    """Groups of jobs whose descriptions overlap by at least `threshold`.

    Uses union-find so that if A~B and B~C, all three land in one group.
    """
    sh = [shingles(j.description) for j in jobs]
    parent = list(range(len(jobs)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(jobs)):
        for j in range(i + 1, len(jobs)):
            if jaccard(sh[i], sh[j]) >= threshold:
                parent[find(i)] = find(j)

    groups: dict[int, list[Job]] = {}
    for i, job in enumerate(jobs):
        groups.setdefault(find(i), []).append(job)
    return [g for g in groups.values() if len(g) > 1]


def choose_canonical(group: list[Job]) -> Job:
    """Which copy to keep: prefer a direct employer over an agency ("via"),
    then the freshest posting."""
    return min(group, key=lambda j: (" via " in j.company.lower(), j.posted_days_ago))


def canonical_map(jobs: list[Job], threshold: float = 0.6) -> dict[str, str]:
    """job id -> id of the listing we keep. Unique jobs map to themselves."""
    mapping = {j.id: j.id for j in jobs}
    for group in find_duplicate_groups(jobs, threshold):
        keep = choose_canonical(group)
        for j in group:
            mapping[j.id] = keep.id
    return mapping
