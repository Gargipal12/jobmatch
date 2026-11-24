"""Loading jobs and eval labels from disk."""
import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Job:
    id: str
    title: str
    company: str
    location: str
    posted_days_ago: int
    description: str

    @property
    def text(self) -> str:
        """The text we index and embed for this job."""
        return f"{self.title}. {self.description}"


def load_jobs(path) -> list[Job]:
    with open(Path(path), newline="", encoding="utf-8") as f:
        return [
            Job(
                id=row["id"],
                title=row["title"],
                company=row["company"],
                location=row["location"],
                posted_days_ago=int(row["posted_days_ago"]),
                description=row["description"],
            )
            for row in csv.DictReader(f)
        ]


def load_eval(path) -> list[dict]:
    """Each item: {"id", "query", "relevant": {job_id: grade}}."""
    with open(Path(path), encoding="utf-8") as f:
        return json.load(f)["queries"]
