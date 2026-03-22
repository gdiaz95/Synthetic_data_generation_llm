from __future__ import annotations

import traceback
from typing import Callable

import Airbnb
import Kickstarter
import consumer_complaint
import fake_job_postings
import harvard_reviews
import student_satisfaction

Processor = Callable[[], object]

PROCESSORS: list[tuple[str, Processor]] = [
    ("Airbnb", Airbnb.main),
    ("Kickstarter", Kickstarter.main),
    ("consumer_complaint", consumer_complaint.main),
    ("fake_job_postings", fake_job_postings.main),
    ("harvard_reviews", harvard_reviews.main),
    ("student_satisfaction", student_satisfaction.main),
]


def main() -> None:
    failures: list[str] = []

    for name, processor in PROCESSORS:
        print(f"=== Running {name} ===", flush=True)
        try:
            processor()
        except FileNotFoundError as exc:
            failures.append(name)
            print(f"[{name}] Failed: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(name)
            print(f"[{name}] Failed with unexpected error: {exc}")
            traceback.print_exc()

    if failures:
        failed = ", ".join(failures)
        raise SystemExit(f"Pre-processing failed for: {failed}")

    print("All pre-processing scripts completed successfully.")


if __name__ == "__main__":
    main()
