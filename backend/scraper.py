"""Collect text from Reddit (PRAW). Replace subreddit / limits as needed."""

from __future__ import annotations

import os
from typing import Iterable

import praw
from dotenv import load_dotenv

load_dotenv()


def reddit_client() -> praw.Reddit:
    cid = os.getenv("REDDIT_CLIENT_ID", "").strip()
    sec = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
    if not cid or not sec or "your_" in cid:
        raise RuntimeError(
            "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in backend/.env "
            "(Reddit app: https://www.reddit.com/prefs/apps)"
        )
    return praw.Reddit(
        client_id=cid,
        client_secret=sec,
        user_agent=os.getenv(
            "REDDIT_USER_AGENT",
            "review-intelligence-pipeline/0.1 by u/local-dev",
        ),
    )


def iter_submission_bodies(
    subreddit: str,
    submission_limit: int = 25,
    max_comments_per_thread: int = 50,
) -> Iterable[str]:
    """Yield comment body strings from recent hot posts in a subreddit."""
    r = reddit_client()
    sub = r.subreddit(subreddit)
    for submission in sub.hot(limit=submission_limit):
        submission.comments.replace_more(limit=0)
        count = 0
        for c in submission.comments.list():
            if count >= max_comments_per_thread:
                break
            body = getattr(c, "body", None)
            if body and isinstance(body, str) and body not in ("[deleted]", "[removed]"):
                yield body.strip()
                count += 1


def fetch_texts(
    subreddit: str,
    submission_limit: int = 25,
    max_comments_per_thread: int = 50,
) -> list[str]:
    return list(
        iter_submission_bodies(
            subreddit,
            submission_limit=submission_limit,
            max_comments_per_thread=max_comments_per_thread,
        )
    )


if __name__ == "__main__":
    import sys

    sub = sys.argv[1] if len(sys.argv) > 1 else "technology"
    texts = fetch_texts(sub, submission_limit=5, max_comments_per_thread=20)
    for t in texts[:10]:
        print(t[:200].replace("\n", " "), "…")
