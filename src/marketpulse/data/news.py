"""Collect news headlines and central bank statements.

All sources end up in one table with the same columns:
    date (datetime, day only), title, source, url, text

Only headlines and public statements are stored. Do not scrape full articles:
most news sites do not allow it.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

NEWS_COLUMNS = ["date", "title", "source", "url", "text"]


def fetch_gdelt(keywords: list[str], start: str, end: str, chunk_days: int = 3,
                pause_s: float = 6.0, cache_dir: Path | None = None,
                max_retries: int = 2, max_consecutive_fails: int = 3) -> pd.DataFrame:
    """Search the GDELT DOC API, one request per date chunk.

    GDELT allows about one request every 5 seconds and returns at most 250 articles
    per request. So all keywords go into one OR query, chunks are a few days long,
    and every request is followed by a pause, including failed ones.

    Each finished chunk is saved to cache_dir. If the run stops (Ctrl+C, internet
    drop), running it again skips the chunks already downloaded.

    If GDELT keeps refusing (several chunks in a row), the run stops early instead
    of retrying for hours: your connection is blocked for a while, so wait and rerun.
    """
    from gdeltdoc import Filters, GdeltDoc

    gd = GdeltDoc()
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    # newest first: recent days matter most for briefs and labelling, and if GDELT
    # blocks the run part-way, what you already have is the useful part
    chunks = list(pd.date_range(start, end, freq=f"{chunk_days}D"))[::-1]
    rows, failed, streak = [], 0, 0
    print(f"GDELT: {len(chunks)} requests, about {len(chunks) * pause_s / 60:.0f} minutes. "
          "Safe to stop with Ctrl+C and rerun later; finished chunks are kept.")
    try:
        for n, chunk_start in enumerate(chunks, 1):
            if streak >= max_consecutive_fails:
                print(f"\n[news] GDELT refused {streak} chunks in a row: your connection is "
                      "rate-limited.\n       Wait 30 to 60 minutes, then run `cli news` again. "
                      "Chunks already downloaded are kept.")
                break
            chunk = _fetch_chunk(gd, Filters, keywords, chunk_start, chunk_days, end, pause_s,
                                 cache_dir, max_retries)
            if chunk is None:
                failed += 1
                streak += 1
                continue
            streak = 0
            rows.append(chunk)
            if n % 20 == 0:
                got = sum(len(r) for r in rows)
                print(f"  {n}/{len(chunks)} chunks done, {got} headlines so far, {failed} skipped")
    except KeyboardInterrupt:
        print("\n[news] stopped by you. Chunks already downloaded are kept; rerun to continue.")
    if cache_dir is not None:
        # An early stop skips the loop over older chunks, so read every cached chunk in
        # the range here. Without this a blocked run would save fewer headlines than
        # are already on disk.
        rows = [pd.read_parquet(cache_dir / f"{c.date()}.parquet") for c in chunks
                if (cache_dir / f"{c.date()}.parquet").exists()]
    rows = [r for r in rows if len(r)]
    if not rows:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    return clean_news(pd.concat(rows, ignore_index=True))


def _fetch_chunk(gd, Filters, keywords, chunk_start, chunk_days, end, pause_s, cache_dir,
                 max_retries):
    """One chunk from cache or GDELT. Returns None if GDELT refused every try."""
    chunk_end = min(chunk_start + pd.Timedelta(days=chunk_days), pd.Timestamp(end))
    cache_file = cache_dir / f"{chunk_start.date()}.parquet" if cache_dir else None
    if cache_file is not None and cache_file.exists():
        return pd.read_parquet(cache_file)
    f = Filters(keyword=keywords, start_date=chunk_start.strftime("%Y-%m-%d"),
                end_date=chunk_end.strftime("%Y-%m-%d"), num_records=250)
    df = None
    for attempt in range(max_retries + 1):
        try:
            df = gd.article_search(f)
            break
        except Exception as exc:  # usually HTTP 429: too many requests
            if attempt == max_retries:
                print(f"  skipped {chunk_start.date()} after {max_retries + 1} tries "
                      f"({type(exc).__name__})")
            else:
                time.sleep(pause_s * 5 * (attempt + 1))
    time.sleep(pause_s)
    if df is None:
        return None
    if df.empty:
        chunk = pd.DataFrame(columns=NEWS_COLUMNS)
    else:
        if "language" in df:
            df = df[df["language"] == "English"]
        chunk = pd.DataFrame({
            "date": pd.to_datetime(df["seendate"], format="%Y%m%dT%H%M%SZ").dt.normalize(),
            "title": df["title"],
            "source": df["domain"],
            "url": df["url"],
            "text": df["title"],
        })
    if cache_file is not None:
        chunk.to_parquet(cache_file)
    return chunk


def fetch_rss(feed_urls: list[str]) -> pd.DataFrame:
    """Read RSS feeds. Feeds only hold recent items, so run this daily to build history."""
    import feedparser

    rows = []
    for url in feed_urls:
        feed = feedparser.parse(url)
        for e in feed.entries:
            published = e.get("published_parsed") or e.get("updated_parsed")
            if not published:
                continue
            rows.append({
                "date": pd.Timestamp(*published[:3]),
                "title": e.get("title", ""),
                "source": feed.feed.get("title", url),
                "url": e.get("link", ""),
                "text": e.get("title", "") + ". " + e.get("summary", "")[:300],
            })
    return clean_news(pd.DataFrame(rows, columns=NEWS_COLUMNS))


def load_statements(folder: Path) -> pd.DataFrame:
    """Load central bank statements saved by hand as text files.

    File name format: YYYY-MM-DD_<source>.txt, for example 2024-09-18_fed.txt.
    Copy the text from the official site (federalreserve.gov, rbi.org.in, ecb.europa.eu).
    Each paragraph becomes one retrievable item.
    """
    rows = []
    for path in sorted(Path(folder).glob("*.txt")):
        day, _, source = path.stem.partition("_")
        paragraphs = [p.strip() for p in path.read_text(encoding="utf-8").split("\n\n") if len(p.strip()) > 40]
        for i, para in enumerate(paragraphs):
            rows.append({
                "date": pd.Timestamp(day),
                "title": f"{source.upper()} statement, paragraph {i + 1}",
                "source": source,
                "url": str(path),
                "text": para,
            })
    return clean_news(pd.DataFrame(rows, columns=NEWS_COLUMNS))


def clean_news(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["title"]).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["title"] = df["title"].str.strip()
    df = df[df["title"].str.len() > 15]
    df = df.drop_duplicates(subset=["date", "title"])
    return df.sort_values("date").reset_index(drop=True)[NEWS_COLUMNS]
