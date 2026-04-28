"""
Wikipedia scraper using the official `wikipedia` Python library.
Clean API — no BeautifulSoup or raw HTML parsing.
"""

import logging
import time
from typing import Optional

import wikipedia
import requests as _requests

logger = logging.getLogger(__name__)

# Respect Wikipedia's rate limit
_REQUEST_DELAY_SEC = 1.5

# Set a proper User-Agent so Wikipedia doesn't reject requests
_UA = "LocalWikiRAG/1.0 (Educational project; Python/wikipedia-api)"
try:
    _session = _requests.Session()
    _session.headers.update({"User-Agent": _UA})
    # Monkey-patch the wikipedia library's internal session
    import wikipedia as _wiki_mod
    if hasattr(_wiki_mod, 'SESSION'):
        _wiki_mod.SESSION = _session
except Exception:
    pass


class WikipediaScraper:
    def __init__(self, language: str = "en"):
        wikipedia.set_lang(language)

    def scrape_page(self, title: str, _retries: int = 5) -> Optional[dict]:
        """
        Fetch a single Wikipedia page.

        Returns dict with keys: title, url, content, summary
        Returns None on failure.
        """
        for attempt in range(1, _retries + 1):
            try:
                page = wikipedia.page(title, auto_suggest=False)
                time.sleep(_REQUEST_DELAY_SEC)
                result = {
                    "title": page.title,
                    "url": page.url,
                    "content": page.content,
                    "summary": page.summary,
                }
                logger.info("Scraped '%s' (%d chars).", page.title, len(page.content))
                return result

            except wikipedia.DisambiguationError as exc:
                # Try the first unambiguous option
                logger.warning(
                    "Disambiguation for '%s'; trying first option '%s'.",
                    title,
                    exc.options[0],
                )
                try:
                    page = wikipedia.page(exc.options[0], auto_suggest=False)
                    time.sleep(_REQUEST_DELAY_SEC)
                    return {
                        "title": page.title,
                        "url": page.url,
                        "content": page.content,
                        "summary": page.summary,
                    }
                except Exception as inner:
                    logger.error("Failed fallback for '%s': %s", title, inner)
                    return None

            except wikipedia.PageError:
                logger.error("Page not found for '%s'.", title)
                return None

            except Exception as exc:
                if attempt < _retries:
                    wait = _REQUEST_DELAY_SEC * (2 ** attempt)
                    logger.warning(
                        "Retryable error scraping '%s' (attempt %d/%d): %s — "
                        "waiting %.1fs",
                        title, attempt, _retries, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "Failed to scrape '%s' after %d attempts: %s",
                        title, _retries, exc,
                    )
                    return None

    def scrape_many(
        self,
        titles: list[str],
        entity_type: str,
        progress_callback=None,
    ) -> list[dict]:
        """
        Scrape multiple pages; attaches entity_type to each result.

        progress_callback(current, total, title) — optional UI hook.
        """
        results = []
        total = len(titles)
        for idx, title in enumerate(titles, start=1):
            if progress_callback:
                progress_callback(idx, total, title)
            data = self.scrape_page(title)
            if data:
                data["entity_type"] = entity_type
                results.append(data)
        return results
