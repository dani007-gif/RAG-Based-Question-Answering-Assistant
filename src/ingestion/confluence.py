"""
src/ingestion/confluence.py
───────────────────────────
Fetches pages from Confluence via the REST API and converts them
into LangChain Document objects ready for the ingestion pipeline.

Requires:
    CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN,
    CONFLUENCE_SPACE_KEY  in .env

Install: pip install atlassian-python-api
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import List, Optional

from langchain_core.documents import Document

from config.settings import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


def _strip_html(html: str) -> str:
    """Very lightweight HTML → plain text (no heavy deps)."""
    # Remove script/style blocks
    html = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", "", html, flags=re.DOTALL)
    # Replace block elements with newlines
    html = re.sub(r"</(p|div|h[1-6]|li|tr)>", "\n", html)
    # Replace <br> with newline
    html = re.sub(r"<br\s*/?>", "\n", html)
    # Strip remaining tags
    html = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    html = re.sub(r" {2,}", " ", html)
    return html.strip()


def load_confluence_pages(
    space_key: Optional[str] = None,
    limit: int = 200,
) -> List[Document]:
    """
    Fetch all pages from a Confluence space.

    Parameters
    ----------
    space_key : str, optional
        Confluence space key.  Defaults to settings.confluence_space_key.
    limit : int
        Max pages to fetch per request.

    Returns
    -------
    List[Document]
        One Document per Confluence page, with metadata:
          - source          : page URL
          - page_id         : Confluence page ID
          - title           : page title
          - space_key       : space key
          - version         : page version number
          - last_modified   : ISO timestamp
          - file_type       : "confluence"
    """
    try:
        from atlassian import Confluence  # lazy import — optional dep
    except ImportError:
        log.error(
            "atlassian_not_installed",
            hint="pip install atlassian-python-api",
        )
        return []

    if not settings.confluence_url:
        log.warning("confluence_not_configured")
        return []

    space = space_key or settings.confluence_space_key

    client = Confluence(
        url=settings.confluence_url,
        username=settings.confluence_username,
        password=settings.confluence_api_token,
        cloud=True,
    )

    docs: List[Document] = []
    start = 0

    while True:
        try:
            results = client.get_all_pages_from_space(
                space=space,
                start=start,
                limit=limit,
                expand="body.storage,version",
            )
        except Exception as exc:  # noqa: BLE001
            log.error("confluence_fetch_error", error=str(exc))
            break

        if not results:
            break

        for page in results:
            page_id = page["id"]
            title = page["title"]
            html_body = page.get("body", {}).get("storage", {}).get("value", "")
            plain_text = _strip_html(html_body)

            if not plain_text.strip():
                continue

            version = page.get("version", {}).get("number", 0)
            last_modified = page.get("version", {}).get(
                "when", datetime.utcnow().isoformat()
            )
            page_url = f"{settings.confluence_url}/wiki/spaces/{space}/pages/{page_id}"

            doc = Document(
                page_content=plain_text,
                metadata={
                    "source": page_url,
                    "page_id": page_id,
                    "title": title,
                    "space_key": space,
                    "version": version,
                    "last_modified": last_modified,
                    "file_type": "confluence",
                    "file_name": f"confluence_{page_id}",
                    "file_hash": f"v{version}",
                },
            )
            docs.append(doc)
            log.info("confluence_page_loaded", title=title, page_id=page_id)

        start += limit
        if len(results) < limit:
            break

    log.info("confluence_loading_complete", total_pages=len(docs), space=space)
    return docs
