"""Base scraper: async download with rate limiting and retry."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_HEADERS = {
    "User-Agent": "pokedex-cv/0.1 (https://github.com/yoshiri/pokedex_cv)"
}


@dataclass
class DownloadResult:
    pokemon_id: int
    sprite_type: str
    path: Path | None        # None if skipped or failed
    status: str              # "ok" | "skipped" | "not_found" | "error"
    error: str = ""


@dataclass
class CollectionSummary:
    ok: int = 0
    skipped: int = 0
    not_found: int = 0
    error: int = 0
    results: list[DownloadResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.ok + self.skipped + self.not_found + self.error

    def add(self, r: DownloadResult) -> None:
        self.results.append(r)
        if r.status == "ok":
            self.ok += 1
        elif r.status == "skipped":
            self.skipped += 1
        elif r.status == "not_found":
            self.not_found += 1
        else:
            self.error += 1

    def __str__(self) -> str:
        return (
            f"total={self.total} ok={self.ok} skipped={self.skipped} "
            f"not_found={self.not_found} error={self.error}"
        )


class BaseScraper(ABC):
    def __init__(
        self,
        output_dir: Path,
        *,
        requests_per_second: float = 5.0,
        concurrency: int = 8,
    ) -> None:
        self.output_dir = Path(output_dir)
        self._delay = 1.0 / max(requests_per_second, 0.1)
        self._semaphore = asyncio.Semaphore(concurrency)
        self._last_request_time = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def collect(self, pokemon_ids: list[int]) -> CollectionSummary:
        summary = CollectionSummary()
        async with httpx.AsyncClient(
            headers=_DEFAULT_HEADERS,
            follow_redirects=True,
            timeout=30.0,
        ) as client:
            tasks = [self._collect_one(client, pid, summary) for pid in pokemon_ids]
            await asyncio.gather(*tasks)
        return summary

    # ------------------------------------------------------------------
    # Abstract interface for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def get_targets(self, pokemon_id: int) -> list[tuple[str, str]]:
        """Return list of (sprite_type, url) for the given pokemon_id."""

    def output_path(self, pokemon_id: int, sprite_type: str) -> Path:
        return self.output_dir / f"{pokemon_id:04d}" / f"{sprite_type}.png"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _collect_one(
        self, client: httpx.AsyncClient, pokemon_id: int, summary: CollectionSummary
    ) -> None:
        for sprite_type, url in self.get_targets(pokemon_id):
            result = await self._download(client, pokemon_id, sprite_type, url)
            summary.add(result)
            if result.status == "ok":
                logger.debug("ok   %s → %s", url, result.path)
            elif result.status == "not_found":
                logger.debug("404  %s", url)
            elif result.status == "error":
                logger.warning("err  %s: %s", url, result.error)

    async def _download(
        self,
        client: httpx.AsyncClient,
        pokemon_id: int,
        sprite_type: str,
        url: str,
        *,
        retries: int = 3,
    ) -> DownloadResult:
        dest = self.output_path(pokemon_id, sprite_type)

        if dest.exists():
            return DownloadResult(pokemon_id, sprite_type, dest, "skipped")

        async with self._semaphore:
            await self._rate_limit()
            for attempt in range(retries):
                try:
                    resp = await client.get(url)
                    if resp.status_code == 404:
                        return DownloadResult(pokemon_id, sprite_type, None, "not_found")
                    resp.raise_for_status()
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(resp.content)
                    return DownloadResult(pokemon_id, sprite_type, dest, "ok")
                except httpx.HTTPStatusError as e:
                    if attempt == retries - 1:
                        return DownloadResult(
                            pokemon_id, sprite_type, None, "error", str(e)
                        )
                    await asyncio.sleep(2 ** attempt)
                except Exception as e:
                    if attempt == retries - 1:
                        return DownloadResult(
                            pokemon_id, sprite_type, None, "error", str(e)
                        )
                    await asyncio.sleep(2 ** attempt)

        return DownloadResult(pokemon_id, sprite_type, None, "error", "unreachable")

    async def _rate_limit(self) -> None:
        loop = asyncio.get_event_loop()
        now = loop.time()
        wait = self._last_request_time + self._delay - now
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request_time = loop.time()
