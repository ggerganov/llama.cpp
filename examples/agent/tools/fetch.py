import aiohttp
import sys
from typing import Optional

from pydantic import BaseModel
import html2text


class FetchResult(BaseModel):
    content: Optional[str] = None
    markdown: Optional[str] = None
    error: Optional[str] = None


async def fetch_page(url: str) -> FetchResult:
    '''
        Fetch a web page (convert it to markdown if possible).
    '''

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as res:
                res.raise_for_status()
                content = await res.text()
    except aiohttp.ClientError as e:
        return FetchResult(error=str(e))

    # NOTE: Pyppeteer doesn't work great in docker, short of installing a bunch of dependencies
    # from pyppeteer import launch
    # from pyppeteer.errors import TimeoutError, NetworkError
    # browser = await launch()
    # try:
    #     page = await browser.newPage()
    #     response = await page.goto(url)

    #     if not response.ok:
    #         return FetchResult(error=f"HTTP {response.status} {response.statusText}")

    #     content=await page.content()
    # except TimeoutError:
    #     return FetchResult(error="Page load timed out")
    # except NetworkError:
    #     return FetchResult(error="Network error occurred")
    # except Exception as e:
    #     return FetchResult(error=str(e))
    # finally:
    #     await browser.close()

    try:
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        h.ignore_emphasis = False
        markdown = h.handle(content)
        return FetchResult(markdown=markdown)
    except Exception as e:
        print(f'Failed to convert HTML of {url} to markdown: {e}', file=sys.stderr)
        return FetchResult(content=content)
