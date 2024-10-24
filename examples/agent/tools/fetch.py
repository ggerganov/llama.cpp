# import aiohttp
import html2text
import logging
import requests


async def fetch_page(url: str):
    '''
        Fetch a web page (convert it to markdown if possible), using aiohttp.
    '''

    try:
        logging.debug(f'[fetch_page] Fetching %s', url)
        response = requests.get(url)
        response.raise_for_status()
        content = response.text
        # async with aiohttp.ClientSession(trust_env=True) as session:
        #     async with session.get(url) as res:
        #         res.raise_for_status()
        #         content = await res.text()
    # except aiohttp.ClientError as e:
    #     raise Exception(f'Failed to fetch {url}: {e}')
    except requests.exceptions.RequestException as e:
        raise Exception(f'Failed to fetch {url}: {e}')

    # NOTE: Pyppeteer doesn't work great in docker, short of installing a bunch of dependencies
    # from pyppeteer import launch
    # from pyppeteer.errors import TimeoutError, NetworkError
    # browser = await launch()
    # try:
    #     page = await browser.newPage()
    #     response = await page.goto(url)

    #     if not response.ok:
    #         return FetchResult(error=f'HTTP {response.status} {response.statusText}')

    #     content=await page.content()
    # except TimeoutError:
    #     return FetchResult(error='Page load timed out')
    # except NetworkError:
    #     return FetchResult(error='Network error occurred')
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
        return markdown
    except Exception as e:
        logging.warning('[fetch_page] Failed to convert HTML of %s to markdown: %s', url, e)
        return content
