import aiohttp
import itertools
import json
import os
import sys
from typing import Dict, List
import urllib.parse


def _extract_values(keys, obj):
    values = {}
    for k in keys:
        v = obj.get(k)
        if v is not None:
            values[k] = v
    return values


# Let's keep this tool aligned w/ llama_stack.providers.impls.meta_reference.agents.tools.builtin.BraveSearch
# (see https://github.com/meta-llama/llama-stack/blob/main/llama_stack/providers/impls/meta_reference/agents/tools/builtin.py)
_result_keys_by_type = {
    "web":       ("type", "title", "url", "description", "date", "extra_snippets"),
    "videos":    ("type", "title", "url", "description", "date"),
    "news":      ("type", "title", "url", "description"),
    "infobox":   ("type", "title", "url", "description", "long_desc"),
    "locations": ("type", "title", "url", "description", "coordinates", "postal_address", "contact", "rating", "distance", "zoom_level"),
    "faq":       ("type", "title", "url", "question", "answer"),
}


async def brave_search(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search the Brave Search API for the specified query.

    Parameters:
        query (str): The query to search for.
        max_results (int): The maximum number of results to return (defaults to 10)

    Returns:
        List[Dict]: The search results.
    """

    url = f"https://api.search.brave.com/res/v1/web/search?q={urllib.parse.quote(query)}"
    headers = {
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip',
        'X-Subscription-Token': os.environ['BRAVE_SEARCH_API_KEY'],
    }

    def extract_results(search_response):
        for m in search_response['mixed']['main']:
            result_type = m['type']
            keys = _result_keys_by_type.get(result_type)
            if keys is None:
                print(f'[brave_search] Unknown result type: {result_type}', file=sys.stderr)
                continue

            results_of_type = search_response[result_type]["results"]
            if (idx := m.get("index")) is not None:
                yield _extract_values(keys, results_of_type[idx])
            elif m["all"]:
                for r in results_of_type:
                    yield _extract_values(keys, r)

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as res:
            res.raise_for_status()
            response = await res.json()

            results = list(itertools.islice(extract_results(response), max_results))
            print(json.dumps(dict(query=query, response=response, results=results), indent=2))
            return results
