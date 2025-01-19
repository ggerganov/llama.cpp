import json
import logging
from SPARQLWrapper import JSON, SPARQLWrapper


def execute_sparql(endpoint: str, query: str) -> str:
    '''
        Execute a SPARQL query on a given endpoint
    '''

    logging.debug(f'[sparql] Executing on %s:\n%s', endpoint, query)
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return json.dumps(sparql.query().convert(), indent=2)


def wikidata_sparql(query: str) -> str:
    'Execute a SPARQL query on Wikidata'

    return execute_sparql("https://query.wikidata.org/sparql", query)


def dbpedia_sparql(query: str) -> str:
    'Execute a SPARQL query on DBpedia'

    return execute_sparql("https://dbpedia.org/sparql", query)

