"""Biomedical registry used by the GreenLight mock interface.

This registry resolves target symbols using the Ensembl REST API and disease
names using the EBI OLS (Ontology Lookup Service) API, which provides access to
both EFO (Experimental Factor Ontology) and MONDO (Monarch Disease Ontology).
"""

from __future__ import annotations

from functools import lru_cache

import httpx


ENSEMBL_SPECIES = "homo_sapiens"
ENSEMBL_LOOKUP_URL = "https://rest.ensembl.org/lookup/symbol/{species}/{symbol}"

# OLS (Ontology Lookup Service) API endpoints
OLS_SEARCH_URL = "https://www.ebi.ac.uk/ols4/api/search"
OLS_ONTOLOGIES = "efo,mondo"  # Search across EFO and MONDO

HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}
TIMEOUT = 5.0


def normalize_term(value: str) -> str:
    return " ".join(value.strip().lower().split())


@lru_cache(maxsize=128)
def resolve_target_id(target_symbol: str) -> tuple[str, list[str]]:
    """
    Resolve a gene symbol to its Ensembl ID.
    
    Args:
        target_symbol: Gene symbol (e.g., "PCSK9")
        
    Returns:
        Tuple of (ensembl_id, suggestions)
        - ensembl_id: The Ensembl gene ID or "unknown"
        - suggestions: List of alternative symbol suggestions if lookup fails
    """
    symbol = target_symbol.strip()
    if not symbol:
        return "unknown", []
    

    try:
        response = httpx.get(
            ENSEMBL_LOOKUP_URL.format(species=ENSEMBL_SPECIES, symbol=symbol),
            headers=HEADERS,
            timeout=TIMEOUT,
        )
    except httpx.HTTPError:
        return "unknown", []

    if response.status_code == 200:
        data = response.json()
        if (
            data.get("object_type") == "Gene"
            and data.get("id")
            and data.get("species") == ENSEMBL_SPECIES
        ):
            return data["id"], []

    return "unknown", []


@lru_cache(maxsize=128)
def resolve_disease_id(disease_name: str) -> tuple[str, list[str]]:
    """
    Resolve a disease name to its EFO or MONDO ID using the OLS API.
    
    Attempts to find the disease via OLS (Ontology Lookup Service) which
    provides access to both EFO and MONDO. Prioritizes exact label matches
    and returns the best match along with suggestions.
    
    Args:
        disease_name: Disease name (e.g., "dementia", "hypercholesterolemia")
        
    Returns:
        Tuple of (disease_id, suggestions)
        - disease_id: The EFO or MONDO ID (e.g., "MONDO_0004975") or "unknown"
        - suggestions: List of alternative disease name suggestions
    """
    normalized = normalize_term(disease_name)
    if not normalized:
        return "unknown", []

    try:
        # Request more results to find the best match
        response = httpx.get(
            OLS_SEARCH_URL,
            params={
                "q": normalized,
                "ontology": OLS_ONTOLOGIES,
                "type": "class",
                "rows": 30,
            },
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        
        if response.status_code != 200:
            return "unknown", []
        
        data = response.json()
        results = data.get("response", {}).get("docs", [])
        
        if not results:
            return "unknown", []
        
        # Score results: exact label matches score highest
        scored_results = []
        for result in results:
            label = result.get("label", "").lower()
            short_form = result.get("short_form", "")
            
            # Calculate relevance score
            score = 0
            if label == normalized:
                score = 1000  # Exact match
            elif label.startswith(normalized):
                score = 500  # Starts with query
            elif normalized in label:
                score = 250  # Contains query
            else:
                score = 0
            
            if score > 0 and short_form:
                scored_results.append((score, result, label, short_form))
        
        if not scored_results:
            # Fallback: just use the first result if no score matches
            best_match = results[0]
            disease_id = best_match.get("short_form")
            suggestions = [
                result.get("label", result.get("short_form"))
                for result in results[1:4]
                if result.get("short_form")
            ]
            if disease_id:
                return disease_id, suggestions
        
        # Sort by score descending and get the best match
        scored_results.sort(reverse=True, key=lambda x: x[0])
        best_score, best_match, _, disease_id = scored_results[0]
        
        # Collect suggestions from remaining high-scoring results
        suggestions = [
            result[2].title() or result[3]
            for result in scored_results[1:4]
        ]
        
        if disease_id:
            return disease_id, suggestions
    
    except httpx.HTTPError:
        pass
    
    return "unknown", []
