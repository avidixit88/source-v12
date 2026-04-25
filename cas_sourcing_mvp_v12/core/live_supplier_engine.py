from __future__ import annotations

from collections import defaultdict
from urllib.parse import urlparse
import pandas as pd

from services.search_service import (
    build_cas_supplier_queries,
    direct_supplier_search_urls,
    filter_likely_supplier_results,
    serpapi_search,
    discover_product_links_from_page,
    SearchResult,
)
from services.page_extractor import extract_product_rows_from_url
from core.procurement_logic import enrich_procurement_trust
from services.supplier_adapters import (
    ADAPTERS,
    canonicalize_url,
    extract_snippet_price,
    classify_price_visibility,
    best_action_for_status,
    supplier_key_from_url,
)


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def _dedupe_results(results: list[SearchResult]) -> list[SearchResult]:
    seen = set()
    out = []
    for result in results:
        key = canonicalize_url(result.url)
        if key in seen:
            continue
        seen.add(key)
        out.append(result)
    return out


def _supplier_key(result: SearchResult) -> str:
    return result.supplier_hint or supplier_key_from_url(result.url)


def _clean_pack(row: pd.Series) -> str:
    size = row.get("pack_size")
    unit = row.get("pack_unit")
    if pd.isna(size) or not unit or pd.isna(unit):
        return ""
    try:
        return f"{float(size):g} {unit}"
    except Exception:
        return f"{size} {unit}"


def _collapse_price_status(statuses: list[str]) -> str:
    priority = [
        "Public price extracted",
        "Search-snippet price only",
        "Login/account price required",
        "Quote required",
        "No public price detected",
        "Extraction failed",
    ]
    for status in priority:
        if status in statuses:
            return status
    return statuses[0] if statuses else "No public price detected"


def _valid_http_url(url: str) -> bool:
    try:
        parsed = urlparse(str(url or "").strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def _looks_like_search_or_account_url(url: str) -> bool:
    hay = str(url or "").lower()
    noisy_markers = [
        "/search", "catalogsearch", "keyword=", "search=", "q=", "query=",
        "login", "signin", "register", "cart", "basket", "order-status", "quick-order"
    ]
    return any(marker in hay for marker in noisy_markers)


def _choose_representative_url(urls: list[str]) -> str:
    """Prefer a durable product-detail URL over search/session URLs for the UI Open Source link."""
    clean = [u for u in dict.fromkeys(str(x).strip() for x in urls if str(x).strip()) if _valid_http_url(u)]
    if not clean:
        return ""
    for u in clean:
        if not _looks_like_search_or_account_url(u):
            return u
    return clean[0]


def _safe_extract_products(cas_number: str, result: SearchResult, supplier: str):
    """Never let one bad supplier page crash the whole Streamlit run."""
    try:
        return extract_product_rows_from_url(
            cas_number,
            result.url,
            supplier_hint=supplier,
            discovery_title=result.title,
            discovery_snippet=result.snippet,
        )
    except Exception as exc:
        from services.page_extractor import ExtractedProductData
        return [ExtractedProductData(
            supplier=supplier or supplier_key_from_url(result.url),
            title=result.title or "Extraction failed",
            cas_exact_match=False,
            purity=None,
            pack_size=None,
            pack_unit=None,
            listed_price_usd=None,
            stock_status="Not visible",
            product_url=result.url,
            extraction_status=f"failed: {type(exc).__name__}: {str(exc)[:180]}",
            confidence=0,
            evidence=f"v12 guarded extraction failure; source preserved for manual review: {str(exc)[:300]}",
            extraction_method="guarded_exception",
            raw_matches="",
            catalog_number=None,
            price_visibility_status="Extraction failed",
            best_action="Open source manually",
            adapter_name=supplier,
            price_pairing_confidence="NONE",
        )]


def summarize_supplier_rows(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return detail_df.copy()
    df = detail_df.copy()
    df["pack_label"] = df.apply(_clean_pack, axis=1)
    records = []
    for supplier, g in df.groupby("supplier", dropna=False):
        visible = g[g.get("listed_price_usd").notna()] if "listed_price_usd" in g.columns else pd.DataFrame()
        eligible = g[g.get("bulk_estimate_eligible", pd.Series(False, index=g.index)).fillna(False).astype(bool)] if "bulk_estimate_eligible" in g.columns else pd.DataFrame()
        statuses = [str(x) for x in g.get("price_visibility_status", pd.Series(dtype=str)).dropna().tolist()]
        status = _collapse_price_status(statuses)
        pack_options = sorted({x for x in g["pack_label"].tolist() if x})
        purities = sorted({str(x) for x in g.get("purity", pd.Series(dtype=str)).dropna().tolist() if str(x) and str(x) != "Not visible"})
        urls = list(dict.fromkeys(g["product_url"].dropna().astype(str).tolist()))[:8]
        cat_nums = sorted({str(x) for x in g.get("catalog_number", pd.Series(dtype=str)).dropna().tolist() if str(x) and str(x) != "nan"})
        source_tiers = sorted({str(x) for x in g.get("source_tier", pd.Series(dtype=str)).dropna().tolist() if str(x)})
        product_forms = sorted({str(x) for x in g.get("product_form", pd.Series(dtype=str)).dropna().tolist() if str(x) and str(x) != "nan"})
        trust_decisions = list(dict.fromkeys(g.get("procurement_trust_decision", pd.Series(dtype=str)).dropna().astype(str).tolist()))[:5]
        row = {
            "supplier": supplier,
            "cas_number": g["cas_number"].iloc[0],
            "cas_exact_match": bool(g.get("cas_exact_match", pd.Series([False])).fillna(False).astype(bool).any()),
            "source_tier": ", ".join(source_tiers) if source_tiers else "unknown",
            "products_found": int(g["canonical_url"].nunique()),
            "catalog_numbers": ", ".join(cat_nums[:8]) if cat_nums else "Not extracted",
            "purities_found": ", ".join(purities[:8]) if purities else "Not visible",
            "pack_options": ", ".join(pack_options[:12]) if pack_options else "Not visible",
            "product_forms": ", ".join(product_forms) if product_forms else "unknown",
            "bulk_estimate_eligible_count": int(len(eligible)),
            "visible_price_count": int(len(visible)),
            "high_confidence_price_pairs": int((visible.get("price_pairing_confidence", pd.Series(dtype=str)) == "HIGH").sum()) if not visible.empty else 0,
            "medium_confidence_price_pairs": int((visible.get("price_pairing_confidence", pd.Series(dtype=str)) == "MEDIUM").sum()) if not visible.empty else 0,
            "best_visible_price_usd": float(visible["listed_price_usd"].min()) if not visible.empty else None,
            "price_visibility_status": status,
            "best_action": best_action_for_status(status),
            "stock_summary": "; ".join(list(dict.fromkeys(g.get("stock_status", pd.Series(dtype=str)).dropna().astype(str).tolist()))[:5]) or "Not visible",
            "max_extraction_confidence": int(g.get("extraction_confidence", pd.Series([0])).fillna(0).max()),
            "source_urls": " | ".join(urls),
            "representative_url": _choose_representative_url(urls),
            "trust_decisions": " | ".join(trust_decisions) if trust_decisions else "",
            "notes": "v12 grouped row. Public prices are displayed, but the curve model only uses CAS-confirmed mass products with paired quantity-price rows and acceptable purity.",
            "data_source": "live_supplier_adapter_summary_v12",
        }
        records.append(row)
    out = pd.DataFrame(records)
    if not out.empty:
        out["_has_public_price"] = out["visible_price_count"] > 0
        out["_tier_rank"] = out["source_tier"].map(lambda x: 3 if "price_first" in str(x) else 2 if "marketplace" in str(x) else 1)
        out = out.sort_values(
            ["cas_exact_match", "_has_public_price", "_tier_rank", "max_extraction_confidence", "products_found"],
            ascending=[False, False, False, False, False],
        ).drop(columns=["_has_public_price", "_tier_rank"])
    return out


def _build_supplier_seed_map(cas_number: str, serpapi_key: str | None, chemical_name: str | None) -> tuple[dict[str, list[SearchResult]], pd.DataFrame]:
    seed_map: dict[str, list[SearchResult]] = defaultdict(list)
    discovery_records = []

    # 1) Registry-first: every curated supplier gets a chance before generic search can dominate.
    direct = direct_supplier_search_urls(cas_number)
    for result in direct:
        seed_map[_supplier_key(result)].append(result)

    # 2) Optional search API broadens coverage, but it is subordinate to the registry and identity-gated later.
    serp_results: list[SearchResult] = []
    if serpapi_key:
        queries = build_cas_supplier_queries(cas_number, chemical_name)
        serp_results = filter_likely_supplier_results(serpapi_search(queries, serpapi_key or ""))
        for result in serp_results:
            seed_map[_supplier_key(result)].append(result)

    for supplier, results in seed_map.items():
        for r in _dedupe_results(results):
            discovery_records.append({
                "supplier": supplier,
                "title": r.title,
                "url": r.url,
                "canonical_url": canonicalize_url(r.url),
                "domain": _domain(r.url),
                "snippet": r.snippet,
                "source": r.source,
                "supplier_hint": r.supplier_hint,
            })
    return {k: _dedupe_results(v) for k, v in seed_map.items()}, pd.DataFrame(discovery_records)


def discover_live_suppliers(
    cas_number: str,
    chemical_name: str | None = None,
    serpapi_key: str | None = None,
    max_pages_to_extract: int = 30,
    include_direct_links: bool = True,
    max_suppliers: int = 24,
    pages_per_supplier: int = 2,
    required_purity: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """v12 strict supplier-registry sourcing with procurement trust enrichment.

    Main correction from v8: the engine now iterates suppliers from the curated registry first,
    gives each supplier its own extraction budget, and requires page-level CAS identity. This prevents
    one or two noisy vendors/search pages from consuming the extraction budget or falsely confirming
    wrong product pages.
    """
    seed_map, discovery_df = _build_supplier_seed_map(cas_number, serpapi_key, chemical_name)
    if not include_direct_links and not serpapi_key:
        seed_map = {}

    # Preserve registry priority order.
    supplier_order = [a.name for a in ADAPTERS if a.name in seed_map]
    supplier_order += [s for s in seed_map if s not in supplier_order]
    supplier_order = supplier_order[:max_suppliers]

    extracted_rows = []
    total_extracted = 0

    adapter_by_name = {a.name: a for a in ADAPTERS}
    for supplier in supplier_order:
        if total_extracted >= max_pages_to_extract:
            break
        seeds = seed_map.get(supplier, [])
        adapter = adapter_by_name.get(supplier)
        # Expand each supplier's own seed/search pages into product candidates.
        expanded: list[SearchResult] = []
        for seed in seeds[:2]:
            try:
                expanded.extend(discover_product_links_from_page(seed, cas_number, max_links=4))
            except Exception:
                # Keep the seed page as fallback; link expansion is helpful but not allowed to crash discovery.
                continue
        candidates = _dedupe_results(expanded + seeds)
        # Product-link candidates first, then direct search pages as fallback.
        candidates = sorted(candidates, key=lambda r: (0 if r.source.startswith("expanded") else 1, canonicalize_url(r.url)))

        supplier_pages_done = 0
        for result in candidates:
            if total_extracted >= max_pages_to_extract or supplier_pages_done >= pages_per_supplier:
                break
            extracted_products = _safe_extract_products(cas_number, result, supplier)
            snippet_price = extract_snippet_price(result.snippet)

            page_kept = False
            for extracted in extracted_products:
                price_visibility_status = extracted.price_visibility_status
                if extracted.listed_price_usd is None and snippet_price is not None and extracted.cas_exact_match:
                    price_visibility_status = classify_price_visibility(None, result.snippet, snippet_price, extracted.extraction_status)

                # v12 keep rule: product layer should contain CAS-confirmed pages or meaningful supplier availability state.
                keep = bool(extracted.cas_exact_match) or price_visibility_status in [
                    "Login/account price required", "Quote required", "Extraction failed"
                ]
                if not keep:
                    continue

                extracted_rows.append({
                    "cas_number": cas_number,
                    "chemical_name": chemical_name or "",
                    "supplier": extracted.supplier,
                    "source_tier": adapter.source_tier if adapter else "unknown",
                    "expected_pricing_behavior": adapter.expected_behavior if adapter else "unknown",
                    "region": "Unknown",
                    "purity": extracted.purity or "Not visible",
                    "pack_size": extracted.pack_size,
                    "pack_unit": extracted.pack_unit,
                    "listed_price_usd": extracted.listed_price_usd,
                    "snippet_price_usd": snippet_price if extracted.cas_exact_match else None,
                    "price_visibility_status": price_visibility_status,
                    "best_action": best_action_for_status(price_visibility_status),
                    "stock_status": extracted.stock_status,
                    "lead_time": "Not visible",
                    "product_url": extracted.product_url,
                    "canonical_url": canonicalize_url(extracted.product_url),
                    "domain": _domain(extracted.product_url),
                    "catalog_number": extracted.catalog_number,
                    "notes": extracted.evidence,
                    "page_title": extracted.title,
                    "cas_exact_match": extracted.cas_exact_match,
                    "extraction_status": extracted.extraction_status,
                    "extraction_confidence": extracted.confidence,
                    "extraction_method": extracted.extraction_method,
                    "price_pairing_confidence": getattr(extracted, "price_pairing_confidence", "NONE"),
                    "raw_matches": extracted.raw_matches,
                    "product_form": getattr(extracted, "product_form", "unknown"),
                    "purity_confidence": getattr(extracted, "purity_confidence", "NONE"),
                    "url_role": getattr(extracted, "url_role", "source_page"),
                    "landing_url": getattr(extracted, "landing_url", extracted.product_url),
                    "canonical_product_url": getattr(extracted, "canonical_product_url", canonicalize_url(extracted.product_url)),
                    "price_noise_flag": getattr(extracted, "price_noise_flag", False),
                    "data_source": "live_extraction_v12_scaleup_model",
                })
                page_kept = True
            if page_kept:
                supplier_pages_done += 1
                total_extracted += 1

    detail_df = pd.DataFrame(extracted_rows)
    if not detail_df.empty:
        dedupe_cols = [c for c in ["supplier", "canonical_url", "catalog_number", "purity", "pack_size", "pack_unit", "listed_price_usd", "price_visibility_status", "price_pairing_confidence"] if c in detail_df.columns]
        detail_df = detail_df.drop_duplicates(subset=dedupe_cols, keep="first")
        # Remove obvious unsafe public-price rows: no CAS identity or zero/free-sample oddities.
        if "listed_price_usd" in detail_df.columns:
            detail_df.loc[~detail_df["cas_exact_match"].astype(bool), "listed_price_usd"] = None
        detail_df = enrich_procurement_trust(detail_df, required_purity=required_purity)
        detail_df = detail_df.sort_values(["cas_exact_match", "source_tier", "bulk_estimate_eligible", "listed_price_usd", "extraction_confidence"], ascending=[False, True, False, True, False])

    summary_df = summarize_supplier_rows(detail_df)
    return detail_df, discovery_df, summary_df
