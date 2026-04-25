from __future__ import annotations

from dataclasses import dataclass, replace
from urllib.parse import urljoin
import json
import re
from typing import Any

import requests
from bs4 import BeautifulSoup

from services.supplier_adapters import (
    PRICE_PUBLIC,
    best_action_for_status,
    canonicalize_url,
    classify_price_visibility,
    extract_catalog_number,
    extract_snippet_price,
    supplier_name_for_url,
)

PRICE_RE = re.compile(
    r"(?:USD\s*)?(?:US\$|\$)\s?([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?|[0-9]+(?:\.[0-9]{2})?)|"
    r"\b([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)\s?(?:USD|US\s?dollars)\b",
    re.I,
)
PACK_RE = re.compile(
    r"\b(?:pack\s*size|size|quantity|qty|amount|unit)?\s*[:\-]?\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s?(ug|µg|microgram|micrograms|mg|milligram|milligrams|g|gram|grams|kg|kilogram|kilograms|ml|mL|milliliter|milliliters|L|l|liter|liters)\b",
    re.I,
)
SOLUTION_PACK_RE = re.compile(
    r"(?i)(\d+(?:\.\d+)?)\s?mL\s*(?:x|\*)?\s*\d+(?:\.\d+)?\s?mM\s*(?:\(?\s*in\s*DMSO\s*\)?)?"
    r"|\d+(?:\.\d+)?\s?mM\s*(?:x|\*)?\s*(\d+(?:\.\d+)?)\s?mL\s*(?:in\s*DMSO)?"
)
PURITY_RE = re.compile(
    r"\b(?:purity|assay|hplc|lc[-\s]?ms|gc|nmr|analysis)\b[^%]{0,80}?(?:>|≥|>=)?\s*([0-9]{2,3}(?:\.[0-9]+)?\s?%)|"
    r"([0-9]{2,3}(?:\.[0-9]+)?\s?%)\s*(?:purity|assay|by\s+hplc|hplc|lc[-\s]?ms|gc)",
    re.I,
)
CAS_CONTEXT_RE = re.compile(r"\bCAS(?:\s*(?:No\.?|Number|#))?\s*[:\-]?\s*([0-9]{2,7}-[0-9]{2}-[0-9])\b", re.I)
STOCK_RE = re.compile(
    r"\b(in stock|available|ships in [^.;,|]{1,45}|usually ships[^.;,|]{0,45}|lead time[^.;,|]{0,45}|out of stock|request quote|request a quote|ask for quotation|quote only|login to view price|sign in to view price|price on request)\b",
    re.I,
)
LOGIN_PRICE_RE = re.compile(r"(?i)(sign in to view price|login to view price|log in to view price|price unavailable|request a quote|request quote|price on request)")
JS_PRICE_KEY_RE = re.compile(r'''(?i)["'](?:price|unitprice|unit_price|listprice|list_price|saleprice|sale_price|catalogprice|catalog_price|customerprice|customer_price|yourprice|your_price|amount)["']\s*[:=]\s*["']?\$?\s*([0-9]{1,5}(?:,[0-9]{3})*(?:\.[0-9]{1,4})?|[0-9]+(?:\.[0-9]{1,4})?)["']?''')
PRICE_GUARD_RE = re.compile(r"(?i)(price|unitprice|listprice|saleprice|catalogprice|customerprice|yourprice|usd|\$)")
PRICE_NOISE_RE = re.compile(
    r"(?i)(free\s+shipping|orders?\s+over|minimum\s+order|cart\b|basket\b|subtotal|checkout|coupon|promo|discount|shipping\s+threshold|handling\s+fee|tax\b|recently\s+added|sign\s+in\s+to\s+checkout|save\s+\d+%)"
)
PURITY_NOISE_RE = re.compile(
    r"(?i)(happy\s+customers|discount|coupon|promo|off\b|save\b|recovery|inhibition|viability|cell|solution|dmso|\bmm\b|shipping|orders?\s+over|free\s+shipping|cart|subtotal|tax|rating|reviews?)"
)
SEARCH_LIKE_URL_RE = re.compile(r"(?i)(/search|catalogsearch|keyword=|search=|q=|query=|find\.cgi|/result|search\.aspx|search\.html)")
PRODUCT_URL_HINT_RE = re.compile(r"(?i)(/compound/|/products?/|/product/|\.html$|/item/|/shop/compound/)")


@dataclass(frozen=True)
class ExtractedProductData:
    supplier: str
    title: str
    cas_exact_match: bool
    purity: str | None
    pack_size: float | None
    pack_unit: str | None
    listed_price_usd: float | None
    stock_status: str
    product_url: str
    extraction_status: str
    confidence: int
    evidence: str
    extraction_method: str
    raw_matches: str
    catalog_number: str | None = None
    price_visibility_status: str = "No public price detected"
    best_action: str = "Check source / RFQ"
    adapter_name: str | None = None
    price_pairing_confidence: str = "NONE"
    product_form: str = "unknown"
    purity_confidence: str = "NONE"
    purity_pass: str = "NOT_EVALUATED"
    url_role: str = "source_page"
    landing_url: str | None = None
    canonical_product_url: str | None = None
    price_noise_flag: bool = False


def supplier_name_from_url(url: str) -> str:
    return supplier_name_for_url(url)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(str(value).replace(",", "").replace("$", "").strip())
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def _normalize_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    u = unit.strip().lower()
    mapping = {
        "microgram": "ug",
        "micrograms": "ug",
        "µg": "ug",
        "ug": "ug",
        "milligram": "mg",
        "milligrams": "mg",
        "gram": "g",
        "grams": "g",
        "kilogram": "kg",
        "kilograms": "kg",
        "milliliter": "mL",
        "milliliters": "mL",
        "ml": "mL",
        "liter": "L",
        "liters": "L",
        "l": "L",
    }
    return mapping.get(u, unit)


def _pack_is_reasonable(size: float | None, unit: str | None) -> bool:
    if size is None or unit is None:
        return False
    caps = {"ug": 1_000_000_000, "mg": 1_000_000, "g": 100_000, "kg": 10_000, "ml": 1_000_000, "mL": 1_000_000, "l": 10_000, "L": 10_000}
    return 0 < size <= caps.get(unit, caps.get(unit.lower(), 100_000))


def _price_is_noise(price: float | None, context: str | None) -> bool:
    if price is None:
        return False
    text = re.sub(r"\s+", " ", str(context or ""))[:1500]
    return bool(PRICE_NOISE_RE.search(text))


def _clean_price_from_match(match: re.Match | None, context: str | None) -> float | None:
    if not match:
        return None
    price = _safe_float(match.group(1) or match.group(2))
    return None if _price_is_noise(price, context) else price


def _extract_purity_from_context(text: str | None) -> tuple[str | None, str]:
    if not text:
        return None, "NONE"
    clean = re.sub(r"\s+", " ", str(text))
    for m in PURITY_RE.finditer(clean):
        window = clean[max(0, m.start() - 90): min(len(clean), m.end() + 90)]
        if PURITY_NOISE_RE.search(window):
            continue
        purity = (m.group(1) or m.group(2) or "").replace(" ", "")
        try:
            val = float(purity.rstrip("%"))
        except ValueError:
            continue
        if 0 < val <= 100:
            return purity, "HIGH"
    if re.search(r"\d{1,3}(?:\.\d+)?\s*%", clean) and PURITY_NOISE_RE.search(clean[:5000]):
        return None, "REJECTED"
    return None, "NONE"


def _classify_product_form(title: str | None = None, url: str | None = None, text: str | None = None, pack_unit: str | None = None, raw: str | None = None) -> str:
    hay = " ".join(str(x or "") for x in [title, url, raw, (text or "")[:3000]]).lower()
    unit = str(pack_unit or "").strip()
    raw_text = str(raw or "").lower()
    if "reference standard" in hay or "analytical standard" in hay or "(standard)" in hay or "standard solution" in hay:
        return "standard/reference"
    if unit in {"mL", "L"}:
        return "solution"
    if unit.lower() in {"ug", "µg", "mg", "g", "kg"}:
        if re.search(r"\b\d+(?:\.\d+)?\s?m[lL]\s*(?:x|\*)?\s*\d+(?:\.\d+)?\s?mM\b|\b\d+(?:\.\d+)?\s?mM\s*(?:in\s*DMSO|x|\*)", raw_text):
            return "solution"
        return "solid/mass"
    if "in dmso" in hay or "mm in" in hay or " solution" in hay:
        return "solution"
    return "unknown"


def _url_is_search_like(url: str | None, title: str | None = None) -> bool:
    hay = f"{url or ''} {title or ''}".lower()
    return bool(SEARCH_LIKE_URL_RE.search(hay) or "search results" in hay or "résultats" in hay)


def _url_role(url: str | None, title: str | None = None) -> str:
    if _url_is_search_like(url, title) and not PRODUCT_URL_HINT_RE.search(str(url or "")):
        return "search_or_directory_page"
    if PRODUCT_URL_HINT_RE.search(str(url or "")):
        return "product_page"
    return "source_page"


def _json_loads_loose(raw: str) -> list[Any]:
    try:
        parsed = json.loads(raw.strip())
        return parsed if isinstance(parsed, list) else [parsed]
    except Exception:
        return []


def _walk_json(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _walk_json(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_json(item)


def _clean_text(html: str) -> tuple[str, str, BeautifulSoup]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else "Untitled page"
    soup_for_text = BeautifulSoup(html, "html.parser")
    for tag in soup_for_text(["script", "style", "noscript", "svg"]):
        tag.decompose()
    text = soup_for_text.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return title, text[:180_000], soup


def _best_product_url(soup: BeautifulSoup, final_url: str) -> str:
    candidates: list[str] = []
    for tag in soup.find_all("link", rel=re.compile("canonical", re.I)):
        if tag.get("href"):
            candidates.append(urljoin(final_url, tag.get("href")))
    for key in ["og:url", "twitter:url"]:
        tag = soup.find("meta", attrs={"property": key}) or soup.find("meta", attrs={"name": key})
        if tag and tag.get("content"):
            candidates.append(urljoin(final_url, tag.get("content")))
    for script in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
        for root in _json_loads_loose(script.get_text(" ", strip=True)):
            for node in _walk_json(root):
                if isinstance(node, dict) and node.get("url"):
                    candidates.append(urljoin(final_url, str(node.get("url"))))
    for candidate in candidates:
        if candidate.startswith("http") and not _url_is_search_like(candidate) and (PRODUCT_URL_HINT_RE.search(candidate) or not _url_is_search_like(final_url)):
            return candidate
    return final_url


def _fetch(url: str, timeout: int) -> tuple[requests.Response, str, str, BeautifulSoup]:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; CAS-Sourcing-MVP/12.0; human-reviewed procurement research)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }
    response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    title, text, soup = _clean_text(response.text)
    return response, title, text, soup


def _cas_identity_confidence(title: str, text: str, url: str, cas_number: str, structured_cas: bool = False) -> tuple[bool, str]:
    title = title or ""
    head = (text or "")[:35000]
    search_like = _url_is_search_like(url, title)
    if structured_cas:
        return True, "CAS confirmed in structured product data"
    if re.search(rf"(?i)\bCAS(?:\s*(?:No\.?|Number|#))?\s*[:\-]?\s*{re.escape(cas_number)}\b", head):
        return True, "CAS identity field on page"
    title_cas = CAS_CONTEXT_RE.search(title)
    if title_cas and title_cas.group(1) != cas_number:
        return False, f"title contains different CAS {title_cas.group(1)}"
    if cas_number in title and not search_like:
        return True, "CAS in product page title"
    if cas_number.lower() in (url or "").lower() and not search_like:
        return True, "CAS in product URL"
    if cas_number in head[:12000] and not search_like:
        return True, "CAS appears in product identity region"
    if search_like:
        return False, "CAS appears only on a search/results page; not accepted as product identity"
    return False, "requested CAS not confirmed in page identity"


def _parse_pack_from_any(value: Any) -> tuple[float | None, str | None]:
    if value is None:
        return None, None
    text = str(value).replace("μ", "u").replace("µ", "u")
    m = PACK_RE.search(text)
    if not m:
        sol = SOLUTION_PACK_RE.search(text)
        if sol:
            ml = sol.group(1) or sol.group(2)
            size = _safe_float(ml)
            return (size, "mL") if _pack_is_reasonable(size, "mL") else (None, None)
        return None, None
    size = _safe_float(m.group(1))
    unit = _normalize_unit(m.group(2))
    if not _pack_is_reasonable(size, unit):
        return None, None
    return size, unit


def _extract_base_signals(cas_number: str, url: str, response_url: str, title: str, text: str, soup: BeautifulSoup, discovery_snippet: str | None) -> dict[str, Any]:
    product_url = _best_product_url(soup, response_url)
    structured_cas = False
    structured_title = None
    structured_product_url = None
    for script in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
        for root in _json_loads_loose(script.get_text(" ", strip=True)):
            for node in _walk_json(root):
                if not isinstance(node, dict):
                    continue
                node_text = json.dumps(node, ensure_ascii=False)[:8000]
                if cas_number in node_text:
                    structured_cas = True
                    structured_title = node.get("name") or structured_title
                    if node.get("url"):
                        structured_product_url = urljoin(response_url, str(node.get("url")))
    if structured_product_url and not _url_is_search_like(structured_product_url):
        product_url = structured_product_url
    identity_ok, identity_reason = _cas_identity_confidence(title, text, product_url, cas_number, structured_cas=structured_cas)
    purity, purity_confidence = _extract_purity_from_context(text[:50000]) if identity_ok else (None, "NONE")
    stock_m = STOCK_RE.search(text[:50000]) or LOGIN_PRICE_RE.search(text[:50000])
    stock = stock_m.group(1).title() if stock_m else "Not visible"

    # Fallback low-confidence price/pack from CAS neighborhood only; structured rows are preferred later.
    fallback_price = None
    fallback_pack_size = None
    fallback_pack_unit = None
    fallback_raw = ""
    if identity_ok:
        for m in re.finditer(re.escape(cas_number), text, re.I):
            window = text[max(0, m.start() - 1200): min(len(text), m.end() + 2200)]
            pack_m = PACK_RE.search(window)
            price_m = PRICE_RE.search(window)
            price = _clean_price_from_match(price_m, window) if price_m else None
            if price and pack_m:
                fallback_pack_size = _safe_float(pack_m.group(1))
                fallback_pack_unit = _normalize_unit(pack_m.group(2))
                fallback_price = price
                fallback_raw = window[:1000]
                break

    snippet_price = extract_snippet_price(discovery_snippet or "") if identity_ok else None
    price_visibility_status = classify_price_visibility(
        listed_price=fallback_price,
        text=f"{text[:12000]} {discovery_snippet or ''} {fallback_raw}",
        snippet_price=snippet_price,
        extraction_status="success",
    )
    product_form = _classify_product_form(structured_title or title, product_url, text, fallback_pack_unit, fallback_raw)
    return {
        "product_url": product_url,
        "title": str(structured_title or title)[:300],
        "identity_ok": identity_ok,
        "identity_reason": identity_reason,
        "purity": purity,
        "purity_confidence": purity_confidence,
        "stock": stock,
        "fallback_price": fallback_price,
        "fallback_pack_size": fallback_pack_size,
        "fallback_pack_unit": fallback_pack_unit,
        "fallback_raw": fallback_raw,
        "price_visibility_status": price_visibility_status,
        "product_form": product_form,
    }


def _variant_rows_from_json_ld(soup: BeautifulSoup) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for script in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
        for root in _json_loads_loose(script.get_text(" ", strip=True)):
            for node in _walk_json(root):
                if not isinstance(node, dict):
                    continue
                offers = node.get("offers")
                if not offers:
                    continue
                offer_nodes = offers if isinstance(offers, list) else [offers] if isinstance(offers, dict) else []
                for offer in offer_nodes:
                    if not isinstance(offer, dict):
                        continue
                    raw = json.dumps(offer, ensure_ascii=False)[:1000]
                    price = _safe_float(offer.get("price") or offer.get("lowPrice") or offer.get("highPrice"))
                    if _price_is_noise(price, raw):
                        price = None
                    pack_size, pack_unit = _parse_pack_from_any(offer.get("sku") or offer.get("name") or offer.get("description") or offer.get("mpn"))
                    availability = offer.get("availability") or node.get("availability")
                    if price is None or pack_size is None:
                        continue
                    rows.append({
                        "method": "json_ld_offer_row",
                        "pack_size": pack_size,
                        "pack_unit": pack_unit,
                        "price": price,
                        "stock": str(availability).split("/")[-1].replace("InStock", "In Stock") if availability else "Not visible",
                        "raw": [raw],
                        "price_pairing_confidence": "HIGH",
                        "product_form": _classify_product_form(raw=raw, pack_unit=pack_unit),
                    })
    return rows


def _variant_rows_from_html_tables(soup: BeautifulSoup) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        headers: list[str] = []
        if rows:
            headers = [h.get_text(" ", strip=True).lower() for h in rows[0].find_all(["th", "td"])]
        for row in rows:
            cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
            row_text = " | ".join(cells)
            if len(row_text) < 3:
                continue
            pack_size, pack_unit = _parse_pack_from_any(row_text)
            price_m = PRICE_RE.search(row_text)
            price = _clean_price_from_match(price_m, row_text) if price_m else None
            if price is None and headers:
                for h, cell in zip(headers, cells):
                    if any(word in h for word in ["price", "usd", "cost", "amount"]):
                        candidate = _safe_float(cell)
                        price = None if _price_is_noise(candidate, row_text) else candidate
                        break
            if not pack_size or price is None:
                continue
            stock_m = STOCK_RE.search(row_text)
            rows_out.append({
                "method": "html_table_row",
                "pack_size": pack_size,
                "pack_unit": pack_unit,
                "price": price,
                "stock": stock_m.group(1).title() if stock_m else "Not visible",
                "raw": [row_text[:1000]],
                "price_pairing_confidence": "HIGH",
                "product_form": _classify_product_form(raw=row_text, pack_unit=pack_unit),
            })
    return rows_out


def _variant_rows_from_public_text(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    lower = text.lower()
    markers = ["size price stock", "pack size price", "price stock", "size /price /stock", "size, price, stock", "size price availability"]
    if not any(m in lower for m in markers):
        return []
    starts = [m.start() for m in re.finditer(r"(?i)(grouped product items|pack size\s+price|size\s*,?\s*price\s*,?\s*(?:stock|availability)|size\s*/price\s*/stock|size\s+price\s+stock)", text)]
    windows = [text[st:st + 7000] for st in starts[:4]] or [text[:18000]]
    row_re = re.compile(
        r"(?P<size>(?:\d+(?:\.\d+)?\s?(?:ug|µg|mg|g|kg|ml|mL|L)|\d+\s?mM\s*(?:\*|x)?\s*\d+\s?mL\s*(?:in\s*DMSO)?|\d+\s?mL\s*x\s*\d+\s?mM\s*(?:\(in\s*DMSO\))?))\s*[,/|]?\s*"
        r"(?:(?:USD|US\$)?\s*)?(?P<price>\$\s*[0-9][0-9,]*(?:\.\d{1,2})?|[0-9][0-9,]*(?:\.\d{1,2})?\s*USD)\s*[,/|]?\s*"
        r"(?P<stock>in\s*stock|out\s*of\s*stock|available|ships?\s*in\s*[^.;,|]{1,30}|\d+[- ]?\d+\s*(?:days|weeks))?",
        re.I,
    )
    rows: list[dict[str, Any]] = []
    for window in windows:
        clean = re.sub(r"\s+", " ", window.replace("μ", "u").replace("µ", "u"))
        for m in row_re.finditer(clean):
            raw = clean[max(0, m.start() - 120): min(len(clean), m.end() + 180)]
            pack_size, pack_unit = _parse_pack_from_any(m.group("size"))
            price_m = PRICE_RE.search(m.group("price"))
            price = _clean_price_from_match(price_m, raw) if price_m else _safe_float(m.group("price").replace("USD", "").replace("$", ""))
            if _price_is_noise(price, raw):
                price = None
            if price is None or not _pack_is_reasonable(pack_size, pack_unit):
                continue
            rows.append({
                "method": "public_price_text_row",
                "pack_size": pack_size,
                "pack_unit": pack_unit,
                "price": price,
                "stock": (m.group("stock") or "In stock").title(),
                "raw": [raw[:1000]],
                "price_pairing_confidence": "MEDIUM",
                "product_form": _classify_product_form(raw=raw, pack_unit=pack_unit),
            })
    return rows


def _dedupe_variant_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    method_priority = {"json_ld_offer_row": 4, "html_table_row": 3, "public_price_text_row": 2}
    form_priority = {"solid/mass": 3, "unknown": 2, "solution": 1, "standard/reference": 0}
    rows = sorted(
        rows,
        key=lambda r: (
            method_priority.get(r.get("method", ""), 0),
            form_priority.get(str(r.get("product_form") or "unknown"), 1),
        ),
        reverse=True,
    )
    for r in rows:
        key = (
            round(float(r.get("pack_size") or 0), 9),
            str(r.get("pack_unit") or "").lower(),
            round(float(r.get("price") or 0), 4),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _base_product_from_fetch(cas_number: str, url: str, supplier: str, response, title: str, text: str, soup: BeautifulSoup, discovery_title: str | None, discovery_snippet: str | None) -> ExtractedProductData:
    signals = _extract_base_signals(cas_number, url, response.url, title, text, soup, discovery_snippet)
    cas_exact = bool(signals["identity_ok"])
    price = signals["fallback_price"] if cas_exact else None
    pack_size = signals["fallback_pack_size"] if cas_exact else None
    pack_unit = signals["fallback_pack_unit"] if cas_exact else None
    raw = signals["fallback_raw"]
    evidence_bits = []
    confidence = 15
    if cas_exact:
        confidence += 30
        evidence_bits.append("requested CAS confirmed by page identity")
        evidence_bits.append(signals["identity_reason"])
    else:
        evidence_bits.append("requested CAS not confirmed by product identity; pricing ignored")
    if price is not None:
        confidence += 10
        evidence_bits.append("low-confidence fallback price extracted")
    if pack_size is not None:
        confidence += 10
        evidence_bits.append("fallback pack size extracted")
    if signals["purity"]:
        confidence += 10
        evidence_bits.append("trusted purity/assay context extracted")
    if signals["stock"] != "Not visible":
        confidence += 8
        evidence_bits.append("availability/quote language found")
    snippet_price = extract_snippet_price(discovery_snippet or "") if cas_exact else None
    if price is None and snippet_price is not None:
        evidence_bits.append("search snippet appears to contain a price; verify source manually")
    status = classify_price_visibility(price, f"{text[:12000]} {discovery_snippet or ''} {raw}", snippet_price, "success")
    product_url = signals["product_url"]
    catalog_number = extract_catalog_number(url, title, signals["title"], raw, discovery_title or "")
    return ExtractedProductData(
        supplier=supplier,
        title=signals["title"],
        cas_exact_match=cas_exact,
        purity=signals["purity"],
        pack_size=pack_size,
        pack_unit=pack_unit,
        listed_price_usd=price,
        stock_status=signals["stock"],
        product_url=product_url,
        extraction_status="success",
        confidence=min(confidence, 100),
        evidence="; ".join(evidence_bits),
        extraction_method="fallback_visible_text" if price is not None else "identity_and_status_only",
        raw_matches=raw[:2500],
        catalog_number=catalog_number,
        price_visibility_status=status,
        best_action=best_action_for_status(status),
        adapter_name=supplier,
        price_pairing_confidence="LOW" if price is not None else "NONE",
        product_form=signals["product_form"],
        purity_confidence=signals["purity_confidence"],
        url_role=_url_role(product_url, title),
        landing_url=response.url,
        canonical_product_url=canonicalize_url(product_url),
        price_noise_flag=False,
    )


def _build_variant_product(base: ExtractedProductData, row: dict[str, Any]) -> ExtractedProductData:
    method = row.get("method") or "structured_variant_row"
    raw = "\n---\n".join(row.get("raw", [])[:3])[:2500]
    price = row.get("price")
    price_noise_flag = _price_is_noise(price, raw)
    if price_noise_flag:
        price = None
    product_form = row.get("product_form") or _classify_product_form(base.title, base.product_url, raw=raw, pack_unit=row.get("pack_unit"))
    confidence = max(base.confidence, 88 if row.get("price_pairing_confidence") == "HIGH" else 78)
    evidence = base.evidence + f"; v12 quantity-price pair parsed from same structured row ({method})"
    if product_form in {"solution", "standard/reference"}:
        evidence += f"; v12 product-form guard: {product_form}"
    status = PRICE_PUBLIC if price is not None else base.price_visibility_status
    return ExtractedProductData(
        supplier=base.supplier,
        title=base.title,
        cas_exact_match=base.cas_exact_match,
        purity=base.purity,
        pack_size=row.get("pack_size"),
        pack_unit=row.get("pack_unit"),
        listed_price_usd=price,
        stock_status=row.get("stock") or base.stock_status,
        product_url=base.product_url,
        extraction_status=base.extraction_status,
        confidence=min(int(confidence), 100),
        evidence=evidence,
        extraction_method=method,
        raw_matches=raw,
        catalog_number=base.catalog_number,
        price_visibility_status=status,
        best_action=best_action_for_status(status),
        adapter_name=base.adapter_name,
        price_pairing_confidence=row.get("price_pairing_confidence") or "MEDIUM",
        product_form=product_form,
        purity_confidence=base.purity_confidence,
        purity_pass=base.purity_pass,
        url_role=base.url_role,
        landing_url=base.landing_url,
        canonical_product_url=base.canonical_product_url,
        price_noise_flag=price_noise_flag,
    )


def extract_product_rows_from_url(cas_number: str, url: str, timeout: int = 18, supplier_hint: str | None = None, discovery_title: str | None = None, discovery_snippet: str | None = None) -> list[ExtractedProductData]:
    supplier = supplier_hint or supplier_name_from_url(url)
    try:
        response, title, text, soup = _fetch(url, timeout)
    except Exception as exc:
        return [ExtractedProductData(
            supplier=supplier,
            title="Could not extract page",
            cas_exact_match=False,
            purity=None,
            pack_size=None,
            pack_unit=None,
            listed_price_usd=None,
            stock_status="Extraction failed",
            product_url=url,
            extraction_status=f"failed: {type(exc).__name__}",
            confidence=10,
            evidence="Page could not be fetched or parsed. Use source link for manual review.",
            extraction_method="fetch_failed",
            raw_matches="",
            catalog_number=None,
            price_visibility_status="Extraction failed",
            best_action="Open source manually",
            adapter_name=supplier,
            url_role=_url_role(url),
            landing_url=url,
            canonical_product_url=canonicalize_url(url),
        )]

    base = _base_product_from_fetch(cas_number, url, supplier, response, title, text, soup, discovery_title, discovery_snippet)
    if not base.cas_exact_match:
        return [replace(base, listed_price_usd=None, pack_size=None, pack_unit=None, price_pairing_confidence="NONE")]

    rows: list[dict[str, Any]] = []
    rows.extend(_variant_rows_from_json_ld(soup))
    rows.extend(_variant_rows_from_html_tables(soup))
    rows.extend(_variant_rows_from_public_text(text))
    rows = _dedupe_variant_rows(rows)
    if not rows:
        return [base]
    return [_build_variant_product(base, row) for row in rows]


def extract_product_data_from_url(cas_number: str, url: str, timeout: int = 18, supplier_hint: str | None = None, discovery_title: str | None = None, discovery_snippet: str | None = None) -> ExtractedProductData:
    return extract_product_rows_from_url(cas_number, url, timeout, supplier_hint, discovery_title, discovery_snippet)[0]
