from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from core.pricing import (
    normalize_price_points,
    quantity_to_grams,
    choose_anchor_price,
    estimate_bulk_price,
)
from core.supplier_engine import find_suppliers_by_cas, supplier_search_links
from core.live_supplier_engine import discover_live_suppliers
from core.procurement_logic import enrich_procurement_trust
from core.quantity_model import build_quantity_models
from core.ranking import rank_supplier_rows
from utils.validation import is_valid_cas

st.set_page_config(
    page_title="CAS Sourcing MVP v12",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 CAS Sourcing & Procurement Intelligence MVP v12")
st.caption(
    "CAS input → supplier discovery → trusted catalog evidence → desired-quantity model → RFQ-ready shortlist. "
    "v12 adds procurement-grade desired-quantity scale-up estimates using observed catalog curves, literature priors, and specialty-chemical risk guardrails."
)

with st.sidebar:
    st.header("Search Inputs")
    cas_number = st.text_input("CAS Number", value="103-90-2", help="Example test CAS: 103-90-2")
    chemical_name = st.text_input("Chemical Name Optional", value="Acetaminophen")
    desired_quantity = st.number_input("Desired Quantity", min_value=0.0001, value=1.0, step=0.5)
    desired_unit = st.selectbox("Desired Unit", ["g", "kg", "mg"], index=1)
    required_purity = st.text_input("Required Purity / Grade", value="98%+")

    st.divider()
    st.header("Data Mode")
    data_mode = st.radio(
        "Supplier data source",
        ["Stable mock data", "Live supplier discovery"],
        index=0,
        help="Mock mode preserves the v1 baseline. Live mode uses search/direct supplier pages and conservative extraction.",
    )

    max_pages = 30
    max_suppliers = 24
    pages_per_supplier = 2
    serpapi_key = ""
    include_direct_links = True
    if data_mode == "Live supplier discovery":
        max_pages = st.slider("Max total pages to extract", min_value=6, max_value=60, value=30)
        max_suppliers = st.slider("Max suppliers to walk", min_value=6, max_value=36, value=24)
        pages_per_supplier = st.slider("Pages per supplier", min_value=1, max_value=5, value=2)
        include_direct_links = st.checkbox("Include direct supplier search links", value=True)
        serpapi_key = st.text_input(
            "SerpAPI key optional",
            value=st.secrets.get("SERPAPI_KEY", "") if hasattr(st, "secrets") else "",
            type="password",
            help="Optional. If empty, the system still shows/extracts from known direct supplier search pages.",
        )

    run_search = st.button("Run CAS Sourcing Search", type="primary")

st.info(
    "Procurement rule: visible catalog prices are evidence. Bulk prices are estimates. RFQ pricing is confirmed truth. "
    "Live extraction is intentionally conservative and every result keeps its source URL for auditability."
)


def render_supplier_table(ranked: pd.DataFrame) -> None:
    preferred_cols = [
        "supplier",
        "chemical_name",
        "cas_number",
        "cas_exact_match",
        "region",
        "purity",
        "purity_value_pct",
        "purity_pass",
        "product_form",
        "page_type",
        "pack_size",
        "pack_unit",
        "listed_price_usd",
        "snippet_price_usd",
        "price_per_g",
        "price_visibility_status",
        "best_action",
        "catalog_number",
        "stock_status",
        "score",
        "extraction_confidence",
        "extraction_method",
        "price_pairing_confidence",
        "bulk_estimate_eligible",
        "procurement_trust_decision",
        "trust_warning",
        "ranking_reason",
        "notes",
        "product_url",
    ]
    cols = [c for c in preferred_cols if c in ranked.columns]
    st.dataframe(ranked[cols], width='stretch', hide_index=True)



def render_supplier_cards(summary_df: pd.DataFrame) -> None:
    st.markdown(
        """
        <style>
        .supplier-card {border:1px solid rgba(49,51,63,.15); border-radius:22px; padding:18px 20px; margin:10px 0; background:linear-gradient(180deg,#ffffff 0%,#fbfbfd 100%); box-shadow:0 8px 24px rgba(0,0,0,.04);} 
        .supplier-title {font-size:1.08rem; font-weight:700; margin-bottom:2px;}
        .supplier-meta {font-size:.86rem; color:#667085; margin-bottom:10px;}
        .pill {display:inline-block; border-radius:999px; padding:4px 9px; font-size:.78rem; margin-right:6px; background:#f2f4f7; color:#344054;}
        .pill-good {background:#ecfdf3; color:#027a48;}
        .pill-warn {background:#fffaeb; color:#b54708;}
        .pill-info {background:#eff8ff; color:#175cd3;}
        .metric-line {font-size:.92rem; margin-top:6px; color:#344054;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    for _, row in summary_df.head(12).iterrows():
        price_count = int(row.get("visible_price_count") or 0)
        status = str(row.get("price_visibility_status") or "No public price detected")
        if price_count > 0:
            price_pill = f'<span class="pill pill-good">{price_count} public price point(s)</span>'
        elif "Login" in status:
            price_pill = '<span class="pill pill-warn">login/account price</span>'
        elif "Quote" in status:
            price_pill = '<span class="pill pill-warn">quote/RFQ</span>'
        else:
            price_pill = '<span class="pill">no public price yet</span>'
        cas_pill = '<span class="pill pill-good">CAS confirmed</span>' if bool(row.get("cas_exact_match")) else '<span class="pill pill-warn">CAS unconfirmed</span>'
        best_price = row.get("best_visible_price_usd")
        best_price_text = f"${float(best_price):,.2f}" if pd.notna(best_price) else "Not public"
        url = str(row.get("representative_url") or "")
        open_link = f'<a href="{url}" target="_blank">Open source</a>' if url else ""
        st.markdown(
            f"""
            <div class="supplier-card">
              <div class="supplier-title">{row.get('supplier')}</div>
              <div class="supplier-meta">{row.get('source_tier','unknown')} · confidence {row.get('max_extraction_confidence','')}</div>
              <div>{cas_pill}{price_pill}<span class="pill pill-info">{row.get('products_found',0)} product page(s)</span></div>
              <div class="metric-line"><b>Best visible price:</b> {best_price_text} · <b>Action:</b> {row.get('best_action','Check source')}</div>
              <div class="metric-line"><b>Price pairing confidence:</b> HIGH {row.get('high_confidence_price_pairs',0)} · MEDIUM {row.get('medium_confidence_price_pairs',0)} · <b>Curve-eligible:</b> {row.get('bulk_estimate_eligible_count',0)}</div>
              <div class="metric-line"><b>Forms:</b> {row.get('product_forms','unknown')} · <b>Packs:</b> {row.get('pack_options','Not visible')}</div>
              <div class="metric-line"><b>Purity:</b> {row.get('purities_found','Not visible')} · <b>Stock:</b> {row.get('stock_summary','Not visible')}</div>
              <div class="metric-line"><b>Trust:</b> {row.get('trust_decisions','')}</div>
              <div class="metric-line">{open_link}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
def render_price_and_bulk_sections(ranked: pd.DataFrame, desired_qty_g: float, desired_quantity: float, desired_unit: str) -> None:
    st.subheader("2. Catalog Price Evidence")
    visible = ranked[ranked["has_visible_price"]].copy() if "has_visible_price" in ranked.columns else pd.DataFrame()
    if visible.empty:
        st.warning(
            "No visible prices were extracted from the current data. This usually means the supplier hides pricing behind login, JavaScript, account-specific contracts, or quote gates. "
            "The system can still provide supplier/source links, but the quantity model needs at least one visible mass-based price point."
        )
        return

    trusted_visible = visible[visible.get("bulk_estimate_eligible", pd.Series(False, index=visible.index)).fillna(False).astype(bool)].copy() if "bulk_estimate_eligible" in visible.columns else visible
    chart_df = trusted_visible if not trusted_visible.empty else visible
    chart_df = chart_df[chart_df.get("price_per_g", pd.Series(dtype=float)).notna()].sort_values("price_per_g")
    if not chart_df.empty:
        fig = px.bar(
            chart_df,
            x="supplier",
            y="price_per_g",
            hover_data=[c for c in ["pack_size", "pack_unit", "listed_price_usd", "purity", "product_form", "procurement_trust_decision", "product_url"] if c in chart_df.columns],
            title="Trusted Mass-Based Catalog Price Normalized to $/g",
        )
        st.plotly_chart(fig, width='stretch')

    evidence_cols = [c for c in [
        "supplier", "page_title", "product_form", "page_type", "pack_size", "pack_unit",
        "listed_price_usd", "pack_size_g", "price_per_g", "purity", "purity_pass",
        "price_pairing_confidence", "bulk_estimate_eligible", "procurement_trust_decision",
        "trust_warning", "product_url"
    ] if c in visible.columns]
    st.dataframe(visible[evidence_cols], width='stretch', hide_index=True)

    st.subheader("3. Desired Quantity Scale-Up Model")
    quantity_models = build_quantity_models(ranked, desired_qty_g)
    if quantity_models.empty:
        st.warning(
            "No rows were eligible for the desired-quantity model. The model requires a CAS-confirmed product page, mass-based units, HIGH/MEDIUM quantity-price pairing, no noisy price context, and purity that is not below the requested threshold."
        )
        return

    best = quantity_models.iloc[0]
    consensus = quantity_models[quantity_models["target_estimated_total_usd"].notna()].copy()
    median_base = float(consensus["target_estimated_total_usd"].median()) if not consensus.empty else float(best["target_estimated_total_usd"])
    median_low = float(consensus["estimate_range_low_usd"].median()) if "estimate_range_low_usd" in consensus.columns and not consensus.empty else float(best["estimate_range_low_usd"])
    median_high = float(consensus["estimate_range_high_usd"].median()) if "estimate_range_high_usd" in consensus.columns and not consensus.empty else float(best["estimate_range_high_usd"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Desired Quantity", f"{desired_quantity:g} {desired_unit}")
    c2.metric("Best RFQ Target", f"${best['target_estimated_total_usd']:,.2f}")
    c3.metric("Consensus Estimate", f"${median_base:,.2f}")
    c4.metric("Estimate Band", f"${median_low:,.0f}–${median_high:,.0f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Largest Public Pack", f"{best['max_pack_g']:.3g} g")
    c6.metric("Scale Gap", f"{best['scale_gap_x']:,.0f}×")
    c7.metric("Effective Exponent", f"{best['effective_total_price_exponent']:.2f}")
    c8.metric("Retail Catalog Ceiling", f"${best.get('retail_catalog_ceiling_usd', best.get('catalog_multiple_cost_usd', 0)):,.0f}")

    st.write(
        f"Best scale-up source: **{best['supplier']}** using **{int(best['price_points_used'])} paired price point(s)**. "
        f"Observed total-price exponent: **{best['observed_total_price_exponent']:.2f}**. "
        f"Effective v12 exponent: **{best['effective_total_price_exponent']:.2f}**. "
        f"Decision: **{best['quantity_decision']}**. Confidence: **{best['model_confidence']}**."
    )
    if str(best.get("extrapolation_risk", "")).lower() in {"high", "very high"}:
        st.warning(str(best["model_reason"]))
    else:
        st.success(str(best["model_reason"]))

    if len(quantity_models) > 1:
        st.info(
            f"Cross-supplier check: {len(quantity_models)} eligible supplier/product curve(s). "
            f"Median base estimate is ${median_base:,.2f}; median modeled band is ${median_low:,.2f}–${median_high:,.2f}. "
            "Use this as RFQ strategy guidance, not confirmed pricing."
        )

    model_cols = [
        "supplier", "product_title", "purity", "product_form", "product_complexity_class",
        "price_points_used", "min_pack_g", "max_pack_g", "pack_span_x", "scale_gap_x",
        "observed_total_price_exponent", "literature_unit_discount_gamma", "literature_total_price_exponent",
        "effective_total_price_exponent", "aggressive_total_price_exponent", "conservative_total_price_exponent",
        "curve_r2", "extrapolation_risk", "model_confidence", "catalog_supported_qty_g",
        "target_estimated_total_usd", "target_estimated_unit_price_per_g",
        "estimate_range_low_usd", "estimate_range_high_usd", "negotiation_anchor_usd",
        "retail_catalog_ceiling_usd", "best_catalog_pack_g", "best_catalog_pack_price_usd",
        "catalog_pack_count_for_desired", "suggested_rfq_tiers", "quantity_decision", "model_reason", "product_url"
    ]
    model_view = quantity_models[[c for c in model_cols if c in quantity_models.columns]]
    st.dataframe(model_view, width='stretch', hide_index=True)

    fig2 = px.bar(
        quantity_models.head(12),
        x="supplier",
        y="target_estimated_total_usd",
        hover_data=[c for c in ["price_points_used", "max_pack_g", "scale_gap_x", "effective_total_price_exponent", "estimate_range_low_usd", "estimate_range_high_usd", "model_confidence"] if c in quantity_models.columns],
        title="v12 Scale-Up RFQ Target Estimate by Supplier/Product",
    )
    st.plotly_chart(fig2, width='stretch')

    quantity_csv = quantity_models.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Desired-Quantity Model CSV",
        data=quantity_csv,
        file_name=f"cas_quantity_model_v12_{cas_number.replace('-', '_')}.csv",
        mime="text/csv",
    )


if run_search:
    cas_valid = is_valid_cas(cas_number)
    desired_qty_g = quantity_to_grams(desired_quantity, desired_unit)

    if not cas_valid:
        st.error("Invalid CAS number format or checksum. Please verify the CAS number.")
        st.stop()

    if desired_qty_g is None or desired_qty_g <= 0:
        st.error("Desired quantity must be convertible to grams and greater than zero.")
        st.stop()

    discovery_df = pd.DataFrame()

    if data_mode == "Stable mock data":
        raw_results = find_suppliers_by_cas(cas_number)
        discovery_df = supplier_search_links(cas_number).rename(columns={"search_url": "url"})
        discovery_df["title"] = discovery_df["supplier"] + " direct CAS search"
        discovery_df["snippet"] = "Manual supplier search link from v1 baseline."
        discovery_df["source"] = "v1_direct_link"
    else:
        with st.spinner("Running live supplier discovery, product-link expansion, and CAS-confirmed extraction..."):
            detail_results, discovery_df, supplier_summary_df = discover_live_suppliers(
                cas_number=cas_number,
                chemical_name=chemical_name,
                serpapi_key=serpapi_key,
                max_pages_to_extract=max_pages,
                include_direct_links=include_direct_links,
                max_suppliers=max_suppliers,
                pages_per_supplier=pages_per_supplier,
                required_purity=required_purity,
            )
        raw_results = detail_results

    if raw_results.empty:
        st.warning("No supplier rows were found or extracted yet for this CAS. Review the discovery/source links below.")
        if not discovery_df.empty:
            st.dataframe(discovery_df, width='stretch', hide_index=True)
        st.stop()

    raw_results = enrich_procurement_trust(raw_results, required_purity=required_purity)
    normalized = normalize_price_points(raw_results)
    normalized = enrich_procurement_trust(normalized, required_purity=required_purity)
    ranked = rank_supplier_rows(normalized)

    if "extraction_confidence" in ranked.columns:
        ranked = ranked.sort_values(["score", "extraction_confidence", "has_visible_price"], ascending=[False, False, False])

    st.subheader("1. Supplier Discovery")
    if data_mode == "Live supplier discovery":
        st.caption(
            "Live mode now expands supplier search pages into likely product-detail pages, then uses layered extraction with embedded-script parsing and a CAS-confirmation safety gate. It may still miss prices hidden behind account logins, heavy JavaScript, or quote gates. "
            "Use source URLs for human verification."
        )
    if data_mode == "Live supplier discovery" and "supplier_summary_df" in locals() and not supplier_summary_df.empty:
        summary_cols = [c for c in ["supplier", "cas_number", "cas_exact_match", "products_found", "catalog_numbers", "purities_found", "product_forms", "pack_options", "visible_price_count", "bulk_estimate_eligible_count", "best_visible_price_usd", "price_visibility_status", "best_action", "stock_summary", "max_extraction_confidence", "high_confidence_price_pairs", "medium_confidence_price_pairs", "trust_decisions", "representative_url"] if c in supplier_summary_df.columns]
        st.markdown("**Supplier-level cards**")
        render_supplier_cards(supplier_summary_df)
        with st.expander("Supplier summary table"):
            st.dataframe(supplier_summary_df[summary_cols], width='stretch', hide_index=True)
        with st.expander("Product-level extraction evidence"):
            render_supplier_table(ranked)
    else:
        render_supplier_table(ranked)

    render_price_and_bulk_sections(ranked, desired_qty_g, desired_quantity, desired_unit)

    st.subheader("4. Discovery / Source Links")
    st.caption("Every live result should remain auditable through its source URL.")
    if not discovery_df.empty:
        st.dataframe(discovery_df, width='stretch', hide_index=True)
    else:
        st.info("No separate discovery links were returned.")

    st.subheader("5. Export")
    export_df = ranked.copy()
    export_df["requested_quantity"] = desired_quantity
    export_df["requested_unit"] = desired_unit
    export_df["required_purity"] = required_purity
    export_df["data_mode"] = data_mode
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Product-Level Evidence CSV",
        data=csv,
        file_name=f"cas_product_evidence_v12_{cas_number.replace('-', '_')}.csv",
        mime="text/csv",
    )
    if data_mode == "Live supplier discovery" and "supplier_summary_df" in locals() and not supplier_summary_df.empty:
        summary_export = supplier_summary_df.copy()
        summary_export["requested_quantity"] = desired_quantity
        summary_export["requested_unit"] = desired_unit
        summary_export["required_purity"] = required_purity
        summary_csv = summary_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Supplier Summary CSV",
            data=summary_csv,
            file_name=f"cas_supplier_summary_v12_{cas_number.replace('-', '_')}.csv",
            mime="text/csv",
        )
else:
    st.subheader("How to test")
    st.markdown(
        """
        1. Keep the default CAS `103-90-2` for the first test.
        2. Start with **Stable mock data** to confirm the v1 baseline still works.
        3. Switch to **Live supplier discovery** for the first real-data test.
        4. Optional: add a SerpAPI key in the sidebar for broader search discovery.
        5. Review supplier ranking, extraction confidence, source links, visible price normalization, and bulk estimate scenarios.

        v12 keeps the baseline intact and adds trust classification plus scale-up pricing logic for desired quantities beyond the catalog ladder.
        """
    )
