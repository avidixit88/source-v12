# CAS Sourcing MVP v12

Streamlit MVP for CAS-based supplier discovery, public catalog price extraction, trust classification, and desired-quantity scale-up modeling.

## What v12 adds

- Keeps prior trust controls: CAS identity gating, strict mass/solution/reference-standard classification, price-noise filtering, and required-purity checks.
- Fixes duplicate mass rows that were incorrectly marked as solution products when a DMSO solution appeared elsewhere in the same supplier table.
- Adds a smarter desired-quantity estimator:
  - fits supplier/product catalog price ladders on a log-log total-price curve,
  - anchors on the largest public pack,
  - blends observed supplier behavior with published chemical quantity-discount priors,
  - adds specialty-chemical and extrapolation-risk guardrails,
  - outputs a buyer-aggressive lower case, base RFQ target, supplier-conservative upper case, and retail catalog ceiling.
- Adds a Desired-Quantity Model CSV export.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Suggested tests

- `487-41-2` with 1 kg or 50 g desired quantity.
- `151-21-3` for public catalog behavior.
- `64-17-5` for commodity/log-in source behavior.

Visible catalog prices are evidence. Desired-quantity estimates are procurement models. Confirmed RFQ pricing is truth.
