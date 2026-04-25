# Data Dictionary — IEEE-CIS Fraud Detection

This document describes every feature group in the IEEE-CIS dataset, the
column shape it occupies in the raw CSV, and the closest analogue in a
production fraud-detection system. It is the reference used by every
feature generator and lineage test from Sprint 1 onwards.

Source: [Kaggle — IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection).
Provider: Vesta Corporation, 2019. License: Kaggle competition terms.
Snapshot this project targets: fingerprinted in `data/raw/MANIFEST.json`
(SHA-256 per file).

## 1. Dataset layout

| Table | Rows (train) | Columns | Join key |
|---|---:|---:|---|
| `train_transaction.csv` | 590,540 | 394 | `TransactionID` |
| `train_identity.csv` | ~144,233 | 41 | `TransactionID` |
| Merged (left-join) | 590,540 | ~433 | — |

`test_*` tables follow the same column shape and are used for
Kaggle-style hold-out evaluation only; this project uses a temporal
split of the train set as the primary evaluation cut.

## 2. Headline statistics

| Indicator | Value |
|---|---|
| Fraud rate (isFraud mean) | 3.5% |
| Identity coverage (any `id_*` present) | ~24% |
| TransactionDT span | ~6 months of seconds |
| Columns >50% null | ~200 (dominated by the V-block) |

These are the numbers that gate the lineage tests in
`tests/lineage/test_raw_lineage.py`.

## 3. Feature groups (transaction)

### 3.1 Identity, target, time, amount

| Column | Dtype | Nullable | Production analogue |
|---|---|---|---|
| `TransactionID` | int64 | no | Internal transaction UUID / event ID |
| `isFraud` | int64 {0,1} | no | Chargeback-confirmed fraud label |
| `TransactionDT` | int64 (seconds) | no | Unix timestamp of the authorisation request |
| `TransactionAmt` | float64 (USD) | no | Authorised amount; sometimes FX-converted to USD |
| `ProductCD` | object (W/H/C/R/S) | no | Merchant product classifier — vertical taxonomy |

`TransactionDT` is anchored to an anonymised base date; only deltas
within the dataset are meaningful. All temporal splits use this column.

### 3.2 Card columns (card1–card6)

| Column | Dtype | Meaning |
|---|---|---|
| `card1` | int64 | Anonymised payment card ID — highest-cardinality key; used as an entity in velocity and graph features. |
| `card2` | float64 | Card issuer code. |
| `card3` | float64 | Card network code. |
| `card4` | object | Card brand (`visa`, `mastercard`, `american express`, `discover`). |
| `card5` | float64 | Issuer category. |
| `card6` | object | Card type (`credit`, `debit`, `debit or credit`, `charge card`). |

**Production analogue:** payment-instrument enrichment returned by the
issuer BIN table and the acquirer processor (Stripe's `card.brand`,
Adyen's `funding_source`).

### 3.3 Address & distance

| Column | Dtype | Meaning |
|---|---|---|
| `addr1` | float64 | Billing address (anonymised zip/postal). |
| `addr2` | float64 | Billing address country. |
| `dist1` | float64 | Distance between two address pairs (e.g. billing vs shipping). |
| `dist2` | float64 | Secondary distance feature. |

**Production analogue:** AVS match score + ship-to/bill-to geodesic
distance from a maps API.

### 3.4 Email domains

| Column | Dtype | Meaning |
|---|---|---|
| `P_emaildomain` | object | Purchaser email domain (e.g. `gmail.com`). |
| `R_emaildomain` | object | Recipient email domain. |

**Production analogue:** email-reputation vendor output (Emailage,
Maxmind), downcast to domain only to preserve k-anonymity.

### 3.5 C1–C14 — count features

`float64` columns, nullable. Vesta's internal counters over entities
related to each transaction (cards, addresses, devices). Example
documented by the dataset author: "how many addresses are found
associated with the payment card."

**Production analogue:** Redis-backed velocity counters incremented at
transaction time on `hash(card_id)` / `hash(device_id)` keys.

### 3.6 D1–D15 — delta features

`float64` columns, nullable. Elapsed-time features — e.g. "days since
previous transaction on this card." Large missing rate on `D5–D15`.

**Production analogue:** streaming deltas computed by Flink / Kafka
Streams jobs and cached to Redis for online lookup.

### 3.7 M1–M9 — match flags

`object` columns. M1, M2, M3, M5, M6, M7, M8, M9 are binary match
flags with values in `{T, F}` (plus NaN) — e.g. "does the card name
match the billing address name?". **M4 is the exception:** a
three-way match indicator with values in `{M0, M1, M2}` (plus NaN).
The schema (`src/fraud_engine/schemas/raw.py`) validates the two
groups separately so a drift in either is caught at ingest.

**Production analogue:** issuer response bits (`nameMatchIndicator`)
and internal-rules outputs (`billing_ship_match`).

### 3.8 V1–V339 — Vesta engineered features

`float64` columns, nullable. Fully anonymised features engineered by
Vesta's internal fraud system: Vesta describes them as a blend of
count, time-delta, and rule-score signals.

**Production analogue:** proprietary rule-engine scores and entity
aggregates — the numeric surface the production model actually
consumes, once anonymised. The block is treated as "black-box
signals" by every feature generator in this project: we never try to
reverse-engineer them, but we do measure their drift over time as a
guardrail in Sprint 4.

## 4. Feature groups (identity)

### 4.1 Core

| Column | Dtype | Meaning |
|---|---|---|
| `TransactionID` | int64 | Foreign key onto `train_transaction`. |
| `DeviceType` | object | `mobile` / `desktop`. |
| `DeviceInfo` | object | User-agent / device model string (e.g. `Windows`, `iOS Device`, `SM-G935V`). |

### 4.2 id_01–id_11 — numeric

`float64` columns, nullable. Session-scoped numeric features — login
count, page-view count, anonymised device fingerprint distances.

**Production analogue:** device-fingerprint vendor output (FingerprintJS,
DeviceAtlas) plus session-clickstream counters.

### 4.3 id_12–id_38 — mixed numeric + categorical

Despite their shared `id_` prefix, this block splits cleanly along
dtype lines against the real Kaggle CSVs. The schema enforces the
split explicitly:

- **Numeric (`float64`):** `id_13`, `id_14`, `id_17`, `id_18`,
  `id_19`, `id_20`, `id_21`, `id_22`, `id_24`, `id_25`, `id_26`,
  `id_32`. These are numeric codes (screen-resolution hash buckets,
  timezone offsets, anonymised session counters) that carry `NaN`s,
  which is why pandas promotes them to float.
- **Categorical (`object`):** `id_12`, `id_15`, `id_16`, `id_23`,
  `id_27`–`id_31`, `id_33`–`id_38`. `id_30` is OS version, `id_31`
  is browser, `id_33` is screen resolution, `id_35`–`id_38` are
  booleans recorded as `T`/`F`.

**Production analogue:** parsed HTTP headers and JS-emitted client
fingerprint attributes; downstream the categorical id_* columns become
target-encoded features (OOF encoding, Sprint 2).

## 5. Null semantics

Missing values are information in this dataset — "absence of
identity" is correlated with fraud. Feature generators never impute
blindly; instead they:

1. Add `has_id = (any id_* not null)` as an explicit feature.
2. Preserve NaN for LightGBM, which handles missing natively.
3. For sklearn-requiring models (calibration, the NN head), impute
   with group-specific fill values learned OOF.

## 6. Temporal integrity

`TransactionDT` monotonically increases within each source file. The
primary evaluation split in Sprint 1 is a strict temporal cut: train
on the first ~80% of rows sorted by `TransactionDT`, validate on the
next ~10%, hold out the final ~10%. No shuffling. No random splits.
This is tested by the temporal-integrity lineage contract in every
feature module.

## 7. Where the schemas live

- `src/fraud_engine/schemas/raw.py` — pandera contracts for
  `train_transaction.csv`, `train_identity.csv`, and their left-join.
- `configs/schemas.yaml` — version registry that points at the above.
- `data/raw/MANIFEST.json` — SHA-256 fingerprint of every CSV.

Schema changes require a `SCHEMA_VERSION` bump, a module-docstring
history entry, and a migration test. See CLAUDE.md §7.1.
