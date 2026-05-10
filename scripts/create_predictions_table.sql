-- Sprint 5 prompt 5.2.a — predictions audit-log schema.
--
-- Mirrors `src/fraud_engine/api/schemas.py:PredictionResponse` (the canonical
-- column list) plus two audit columns (`id` BIGSERIAL primary key,
-- `created_at` server-side timestamp).
--
-- Idempotent: every CREATE TABLE / CREATE INDEX uses IF NOT EXISTS so this
-- file can be re-run safely. Production deployment runs this once via
-- `psql -f scripts/create_predictions_table.sql $POSTGRES_URL`; the
-- 5.2.a integration test fixture runs it on every test-module setup.
--
-- Sprint 5.2.b will wire `PredictionLogger.log(response)` into main.py's
-- /predict route handler; this file establishes the schema that the
-- logger writes against.
--
-- Cross-references:
--   - src/fraud_engine/api/schemas.py:693-819 (PredictionResponse)
--   - src/fraud_engine/api/schemas.py:268-333 (Reason — JSONB sub-schema)
--   - src/fraud_engine/api/prediction_logger.py (the writer)
--   - sprints/sprint_5/prompt_5_2_a_report.md (9 design decisions)

-- ---------------------------------------------------------------------
-- Main table.
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS predictions (
    -- Audit columns.
    id              BIGSERIAL    PRIMARY KEY,
    -- Server-side wall-clock; the request's `latency_ms` is a separate
    -- per-request measurement. Default NOW() so the writer doesn't have
    -- to plumb a clock through.
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    -- Mirrored from PredictionResponse (8 fields).
    request_id      UUID         NOT NULL,
    txn_id          BIGINT       NOT NULL,
    -- Optional client identifier from RequestMetadata.client_id (e.g.
    -- "wealthsimple-prod"). Nullable; older requests may not carry it.
    client_id       TEXT         NULL,
    score           DOUBLE PRECISION NOT NULL,
    -- TEXT + CHECK rather than ENUM: matches Pydantic's
    -- Literal["block","allow"] without forcing a CREATE TYPE migration
    -- when Sprint 5.x adds a "review" three-way decision.
    decision        TEXT         NOT NULL CHECK (decision IN ('block', 'allow')),
    -- top_reasons is a JSON array of {feature_name, contribution, direction}
    -- objects (max 10 per PredictionResponse). JSONB allows in-Postgres
    -- analytics via jsonb_array_elements without needing a child table.
    top_reasons     JSONB        NOT NULL DEFAULT '[]'::jsonb,
    -- ge=0.0 in the schema → CHECK at the database boundary too.
    latency_ms      DOUBLE PRECISION NOT NULL CHECK (latency_ms >= 0),
    -- max_length=128 in the schema; TEXT in Postgres (no length cap)
    -- since the schema is already enforcing the limit at ingest.
    model_version   TEXT         NOT NULL,
    -- Default FALSE so a legacy log row without explicit value is
    -- treated as "non-degraded".
    degraded_mode   BOOLEAN      NOT NULL DEFAULT FALSE
);

COMMENT ON TABLE  predictions                  IS 'Audit log of every /predict response (Sprint 5.2.a).';
COMMENT ON COLUMN predictions.id               IS 'Monotonic primary key. BIGSERIAL → safe partitioning later.';
COMMENT ON COLUMN predictions.created_at       IS 'Server-side wall-clock at insert time (NOT request_start).';
COMMENT ON COLUMN predictions.request_id       IS 'UUID bound by the API middleware; correlates structlog lines.';
COMMENT ON COLUMN predictions.txn_id           IS 'TransactionID echoed from the request body.';
COMMENT ON COLUMN predictions.client_id        IS 'Optional client identifier from RequestMetadata.client_id.';
COMMENT ON COLUMN predictions.score            IS 'Calibrated fraud probability in [0, 1] (post-isotonic).';
COMMENT ON COLUMN predictions.decision         IS 'Server decision; block iff score >= Settings.decision_threshold.';
COMMENT ON COLUMN predictions.top_reasons      IS 'SHAP top-k contributions: list of {feature_name, contribution, direction}.';
COMMENT ON COLUMN predictions.latency_ms       IS 'End-to-end /predict latency in ms (server-side, excludes network round-trip).';
COMMENT ON COLUMN predictions.model_version    IS 'SHA-256 content_hash from the model manifest (or future MLflow run_id).';
COMMENT ON COLUMN predictions.degraded_mode    IS 'TRUE iff Tier-1-only fallback because Redis/Postgres was unreachable.';

-- ---------------------------------------------------------------------
-- Indexes covering the typical analytical query patterns.
-- ---------------------------------------------------------------------

-- "Recent predictions" / per-day analytics.
CREATE INDEX IF NOT EXISTS predictions_created_at_desc_idx
    ON predictions (created_at DESC);

-- "Look up the prediction with this UUID" (debugging / tracing).
-- NOT UNIQUE on purpose: a rare double-log (e.g., from a retry) should
-- not crash the writer; better to store both rows for forensic analysis.
CREATE INDEX IF NOT EXISTS predictions_request_id_idx
    ON predictions (request_id);

-- "Look up predictions for this transaction" (audit trail).
CREATE INDEX IF NOT EXISTS predictions_txn_id_idx
    ON predictions (txn_id);

-- "Block rate over time" / decision distribution analytics.
-- Composite (decision, created_at DESC) → time-windowed scans by decision.
CREATE INDEX IF NOT EXISTS predictions_decision_created_at_idx
    ON predictions (decision, created_at DESC);

-- "Per-model-version analytics" / regression checks.
CREATE INDEX IF NOT EXISTS predictions_model_version_idx
    ON predictions (model_version);
