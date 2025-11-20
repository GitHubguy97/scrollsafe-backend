-- Postgres schema for doomscroller pipeline

CREATE TABLE IF NOT EXISTS videos (
    platform        TEXT        NOT NULL,
    video_id        TEXT        NOT NULL,
    first_seen_at   TIMESTAMPTZ NOT NULL,
    last_seen_at    TIMESTAMPTZ NOT NULL,
    title           TEXT,
    channel         TEXT,
    published_at    TIMESTAMPTZ,
    region          TEXT,
    source_url      TEXT,
    views_per_hour  DOUBLE PRECISION,
    status          TEXT DEFAULT 'active',
    PRIMARY KEY (platform, video_id)
);

CREATE TABLE IF NOT EXISTS analyses (
    platform        TEXT        NOT NULL,
    video_id        TEXT        NOT NULL,
    analyzed_at     TIMESTAMPTZ NOT NULL,
    label           TEXT        NOT NULL,
    confidence      NUMERIC(5,4),
    reason          TEXT,
    features        JSONB,
    model_version   TEXT        NOT NULL,
    frame_policy    TEXT        NOT NULL,
    batch_time_ms   INTEGER,
    frames_count    INTEGER,
    source_url      TEXT,
    PRIMARY KEY (platform, video_id)
);

CREATE INDEX IF NOT EXISTS idx_analyses_label_time
    ON analyses (label, analyzed_at DESC);

CREATE INDEX IF NOT EXISTS idx_analyses_time
    ON analyses (analyzed_at DESC);

CREATE TABLE IF NOT EXISTS admin_labels (
    platform     TEXT        NOT NULL,
    video_id     TEXT        NOT NULL,
    label        TEXT        NOT NULL,
    notes        TEXT,
    source_url   TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (platform, video_id)
);

CREATE INDEX IF NOT EXISTS idx_admin_labels_created
    ON admin_labels (created_at DESC);
