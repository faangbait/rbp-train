//! Autotrain Binary
//!
//! Unified training pipeline with postgres as source of truth.
//!
//! Options: --status, --initialize, --fast, --slow, --cluster, --reset
//! Env: DB_URL (required, e.g. postgres://host), PLAYERS (required, 2..=9), TRAIN_DURATION

#[tokio::main]
async fn main() {
    rbp_core::log();
    rbp_core::kys();
    rbp_core::brb();
    rbp_autotrain::Mode::run().await;
}
