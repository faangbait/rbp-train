//! Epoch metadata table schema

pub const PLAYERS_KEY: &str = "players";

/// Suggested Postgres database name for a given table size (`pluribus3`, `pluribus6`, …).
pub fn database_name(players: usize) -> String {
    format!("pluribus{players}")
}

/// Target table size from required `PLAYERS` env var (2..=9).
pub fn players_from_env() -> usize {
    let players = match std::env::var("PLAYERS") {
        Ok(value) => value
            .parse::<usize>()
            .unwrap_or_else(|_| panic!("PLAYERS must be an integer, got {value:?}")),
        Err(_) => {
            eprintln!("PLAYERS env var required (2..=9)");
            std::process::exit(1);
        }
    };
    validate_players_range(players);
    players
}

pub fn validate_players_range(players: usize) {
    if !(2..=rbp_core::MAX_N).contains(&players) {
        eprintln!(
            "PLAYERS must be between 2 and {}, got {players}",
            rbp_core::MAX_N
        );
        std::process::exit(1);
    }
}

/// Newtype wrapper for epoch counter (enables Schema implementation).
pub struct EpochMeta;

impl rbp_database::Schema for EpochMeta {
    fn name() -> &'static str {
        rbp_database::EPOCH
    }
    fn creates() -> &'static str {
        const_format::concatcp!(
            "CREATE TABLE IF NOT EXISTS ",
            rbp_database::EPOCH,
            " (
                key   TEXT PRIMARY KEY,
                value BIGINT NOT NULL
            );
            INSERT INTO ",
            rbp_database::EPOCH,
            " (key, value)
            VALUES ('current', 0)
            ON CONFLICT (key) DO NOTHING;"
        )
    }
    fn indices() -> &'static str {
        unimplemented!()
    }
    fn copy() -> &'static str {
        unimplemented!()
    }
    fn truncates() -> &'static str {
        const_format::concatcp!(
            "UPDATE ",
            rbp_database::EPOCH,
            " SET value = 0 WHERE key = 'current'"
        )
    }
    fn freeze() -> &'static str {
        unimplemented!()
    }
    fn columns() -> &'static [tokio_postgres::types::Type] {
        unimplemented!()
    }
}
