//! Bootstrap persistent tables before clustering or training.
use crate::{database_name, EpochMeta};
use rbp_clustering::{Future, Lookup, Metric};
use rbp_database::Schema;
use rbp_nlhe::NlheProfile;
use std::sync::Arc;
use tokio_postgres::Client;

/// Create base training tables and stamp the target player count.
pub async fn run(client: &Arc<Client>, players: usize) {
    for ddl in [
        Lookup::creates(),
        Metric::creates(),
        Future::creates(),
        <NlheProfile as Schema>::creates(),
        <EpochMeta as Schema>::creates(),
        Lookup::indices(),
        Metric::indices(),
        Future::indices(),
        <NlheProfile as Schema>::indices(),
    ] {
        client
            .batch_execute(ddl)
            .await
            .expect("initialize schema");
    }
    stamp_players(client, players).await;
    log::info!(
        "database {:?} initialized for {} players",
        database_name(players),
        players
    );
}

pub async fn stamp_players(client: &Client, players: usize) {
    client
        .execute(
            "INSERT INTO epoch (key, value) VALUES ('players', $1)
             ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            &[&(players as i64)],
        )
        .await
        .expect("stamp players");
}

pub async fn stored_players(client: &Client) -> Option<usize> {
    client
        .query_opt(
            "SELECT value FROM epoch WHERE key = 'players'",
            &[],
        )
        .await
        .ok()
        .flatten()
        .map(|row| row.get::<_, i64>(0) as usize)
}
