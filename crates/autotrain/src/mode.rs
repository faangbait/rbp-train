//! Training mode selection from command line arguments.
use crate::*;
use rbp_database::Check;
use rbp_database::Schema;
use rbp_nlhe::NlheProfile;

/// Training mode parsed from command line arguments
pub enum Mode {
    Status,
    Initialize,
    Cluster,
    Fast,
    Slow,
    Reset,
}

impl Mode {
    pub fn from_args() -> Self {
        std::env::args()
            .find_map(|a| match a.as_str() {
                "--cluster" => Some(Self::Cluster),
                "--initialize" => Some(Self::Initialize),
                "--status" => Some(Self::Status),
                "--fast" => Some(Self::Fast),
                "--slow" => Some(Self::Slow),
                "--reset" => Some(Self::Reset),
                _ => None,
            })
            .unwrap_or_else(|| {
                eprintln!(
                    "Usage: trainer --status | --initialize | --cluster | --fast | --slow | --reset"
                );
                eprintln!(
                    "Env: DB_URL (required, e.g. .../{}), PLAYERS (required, 2..=9)",
                    database_name(3)
                );
                std::process::exit(1);
            })
    }

    pub async fn run() {
        let mode = Self::from_args();
        let players = players_from_env();
        rbp_core::init_players(players);
        let client = rbp_database::db().await;
        match mode {
            Self::Initialize => initialize::run(&client, players).await,
            Self::Fast => {
                Self::ensure_db_players(&client, players).await;
                FastSession::new(client).await.train().await;
            }
            Self::Slow => {
                Self::ensure_db_players(&client, players).await;
                SlowSession::new(client).await.train().await;
            }
            Self::Reset => Self::reset(&client).await,
            Self::Status => {
                Self::ensure_db_players(&client, players).await;
                client.status().await;
            }
            Self::Cluster => {
                Self::ensure_db_players(&client, players).await;
                PreTraining::run(&client).await;
            }
        }
    }

    async fn ensure_db_players(client: &tokio_postgres::Client, players: usize) {
        match initialize::stored_players(client).await {
            Some(stored) if stored == players => {}
            Some(stored) => {
                eprintln!(
                    "database stamped for {stored} players but PLAYERS={players}; use DB_URL=.../{} or re-run --initialize",
                    database_name(players)
                );
                std::process::exit(1);
            }
            None => {
                eprintln!("database not initialized; run trainer --initialize first");
                std::process::exit(1);
            }
        }
    }

    async fn reset(client: &tokio_postgres::Client) {
        log::info!("Truncating blueprint table...");
        client
            .execute(<NlheProfile as Schema>::truncates(), &[])
            .await
            .expect("truncate blueprint");
        log::info!("Resetting epoch counter...");
        client
            .execute(<EpochMeta as Schema>::truncates(), &[])
            .await
            .expect("reset epoch");
        log::info!("Reset complete.");
    }
}
