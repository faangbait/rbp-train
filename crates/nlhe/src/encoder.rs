use super::*;
use rbp_cards::*;
use rbp_gameplay::*;
use rbp_mccfr::*;

type NlheTree = Tree<NlheTurn, NlheEdge, NlheGame, NlheInfo>;

/// Encoder that maps poker game states to information set identifiers.
///
/// Maps suit-isomorphic hand representations ([`Isomorphism`]) to strategic
/// abstraction buckets ([`Abstraction`]) — output of the k-means clustering
/// pipeline.
///
/// # Storage
///
/// The river layer alone is ~123M isomorphisms. A per-process `BTreeMap`
/// costs several GB; with one worker process per table size that is paid 8×
/// for *identical* data (the card abstraction is independent of player count).
///
/// Instead the table is materialized once to a flat, sorted, read-only file
/// (`ISO_CACHE`, default `/tmp/rbp-iso.bin`) and `mmap`'d by every worker. The
/// OS shares the underlying physical pages across processes, so the abstraction
/// lookup costs RAM once regardless of how many workers run. Lookup is a binary
/// search over the memory-mapped `obs` slice.
///
/// File layout (native-endian, single-machine cache):
///   [u64 len][len × i64 obs, ascending][len × i16 abs, parallel]
#[derive(Default)]
pub struct NlheEncoder {
    mmap: Option<memmap2::Mmap>,
    len: usize,
}

impl NlheEncoder {
    /// `(obs, abs)` views into the mmap'd file. obs is sorted ascending; abs is
    /// parallel. Header is 8 bytes, so obs starts 8-aligned and abs starts
    /// `8 + len*8` (also a multiple of 8) — both correctly aligned for the cast.
    fn slices(&self) -> (&[i64], &[i16]) {
        let m = self.mmap.as_ref().expect("encoder not hydrated");
        let len = self.len;
        // ponytail: native-endian cache. Delete ISO_CACHE to rebuild if moved across arch.
        let obs = unsafe { std::slice::from_raw_parts(m[8..].as_ptr().cast::<i64>(), len) };
        let abs =
            unsafe { std::slice::from_raw_parts(m[8 + len * 8..].as_ptr().cast::<i16>(), len) };
        (obs, abs)
    }
    /// Looks up the abstraction bucket for an observation.
    ///
    /// Internally converts to canonical isomorphism for lookup.
    /// Panics if the isomorphism is not in the lookup table.
    pub fn abstraction(&self, obs: &Observation) -> Abstraction {
        let key = i64::from(Isomorphism::from(*obs));
        let (obs, abs) = self.slices();
        let i = obs
            .binary_search(&key)
            .expect("isomorphism not found in abstraction lookup");
        Abstraction::from(abs[i])
    }
    /// Creates an info set for the root game state.
    pub fn root(&self, game: &NlheGame) -> NlheInfo {
        let subgame = Path::default();
        let present = self.abstraction(&game.sweat());
        let choices = game.as_ref().choices(0);
        NlheInfo::from((subgame, present, choices))
    }
}

impl rbp_mccfr::Encoder for NlheEncoder {
    type T = NlheTurn;
    type E = NlheEdge;
    type G = NlheGame;
    type I = NlheInfo;
    fn seed(&self, root: &Self::G) -> Self::I {
        self.root(root)
    }
    fn info(&self, tree: &NlheTree, leaf: Branch<Self::E, Self::G>) -> Self::I {
        NlheInfo::from((self, tree, leaf))
    }
    fn resume(&self, past: &[Self::E], game: &Self::G) -> Self::I {
        // THERE MAY BE TRUNCATION HERE
        // BUT i think it's okay? SUBGAME_DEPTH ?
        let subgame = past.iter().map(|e| Edge::from(*e)).collect::<Path>();
        let present = self.abstraction(&game.sweat());
        let choices = game.as_ref().choices(subgame.aggression());
        NlheInfo::from((subgame, present, choices))
    }
}

#[cfg(feature = "database")]
#[async_trait::async_trait]
impl rbp_database::Hydrate for NlheEncoder {
    async fn hydrate(client: std::sync::Arc<tokio_postgres::Client>) -> Self {
        let path = std::env::var("ISO_CACHE").unwrap_or_else(|_| "/tmp/rbp-iso.bin".to_string());
        build_iso_cache_if_missing(&path, &client).await;
        log::info!("{:<32}{:<32}", "mmapping isomorphism", path.as_str());
        let file = std::fs::File::open(&path).expect("open iso cache");
        let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap iso cache") };
        let len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        Self {
            mmap: Some(mmap),
            len,
        }
    }
}

/// Materialize the isomorphism→abstraction table to a flat, sorted file the
/// workers can share via `mmap`. The data is identical across the `pluribusN`
/// databases (card abstraction is player-count independent), so the cache is
/// keyed only by `ISO_CACHE` path, not by table size.
///
/// Cross-process build race: the first worker to create `<path>.lock` builds;
/// the rest spin until the file appears. The file is written to a temp path and
/// atomically renamed so readers never see a partial cache.
#[cfg(feature = "database")]
async fn build_iso_cache_if_missing(path: &str, client: &tokio_postgres::Client) {
    use futures_util::{TryStreamExt, pin_mut};
    use std::io::Write;

    if std::path::Path::new(path).exists() {
        return;
    }
    let lock = format!("{path}.lock");
    if std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock)
        .is_err()
    {
        // Another worker is building. Wait for the finished file.
        // ponytail: stale lock (builder crashed) blocks startup forever; rm <path>.lock to recover.
        while !std::path::Path::new(path).exists() {
            std::thread::sleep(std::time::Duration::from_millis(250));
        }
        return;
    }

    log::info!("{:<32}{:<32}", "building isomorphism cache", path);
    let sql = const_format::concatcp!("SELECT obs, abs FROM ", rbp_database::ISOMORPHISM);
    // Stream rows: ~123M of them, so never buffer the heavy Row structs at once.
    let stream = client
        .query_raw(sql, std::iter::empty::<i32>())
        .await
        .expect("isomorphism query");
    pin_mut!(stream);
    let mut rows: Vec<(i64, i16)> = Vec::new();
    while let Some(row) = stream.try_next().await.expect("isomorphism row") {
        rows.push((row.get::<_, i64>(0), row.get::<_, i16>(1)));
    }
    rows.sort_unstable_by_key(|&(obs, _)| obs);

    let tmp = format!("{path}.tmp.{}", std::process::id());
    let mut w = std::io::BufWriter::new(std::fs::File::create(&tmp).expect("create iso tmp"));
    w.write_all(&(rows.len() as u64).to_le_bytes()).unwrap();
    for &(obs, _) in &rows {
        w.write_all(&obs.to_le_bytes()).unwrap();
    }
    for &(_, abs) in &rows {
        w.write_all(&abs.to_le_bytes()).unwrap();
    }
    w.flush().unwrap();
    drop(w);
    std::fs::rename(&tmp, path).expect("rename iso cache");
    let _ = std::fs::remove_file(&lock);
    log::info!("{:<32}{} isomorphisms", "built isomorphism cache", rows.len());
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // Round-trips the flat file format through mmap + slices + binary search.
    // Catches header/offset/alignment/endianness regressions without a database.
    #[test]
    fn iso_cache_roundtrip() {
        let pairs: [(i64, i16); 4] = [(-9, 1), (3, 2), (3000, 7), (1_i64 << 40, 9)];
        let path = std::env::temp_dir().join(format!("rbp-iso-test-{}.bin", std::process::id()));

        let mut w = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
        w.write_all(&(pairs.len() as u64).to_le_bytes()).unwrap();
        for &(obs, _) in &pairs {
            w.write_all(&obs.to_le_bytes()).unwrap();
        }
        for &(_, abs) in &pairs {
            w.write_all(&abs.to_le_bytes()).unwrap();
        }
        w.flush().unwrap();
        drop(w);

        let file = std::fs::File::open(&path).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
        let len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let enc = NlheEncoder {
            mmap: Some(mmap),
            len,
        };

        let (obs, abs) = enc.slices();
        assert_eq!(obs, &[-9, 3, 3000, 1_i64 << 40]);
        for &(k, v) in &pairs {
            let i = obs.binary_search(&k).expect("present key");
            assert_eq!(abs[i], v);
        }
        assert!(obs.binary_search(&4).is_err()); // absent key

        std::fs::remove_file(&path).ok();
    }
}
