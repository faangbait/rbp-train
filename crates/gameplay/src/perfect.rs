//! Complete-information game history for **training time**.
//!
//! # Information Boundary
//!
//! | Type | Perspective | Context |
//! |------|-------------|---------|
//! | `Partial` | Hero only | Inference (strategy lookup) |
//! | `Perfect` | Both hands | Training (CFR traversal) |
//!
//! During CFR training, we traverse the game tree knowing both players' cards
//! (god's view), but strategies are indexed only by `NlheInfo` (public edges +
//! private bucket). `Perfect` stores the complete root state needed for reach
//! probability computation and counterfactual value calculation.
//!
//! # Conversions
//!
//! ```text
//! Perfect::from((partial, holes))  ───►  Perfect     (add N-1 opponents' info)
//!                                  ◄───
//! perfect.partial(hero)                              (erase opponent info)
//!
//! partial.histories() ─────►  Vec<(Obs, Perfect)>  (sample joint opponent deals)
//! ```
//!
//! # Blind Handling
//!
//! Like `Partial`, blinds are constant and NOT stored in `actions`.
//! The `root` field stores a POST-blind game state.
use super::*;
use rbp_cards::*;

/// Complete game history with both players' cards known.
///
/// Stores root game state (POST-blind, with all cards set) and action sequence
/// (excluding blinds). Game states are derived by applying actions to root.
#[derive(Debug, Clone)]
pub struct Perfect {
    root: Game,
    actions: Vec<Action>,
}

impl From<(&Partial, Vec<Hole>)> for Perfect {
    /// Creates history from partial with an assumed multiway opponent assignment.
    ///
    /// Hero is derived from `partial.turn()`. The root game has:
    /// - Hero's cards from `partial.seen()`
    /// - Each non-hero seat's cards from a distinct entry of `holes`
    ///   (consumed in ascending seat order; see [`Game::assume`])
    /// - Blinds already posted (POST-blind state)
    ///
    /// `holes` should contain `n - 1` distinct hands for a full assignment;
    /// supplying one hand reproduces the heads-up behaviour.
    fn from((partial, holes): (&Partial, Vec<Hole>)) -> Self {
        // Start from the post-blind root (blinds are posted correctly for any N
        // via the preflop ticker), seed every seat with hero's hole, then overwrite
        // each non-hero seat with its distinct sampled hand.
        let root = Game::root()
            .wipe(Hole::from(partial.seen()))
            .assume(partial.turn(), &holes);
        Self {
            root,
            actions: partial.actions().to_vec(),
        }
    }
}

impl Recall for Perfect {
    fn root(&self) -> Game {
        self.root
    }
    fn actions(&self) -> &[Action] {
        &self.actions
    }
}

#[allow(dead_code)]
impl Perfect {
    /// Erases opponent information, returning hero's perspective.
    ///
    /// Extracts hero's hole cards and board from root, discarding
    /// opponent's cards. Inverse of construction from `(&Partial, Hole)`.
    fn erase(&self, hero: Turn) -> Partial {
        let hole = self.root.seats()[hero.position()].cards();
        let board = Hand::from(self.root.board());
        let observation = Observation::from((Hand::from(hole), board));
        let actions = self
            .actions
            .iter()
            .filter(|a| a.is_choice())
            .cloned()
            .collect();
        Partial::from((hero, observation, actions))
    }
}
