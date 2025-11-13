# Ideas for features:

- Rock-paper-scissors matchup between opposing pokemon types
    - Reconstruct enemy pokemon roster
- Identify strategies used by players to assess their skill?
    - Identify "bad" moves?
    - Some strategies are on the [smogon wiki (e.g. for alakazam)](https://www.smogon.com/dex/rb/pokemon/alakazam/)

- implement with Random forest
- count switching pokemon
- accounting for boosts
- Ratio between dealt and received damage
- Count Pokemon not only with no hp but with low/critical hp
- Not only count status effects but count rounds with negative status effects
- If the battle takes less than 30 turns, a clear winner is found
    - It turns out that there are no battles in the test/training dataset where players lose due to inactivity/forfeiting
- Build stronger relative/aggregated features in create_simple_features.py: differences (p1_hp_loss - p2_hp_loss), ratios (e.g. status round ratio), KOs per side, lead speed difference, remaining team size, initiative advantages (who moves first), number of forced switches, hazards such as spikes/toxic stacks – everything can be represented in a table and strengthens classic models.
- Consider interaction features: e.g. mean type effectiveness across active Pokémon, variance in opponent win rates, latest-turn momentum (HP difference turn t − turn t−1), or indicators for “critical conditions” (HP < 30%). You can generate such features directly in the timeline loop using rolling or aggregation logic.

# Notes about the dataset:
- Battles aren't necessarily documented from start to finish, "only" the first 30 rounds are documented
- Additional effects of attacks aren't documented, you can just see the results based on the status (effects) of the enemy pokemon on the following turn