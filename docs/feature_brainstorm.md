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

# Notes about the dataset:
- Battles aren't necessarily documented from start to finish, "only" the first 30 rounds are documented
- Additional effects of attacks aren't documented, you can just see the results based on the status (effects) of the enemy pokemon on the following turn