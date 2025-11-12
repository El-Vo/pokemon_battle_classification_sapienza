# Feedback on Amirs tried features
- There are no forfeits/timeouts that end battles before 30 rounds have passed.
- successful_explosion is statistically insignificant (double-sided p-value is 37.6% -> zero-hypothesis cannot be challenged)
- successful_explosion_p2 is broken (Coefficient is 0)
- high_damage_moves does not account for switching pokemon
- p1_avg_team_winrate and p2_avg_team_winrate can exceed 1 because win counts are pulled from the full training set while appearances are only tallied on the currently processed data; align both sources (or clamp) to keep rates within [0,1].