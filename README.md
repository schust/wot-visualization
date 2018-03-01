# wot-visualization

This repository is a project about data analysis for World of Tanks.

As of this moment I have gathered data from over 1 million players which played during patch 9.22 on the RU cluster.

## Plots
### Player/Tank winrate plots
[Here (average)](wot-visualization/plots/pt_average) and [here (scatter)](wot-visualization/plots/pt_scatter), you can see player/tank winrate plots. I believe that this kind of visualization is the best way to showcase the concept of *OPness* in WoT. The idea is simple: Plot the winrate of players vs the winrate they have in single tanks. In the first case we show the average tank winrates.
In the first case, we show only the _average_ tank winrate for each player winrate. In the second case we show all data points, each point representing a player who played the tank for more than 100 matches.

In _overpowered_ tanks, players will have a higher winrate than usual, in _bad_ tanks they will have a lower winrate.
Furthermore, these plots show that _OP_ isn't a black-or-white concept. Tanks can overperform for bad players while underperform for good players and vice versa.

It is important to note however that this data is not 'clean'. The data used for these graphs is historic data, meaning that the whole player's history is taken into account. Tanks change every patch, and people tend to play more games using tanks that are overpowered during patches. Typical cases are the Hellcat or the T110E5, which were played quite heavily when they were better than nowadays.
Additionally, there are a lot of tanks which have very little games played in general. In such cases, the graphs will be unsteady and drawing conclusions is difficult. Super rare tanks, which were played less than 100 times by players in this data set were not included.

### Winrate Histograms
[Here](wot-visualization/plots/total_played_histo) we show the amount of battles played in a tank. In the blue histogram, we show the _overall winrates_ vs matches played in the relevant tank. In the orange histogram, we show _tank winrate_ vs matches played in the tank. By overlaying both histograms, we can see how in-tank performances compare to player performances. If the orange plot shows higher battle counts than the blue histogram for a specific winrate, this tank is overperforming for the specified player group, and vice versa.
Additionally, these plots can showcase very well how popular tanks are in different player groups. You can see if tanks are mainly played by players with lower overall winrates, which is often the case for low-tier tanks.