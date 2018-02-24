# wot-visualization

This repository is a project about data analysis for World of Tanks.

As of this moment I have gathered data from over 900k players which played during patch 9.22 on the RU cluster.

## Plots
### Player/Tank winrate plots
In the `plots` folder, you can see player/tank winrate plots. I believe that this kind of visualization is the best way to showcase the concept of _OPness_ in WoT. The idea is simple: Plot the winrate of players vs the winrate they have in single tanks. In _overpowered_ tanks, they will have a higher winrate than usual, in _bad_ tanks they will have a lower winrate.
Furthermore, these plots show that _OP_ isn't a black-or-white concept. Tanks can overperform for bad players while underperform for good players and vice versa.

What is important to note however is that this data is not 'clean'. The data used for these graphs is historic data, meaning that the whole player's history is taken into account. Tanks change every patch, and people tend to play more games using tanks that are overpowered during patches. Typical cases are the Hellcat or the T110E5, which were played quite heavily when they were better than nowadays.
Additionally, there are a lot of tanks which have very little games played in general. In such cases, the graphs will be unsteady and drawing conclusions is difficult. I didn't include some super rare tanks, which were played less than 100 times by players in this data set.