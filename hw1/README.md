## Implementing a 10-armed bandit and plotting a 1000-step Average Reward chart

1. Implement the 10-armed bandit from Slide 8 in Chapter 2 of the course slides
* Each arm i has a true value qi∗​ generated from a Normal distribution N(0,1).
* Each time arm i is pulled, the output reward is a sample from a Normal distribution N(qi∗,1).

2. Implement the epsilon-greedy algorithm from Slide 13 using the sample-average method

3. Plot the 1000-step Average Reward chart from Slide 9
* Draw three curves for ϵ=0, 0.01, and 0.1.
* The average reward at each step should be averaged over at least 100 independent runs (i.e., take the average of 100 runs of 1000 steps each).

4. Plot a fourth curve
* Start with a larger initial ϵ, and gradually decrease it as steps increase. The method of decay can be designed freely.
* The final performance must outperform the curve of ϵ=0.1.