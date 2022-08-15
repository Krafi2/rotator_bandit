# Rotator bandit

An implementation of the Thompson (beta)sampling agent running a multiarmed
bandit problem with prepared data. The algorithm is based on this [blog
post](https://conrmcdonald.medium.com/solving-multiarmed-bandits-a-comparison-of-epsilon-greedy-and-thompson-sampling-d97167ca9a50) by Connor Mc.

# Running

This will run the agent for 5000 impressions against the data at `data/ctrs.txt`

```sh
$ cargo run -- data/ctrs.txt 5000
```
