# Rotator bandit

An implementation of the Thompson (beta)sampling agent running a multiarmed
bandit problem with prepared data. The algorithm is based on this [blog
post](https://conrmcdonald.medium.com/solving-multiarmed-bandits-a-comparison-of-epsilon-greedy-and-thompson-sampling-d97167ca9a50)
by Connor Mc. Additionally supports estimation of the clickability matrix and
searching for optimal hyperparameters.

# Building

To build the program you will need to install Rust. You can find a guide
[here](https://www.rust-lang.org/tools/install).

# Configuration

To configure the program, change the constants located in `the main.rs` file. For
explanation see the comments there.

# Running

This will build the program and run multiple experiments with the dataset
`data/` and save the results to `output/`.

```sh
$ cargo run --release -- experiment data/ output/
```


This will build the program and perform a grid search for the optimal alpha and
beta params. The simulation uses data from `data/` and saves the resulting plot
to `output/`.

```sh
$ cargo run --release -- optimize_dist data/ output/
```


This will build the program and perform a grid search for the optimal test
probability. The simulation uses data from `data/` and saves the resulting plot
to `output/`.

```sh
$ cargo run --release -- optimize_testp data/ output/
```
