# Learning Causal Effects of Textual Interventions

This is a novel framework for texual intervention effect estimation **even with missing observations**.

We are targeting interesting causal questions e.g.:
  * Would toning down the aggressiveness in a post increase or decrease people’s engagement (comment, repost)?
  
  * Would “peer influence” from a community’s top comment on a post change people’s impression of the original story? (r/AITA)
  
  * How would stylistic choices in a news headline affect the public’s perception of the same underlying event?

To deal with missing obesrvations, we either utilize the power of LLMs to synthesize counterfactual data, or to sample from suitable observed data to construct the counterfactual distribution.

Next we build NN predictors (e.g. BERT based) for potential outcomes, and we adapt the doubly robust framework to mitigate the effect estimation biases.
