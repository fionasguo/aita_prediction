# Estimating Causal Effects of Text Interventions Leveraging LLMs

This is a novel framework for texual intervention effect estimation using LLMs for counterfactual generation **even with missing observations** with domain adaptation.

Our paper: https://arxiv.org/abs/2410.21474

We are targeting interesting causal questions e.g.:
  * Would toning down the aggressiveness in a post increase or decrease people’s engagement (comment, repost)?
  
  * Would “peer influence” from a community’s top comment on a post change people’s impression of the original story? (r/AITA)
  
  * How would stylistic choices in a news headline affect the public’s perception of the same underlying event?

To deal with missing obesrvations, we either utilize the power of LLMs to synthesize counterfactual data, or to sample from suitable observed data to construct the counterfactual distribution.

Next we build NN predictors (e.g. BERT based) for potential outcomes, and we adapt the domain adaptation neural network (DANN) framework to mitigate the effect estimation biases.

![image](https://github.com/user-attachments/assets/9541f8fd-eb54-4105-b2ac-43575267ad97)

Citation:
```
@article{guo2024estimating,
  title={Estimating Causal Effects of Text Interventions Leveraging LLMs},
  author={Guo, Siyi and Marmarelis, Myrl G and Morstatter, Fred and Lerman, Kristina},
  journal={arXiv preprint arXiv:2410.21474},
  year={2024}
}
```
