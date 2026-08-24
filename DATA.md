# Data

## Jeans catalog

`data/amazon_jeans_shop.json` is a product-feed subset derived from the
publicly available **Amazon Reviews 2023** dataset:

> Yupeng Hou, Jiacheng Li, Zhankui He, An Yan, Xiusi Chen, and Julian McAuley.
> Bridging Language and Items for Retrieval and Recommendation.
> arXiv:2403.03952, 2024.

Original data: https://amazon-reviews-2023.github.io/

This repository redistributes only a compact product-metadata slice used in
the paper (jeans category). It does **not** include customer reviews,
conversation traces, or judge outputs.

## Frozen scenarios

`data/amazon_jeans_sample_20.json` contains the 20 jeans scenarios used as
the paper's worked example. `scripts/generate_scenarios.py` is a generic
template for building *new* scenarios; it is **not** how these 20 were
produced and does not ship a query pool.
