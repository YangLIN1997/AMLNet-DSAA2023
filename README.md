# AMLNet: Adversarial Mutual Learning Neural Network for Non-AutoRegressive Multi-Horizon Time Series Forecasting

2023 IEEE 10th International Conference on Data Science and Advanced Analytics (DSAA)

Author: [Yang Lin](https://yanglin1997.github.io/)

E-mail: linyang1997@yahoo.com.au


## Abstract
<p align="justify">
In this paper, we introduce ProNet, an novel deep learning approach designed for multi-horizon time series forecasting, adaptively blending autoregressive (AR) and non-autoregressive (NAR) strategies. Our method involves dividing the forecasting horizon into segments, predicting the most crucial steps in each segment non-autoregressively, and the remaining steps autoregressively. The segmentation process relies on latent variables, which effectively capture the significance of individual time steps through variational inference. In comparison to AR models, ProNet showcases remarkable advantages, requiring fewer AR iterations, resulting in faster prediction speed, and mitigating error accumulation. On the other hand, when compared to NAR models, ProNet takes into account the interdependency of predictions in the output space, leading to improved forecasting accuracy. Our comprehensive evaluation, encompassing four large datasets, and an ablation study, demonstrate the effectiveness of ProNet, highlighting its superior performance in terms of accuracy and prediction speed, outperforming state-of-the-art AR and NAR forecasting models.

This repository provides an implementation for ProNet as described in the paper:

> AMLNet: Adversarial Mutual Learning Neural Network for Non-AutoRegressive Multi-Horizon Time Series Forecasting
> Yang Lin.
> International Conference on Data Science and Advanced Analytics (DSAA)
> [[Paper]](https://arxiv.org/pdf/2310.19289)

**Citing**

If you find AMLNet and the new datasets useful in your research, please consider adding the following citation:

```bibtex
@INPROCEEDINGS{10302580,
  author={Lin, Yang},
  booktitle={2023 IEEE 10th International Conference on Data Science and Advanced Analytics (DSAA)}, 
  title={AMLNet: Adversarial Mutual Learning Neural Network for Non-AutoRegressive Multi-Horizon Time Series Forecasting}, 
  year={2023},
  volume={},
  number={},
  pages={1-10},
  keywords={Training;Computational modeling;Time series analysis;Neural networks;Predictive models;Data science;Decoding;time series forecasting;deep learning;Transformer;knowledge distillation},
  doi={10.1109/DSAA60987.2023.10302580}}
```

## List of Implementations:

Sanyo: http://dkasolarcentre.com.au/source/alice-springs/dka-m4-b-phase

Hanergy: http://dkasolarcentre.com.au/source/alice-springs/dka-m16-b-phase

Solar: https://www.nrel.gov/grid/solar-power-data.html
