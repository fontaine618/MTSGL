# MTSGL

Multi-task regression with sparse groupe regularization using a proximal ADMM. This Python module was created as part of
a class project for STATS 606 at the University of Michigan during the winter 2020 term by Simon Fontaine, Jinming Li
and Yang Li.

We consider the following optimization problem

$$
    \text{minimize}_\beta
    \sum_{k=1}^K L(Y^{(k)},X^{(k)}\beta^{(k)})
    + \lambda P_{q,\alpha}(\beta),
$$
where

$$
    P_{q,\alpha}(\beta)
    = \sum_{j=1}^p \alpha\Vert\beta\Vert_1
    + (1-\alpha)\Vert\beta\Vert_q,
$$
for $q \in \{2,\infty\}$, $\alpha\in[0,1]$ and $\lambda>0$.

## Installation

## Example

## References

## Authors
Simon Fontaine, simfont@umich.edu

Jinming Li, lijinmin@umich.edu

Yang Li, yangly@umich.edu

## License

MIT

## Acknowledgments
