# CG Cache

## Base Test Framework

TIGER (2023, WWW): A TGNN model with adaptive neighborhood sampling.

Some implementation details:
- Class `NumericalFeature` moves all edge and node features to GPU before model computation.
- Class `ComputationGraph` contains only CG structures (nids, eids, tss).
- The layers in TIGER are arranged by 0-th, L-th, L-1-th, ..., 1-st.

## CG Cache

For each node, we cache its all layers of neighbor nids, tss, and eids.

The cached data is modified incrementally by discarding outdated neighbors' cgs and adding new neighbors' cgs.

## Implementation

- Use GPU tensor apis if possible (e.g. scatter, gather). 

# Logs

```
nids[:3] [1125  391 1088]
ts[:3] [179775. 179777. 179851.]
len layers: 3
per layer: 3, type: <class 'list'>
0 -> (600,), <class 'numpy.ndarray'>
1 -> (6000, 10), <class 'numpy.ndarray'>
nids [[0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]]
eids [[0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]]
tss [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
2 -> (600, 10), <class 'numpy.ndarray'>
nids [[  0   0   0   0   0   0   0   0   0   0]
 [258 258 258 258 258 258 258 258 258 258]
 [437 437 437 437 437 437 437 437 437 437]]
eids [[   0    0    0    0    0    0    0    0    0    0]
 [6346 6405 6419 6448 6747 7269 7318 7331 7368 7547]
 [7643 7663 7667 7679 7697 7699 7701 7703 7710 7721]]
tss [[     0.      0.      0.      0.      0.      0.      0.      0.      0.
       0.]
 [154038. 154722. 155095. 155638. 161492. 170749. 171517. 171626. 172236.
  175511.]
 [176876. 177357. 177408. 177715. 178099. 178269. 178320. 178344. 178529.
  178892.]]
```