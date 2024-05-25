# ece509-final
ECE509 Convex Optimization Final

A study on the Alternating Direction Method of Multipliers (ADMM).

**Project Report**: [https://github.com/TotoroTron/ece509-final/blob/main/Convex_Project_ADMM_Swapnil_Brian.pdf]

**Presentation Slide** : [Distributed Optimization ADMM_Swapnil_Brian.pdf](https://github.com/TotoroTron/ece509-final/blob/main/Distributed%20Optimization%20ADMM_Swapnil_Brian.pdf)

**Main Reference**: Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers (Stephen Boyd, Neal Parikh, Eric Chu, Borja Peleato, Jonathan Eckstein).  
https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf

**Course Reference**: Convex Optimization (Stephen Boyd, Lieven Vandenburghe)   
https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf 

**File structure:**
```
├── admm
├── dist_lasso
│   ├── __pycache__
│   ├── logs_old
│   ├── square_2800x2800
│   │   ├── test_0
│   │   ├── test_1
│   │   └── test_2
│   ├── tall_4000x2000
│   └── wide_2000x4000
│       ├── test_0
│       ├── test_1
│       └── test_2
├── hello_mpi
└── lasso

15 directories
```

## Directories

### `admm/`

**Description:** Our first dive into the Alternating Direction Method of Multipliers (ADMM) algorithm. Contains aJupyter notebook where we apply ADDM to a simple quadratic problem to understand the fundamentals of ADMM.

### `lasso/`

**Description:** Our first attempt to apply ADMM to the Lasso regression in Jupyter notebooks.

### `hello_mpi/`

**Description:** Our first dive into the Message Passing Interface (MPI) adapted to Python, mpi4py. Contains simple problems using Scatter, Broadcast, and Gather to split up large tasks among multiple nodes in a computer cluster.

### `dist_lasso/`

**Description:** Our investigation into distributed ADMM using the global consensus strategy applied to the Lasso regression problem. Implemented with MPI. Test results for square, tall, and wide input data matrices. Dense matrices.

