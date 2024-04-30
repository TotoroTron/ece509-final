# ece509-final
ECE509 Convex Optimization Final

A study on the Alternating Direction Method of Multipliers (ADMM).

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

**Description:** Our first dive into the Alternating Direction Method of Multipliers (ADMM) algorithm. This directory contains Jupyter notebooks used to perform experiments on simple problems to understand the fundamentals of ADMM.

### `hello_mpi/`

**Description:** Our introductory exploration into using the Message Passing Interface (MPI). This directory includes examples and experiments demonstrating how to use MPI for distributing large computational tasks across multiple CPU nodes in a computer cluster.

### `dist_lasso/`

**Description:** Our investigation into distributed global consensus using ADMM. This directory focuses on the distributed implementation of the Lasso problem, employing ADMM techniques to achieve consensus in distributed system.
