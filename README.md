# Graph-Based Floor Separation Using Node Embeddings and Clustering of Wi-Fi Trajectories

[![arXiv](https://img.shields.io/badge/arXiv-2505.08088v3-b31b1b.svg)](https://arxiv.org/abs/2505.08088v3)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

This repository contains the implementation of a novel graph-based framework for blind floor separation in multistory indoor environments using only Wi-Fi fingerprint trajectories. The method eliminates the need for prior building information or knowledge of the number of floors, making it highly practical for real-world indoor positioning systems.

**Key Features:**
- ✅ Fully data-driven approach requiring no building metadata
- ✅ Automatic floor number estimation
- ✅ Robust to signal noise and environmental variability
- ✅ State-of-the-art performance on multiple benchmark datasets
- ✅ Scalable to different building architectures

## Abstract

Vertical localization, particularly floor separation, remains a major challenge in indoor positioning systems operating in GPS-denied multistory environments. This paper proposes a fully data-driven, graph-based framework for blind floor separation using only Wi-Fi fingerprint trajectories.

The framework represents Wi-Fi fingerprints as nodes in a trajectory graph, where edges capture both signal similarity and sequential movement context. Structural node embeddings are learned via Node2Vec, and floor-level partitions are obtained using K-Means clustering with automatic cluster number estimation.

**Results:** The proposed approach achieves **76.60% accuracy** on the Huawei University Challenge 2021 dataset and demonstrates that signal-based distance estimation can match the performance of geometric ground-truth distances on the UJIIndoorLoc benchmark.

## Methodology

### Pipeline Overview

```
Wi-Fi Trajectories → Graph Construction → Node2Vec Embedding → K-Means Clustering → Floor Labels
```

### Core Components

1. **Graph Construction**
   - Nodes: Wi-Fi fingerprints with RSSI measurements
   - Edges: Weighted by signal similarity and sequential movement
   - Includes horizontal (step) and vertical (elevation) transitions

2. **Distance Estimation (WBDE)**
   - Supervised ML model (XGBoost) trained on external datasets
   - Converts RSSI measurements to physical distance estimates
   - No data leakage from floor classification datasets

3. **Node Embedding (Node2Vec)**
   - Generates 32-dimensional vector representations
   - Balances local (BFS) and global (DFS) graph structure
   - Captures floor-level topology without labels

4. **Clustering**
   - K-Means with automatic K selection via Calinski-Harabasz Index
   - Evaluates K ∈ [3, 20] to match building constraints
   - Post-hoc majority voting for cluster-to-floor mapping

## Installation

### Prerequisites

```bash
Python >= 3.8
pip >= 21.0
```

### Setup

```bash
# Clone the repository
git clone https://github.com/kahramankostas/DetectFloor.git
cd DetectFloor

# Install dependencies
pip install -r requirements.txt
```

### Required Python Packages

```
networkx
node2vec
scikit-learn
xgboost
pandas
numpy
matplotlib
seaborn
```

## Datasets

### 1. Huawei University Challenge 2021 (Primary)

**Description:** Large-scale floor prediction dataset from a university campus with 3-20 floors per building.

**Files:**
- `fingerprints.json`: Wi-Fi fingerprints with AP signal strengths
- `steps.csv`: Sequential connections (horizontal movement)
- `elevations.csv`: Vertical transitions (stairs/elevators)
- `estimated_wifi_distances.csv`: Noisy pairwise distances (baseline)
- `GT.json`: Ground-truth floor labels (evaluation only)

**Download:** [Huawei University Challenge 2021](https://github.com/kahramankostas/DetectFloor/blob/main/datasets/floor-detection-datasets/Huawei-UK-University-Challenge-Competition-2021.rar)

### 2. UJIIndoorLoc (Validation)

**Description:** Benchmark dataset covering 3 buildings with 4-5 floors each (13 total floor-building combinations).

**Subsets:**
- Training (UJI-T): ~19,000 fingerprints
- Validation (UJI-V): ~1,100 fingerprints

**Download:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc)



### Data Preparation Workflow

The visual abstract shows the complete data preparation pipeline:

```
000.ipynb → Extract zip files
001.ipynb → Convert to JSON format
002.ipynb → Calculate coordinate-based distances
003.ipynb → Extract distance metrics from RSSI
004.ipynb → Merge datasets for training/testing
005.ipynb → Train XGBoost distance estimation model
```

### Experimental Scenarios

Six scenarios are implemented (see Table I in paper):

| Scenario | Dataset | Distance Source | Description |
|----------|---------|-----------------|-------------|
| **HW-Def** | Huawei | Default | Noisy competition distances (baseline) |
| **HW-WBDE** | Huawei | WBDE | ML-improved distances (proposed) |
| **UJI-Geo-T** | UJI Train | Geometric | Ground-truth coordinates (upper bound) |
| **UJI-WBDE-T** | UJI Train | WBDE | Signal-based estimation (proposed) |
| **UJI-Geo-V** | UJI Valid | Geometric | Ground-truth coordinates (upper bound) |
| **UJI-WBDE-V** | UJI Valid | WBDE | Signal-based estimation (proposed) |



## Results

### Performance Summary

| Method | Huawei (Acc %) | UJI-T (Acc %) | UJI-V (Acc %) |
|--------|----------------|---------------|---------------|
| Fast Greedy | 33.0 | 22.3 | 20.6 |
| Louvain | 32.9 | 17.5 | 20.3 |
| Leiden | 33.0 | 22.4 | 20.4 |
| Label Prop | 33.9 | 0.0 | 20.5 |
| Infomap | 34.4 | 11.8 | 18.0 |
| GCN | 39.9 | 25.3 | 47.4 |
| GAT | 50.2 | 35.5 | 51.3 |
| **Node2Vec** | **76.6** | **66.1** | **59.4** |

### Key Findings

1. **Node2Vec outperforms all baselines** by significant margins (26-43% improvement)
2. **Signal-based distances match geometric ground truth** on UJIIndoorLoc
3. **GNNs underperform** due to noisy node features and over-smoothing
4. **Traditional methods over-segment** (high purity, low completeness)

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{kostas2025graph,
  title={Graph-Based Floor Separation Using Node Embeddings and Clustering of Wi-Fi Trajectories},
  author={Kostas, Rabia Yasa and Kostas, Kahraman},
  journal={arXiv preprint arXiv:2505.08088},
  year={2025}
}
```

## Repository Structure

```
DetectFloor/
├── data/
│   ├── huawei/              # Huawei Challenge 2021 dataset
│   ├── uji/                 # UJIIndoorLoc dataset
│   └── external/            # WBDE training datasets
├── notebooks/
│   ├── 000.ipynb            # Data extraction
│   ├── 001.ipynb            # Format conversion
│   ├── 002.ipynb            # Distance calculation
│   ├── 003.ipynb            # Feature extraction
│   ├── 004.ipynb            # Dataset merging
│   └── 005.ipynb            # WBDE model training
├── src/
│   ├── graph_builder.py     # Graph construction
│   ├── distance_estimator.py  # WBDE model
│   ├── embeddings.py        # Node2Vec & GNN implementations
│   ├── clustering.py        # K-Means + CH Index
│   └── evaluation.py        # Metrics & bootstrapping
├── experiments/
│   └── run_experiments.py   # Main experimental script
├── results/                 # Output folder for results
├── requirements.txt
└── README.md
```

## Evaluation Metrics

The framework uses comprehensive evaluation:

- **Classification Metrics:** Accuracy, F1-Score (weighted)
- **Clustering Metrics:** ARI, NMI, Purity
- **Statistical Validation:** Bootstrap resampling (1,000 iterations) with 95% CI

## Key Advantages

1. **No Building Metadata Required** - Works without floor plans or AP locations
2. **Automatic Floor Detection** - Estimates number of floors from data
3. **Hardware Independent** - Uses only RSSI measurements
4. **Scalable** - Linear complexity with Node2Vec
5. **Generalizable** - Validated across multiple building types

## Limitations & Future Work

### Current Limitations
- Requires continuous trajectory data or careful synthetic edge construction
- Performance degrades with very sparse datasets (< 100 fingerprints)
- Assumes stable AP infrastructure during data collection

### Future Directions
- Real-time online clustering for live navigation
- Multi-device heterogeneity handling
- Integration with IMU sensors for hybrid approaches
- Transfer learning across buildings

## Authors

- **Rabia Yasa Kostas** - Gümüşhane University, Turkey
  - [Personal Website](https://rykostas.github.io/)
  - Research: Machine Learning, NLP, Indoor Positioning

- **Kahraman Kostas** - Turkish Ministry of National Education
  - [Personal Website](https://kahramankostas.github.io/)
  - Research: IoT Security, Network Forensics, ML

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Huawei University Challenge 2021 for the primary dataset
- UCI Machine Learning Repository for the UJIIndoorLoc benchmark
- Contributors to Node2Vec, NetworkX, and scikit-learn libraries

## Contact

For questions or collaboration opportunities:
- 📧 Email: kahramankostas@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/kahramankostas/DetectFloor/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/kahramankostas/DetectFloor/discussions)

---

**Note:** This repository is actively maintained. For the latest updates, please check the [GitHub repository](https://github.com/kahramankostas/DetectFloor).
