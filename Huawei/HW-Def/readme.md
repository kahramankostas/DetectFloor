# Graph-Based Floor Separation Using Node Embeddings and Clustering of WiFi Trajectories

This repository contains the implementation of the paper *"Graph-Based Floor Separation Using Node Embeddings and Clustering of WiFi Trajectories"* by Rabia Yasa Kostas and Kahraman Kostas, published as part of the Huawei University Challenge 2021. The proposed method leverages WiFi fingerprint trajectories, graph construction, Node2Vec embeddings, and K-means clustering to achieve floor separation in multi-storey indoor environments.

## Overview

Indoor positioning systems (IPSs) are critical for location-based services in complex multi-storey buildings. This project introduces a novel graph-based approach for floor separation using WiFi fingerprint trajectories. The methodology constructs a graph where nodes represent WiFi fingerprints, and edges are weighted by signal similarity and contextual transitions. Node2Vec generates low-dimensional embeddings, which are clustered using K-means to identify distinct floors. The approach achieves an accuracy of 68.97%, an F1-score of 61.99%, and an Adjusted Rand Index (ARI) of 57.19% on the Huawei University Challenge 2021 dataset.

## Dataset

The experiments utilize the publicly available **Huawei University Challenge 2021 Dataset**, designed for floor prediction in large-scale indoor environments. The dataset includes:

- **fingerprints.json**: WiFi fingerprints with access point signal strengths and locations.
- **steps.csv**: Sequential step pairs within the same trajectory (same floor).
- **elevations.csv**: Fingerprint pairs indicating vertical transitions (e.g., elevators or stairs).
- **estimated_wifi_distances.csv**: Pre-estimated distances between fingerprints based on signal strength.
- **lookup.json**: Mapping between fingerprint IDs and trajectory IDs.
- **GT.json**: Ground-truth floor labels for a subset of trajectories.

The dataset is clean, with no outlier fingerprints, and is publicly available. See the paper for details on accessing the preprocessed dataset.

## Installation

To run the code, ensure you have Python 3.10 installed. Follow these steps to set up the environment:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kahramankostas/IPIN2025.git
   cd IPIN2025
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` includes:
   - `numpy`
   - `pandas`
   - `scikit-learn`
   - `node2vec`
   - `networkx`
   - `matplotlib`
   - `scipy`
   - `tqdm`
   - `igraph`
   - `leidenalg`



4. **Download the dataset**:
   Place the Huawei University Challenge 2021 dataset files in the `task2_for_participants/` directory. Ensure the directory structure matches the expected format (see `task2_for_participants/README.md` for details).

## Usage

The codebase is organized to replicate the experiments described in the paper. Key scripts include:

- **`theCode.ipynb`**: Converts input files into a graph structure (nodes: trajectories/APs, edges: proximity with elevation hints).
- Generates 32-dimensional Node2Vec embeddings for the graph.
-  Applies K-means clustering (k=17) on embeddings and baseline community detection algorithms (Louvain, Leiden, etc.).
- **`evaluate.py`**: Computes evaluation metrics (Accuracy, F1-score, ARI, NMI, Purity).

To run the `evaluate.py`:
```bash
python evaluate.py --gt_file=GT.json --result_file=node2vec.csv --output_dir=./community_results/node2vec 
```

This will:
1. Preprocess the dataset.
2. Generate Node2Vec embeddings.
3. Perform clustering with K-means and baseline methods.
4. Save results (CSV files and confusion matrices) in the `./results/` directory.



## Results

The proposed Node2Vec + K-means pipeline outperforms traditional community detection algorithms. Performance metrics on the Huawei University Challenge 2021 dataset are:

| Algorithm       | Accuracy | F1-Score | ARI   | NMI   | Purity  |
|-----------------|----------|----------|-------|-------|---------|
| Node2Vec        | 0.6897   | 0.6199   | 0.5719| 0.5747| 79.50%  |
| Fast Greedy     | 0.4148   | 0.3555   | 0.1052| 0.1382| 94.19%  |
| Infomap         | 0.4597   | 0.4013   | 0.0928| 0.1696| 82.45%  |
| Label Prop.     | 0.4533   | 0.3882   | 0.0764| 0.1579| 86.19%  |
| Leiden          | 0.4437   | 0.3651   | 0.0660| 0.1309| 92.37%  |
| Louvain         | 0.4437   | 0.3651   | 0.0600| 0.1307| 92.78%  |

Confusion matrices (see `community_results/`) show that Node2Vec + K-means excels in denser clusters, while Leiden and Louvain offer stable results with high purity but lower global alignment.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for bug reports, feature requests, or improvements. Ensure code follows PEP 8 style guidelines and includes relevant tests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or dataset in your research, please cite the original paper:

```bibtex
@article{kostas2025graph,
  title={Graph-Based Floor Separation Using Node Embeddings and Clustering of WiFi Trajectories},
  author={Kostas, Rabia Yasa and Kostas, Kahraman},
  journal={TBD},
  year={2025},
  publisher={TBD}
}
```
## Contact
For questions or inquiries, contact:
- Kahraman Kostas: kahramankostas@gmail.com
