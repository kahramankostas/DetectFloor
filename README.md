# DetectFloor

## Project Description
This repository contains the implementation and evaluation framework for the research paper **"Graph-Based Floor Separation Using Node Embeddings and Clustering of Wi-Fi Trajectories"**.

The project addresses the challenge of vertical localization in multi-story buildings by proposing a novel approach that constructs a graph where nodes represent Wi-Fi fingerprints and edges represent signal similarity and transitions. It utilizes **Node2Vec** to generate low-dimensional embeddings of these fingerprints and applies clustering algorithms (such as K-Means and various community detection methods) to identify distinct floors without requiring extensive labeled data.

## features
*   **Graph-Based Approach:** Constructs a trajectory graph from Wi-Fi fingerprints.
*   **Node Embeddings:** Uses `node2vec` to create vector representations of Wi-Fi fingerprints, capturing structural relationships.
*   **Comprehensive Evaluation Metrics:**
    *   Accuracy, Precision, Recall, F1-Score.
    *   Clustering Quality: Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), Homogeneity, Completeness.
*   **Statistical Validation:**
    *   **Bootstrap Confidence Intervals:** Robust performance estimation using bootstrapping.
    *   **Hypothesis Testing:** Statistical comparison against baselines using McNemar's Test, Paired t-test, and Wilcoxon Signed-Rank Test.
*   **Cluster-to-Floor Mapping:** Automated mapping of unsupervised clusters to ground truth floors.
*   **Visualization:** Confusion matrices with confidence intervals, cluster purity analysis, and distribution plots.

## Reference Paper
If you use this code or dataset in your research, please cite the following paper:

> **Graph-Based Floor Separation Using Node Embeddings and Clustering of Wi-Fi Trajectories**  
> *Rabia Yasa Kostas, Kahraman Kostas*  
> arXiv preprint, 2025.

**Abstract:**  
The paper proposes a graph-based method for floor separation using Wi-Fi trajectories. By modeling the environment as a graph and utilizing node embeddings, the method achieves robust floor-level localization. The approach was evaluated on the **Huawei University Challenge 2021** dataset, demonstrating superior performance over traditional methods.

## Project Structure
*   **`Huawei/`**: Experiments and scripts related to the Huawei University Challenge 2021 dataset.
    *   **`HW-Def/`**: Baseline methods and statistical analysis scripts (`new.py`).
    *   **`HW-WBDE/`**: Implementation of the proposed Graph-Based method with Node2Vec (`node.py`).
*   **`UJIndoorLoc/`**: Adaptation of the methods for the UJIndoorLoc dataset.
*   **`datasets/`**: Raw and preprocessed datasets used in the study.

## Requirements
To run the code, install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy node2vec networkx
```

## Usage

### 1. Generating Node Embeddings
The core validation of the paper's method involves generating embeddings from the constructed graph:

```bash
python Huawei/HW-WBDE/node.py
```
*   **Input**: `my.adjlist` (Graph adjacency list).
*   **Output**: `vectors.emb` (Node embeddings).

### 2. Evaluation and Statistical Verification
To evaluate the clustering results and perform the statistical tests mentioned in the paper:

```bash
python Huawei/HW-Def/new.py --path ./Huawei --baseline_file ./Huawei/baseline_communities.csv
```

**Arguments:**
*   `--path`: Path to result files.
*   `--baseline_file`: (Optional) Baseline results for statistical comparison (McNemar/t-test).
*   `--n_bootstrap`: 1000 (Default) for 95% Confidence Intervals.

## License
This project is licensed under the [MIT](LICENSE) license.
