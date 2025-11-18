# 🌌 **Mapping the Cosmic Web using Graph Neural Networks (GNNs)**

*A lightweight, scalable pipeline to analyze galaxy distributions from IllustrisTNG and classify large-scale cosmic structures.*

## 🚀 Overview

This project explores **how Graph Neural Networks (GNNs)** can be used to interpret the **Cosmic Web** — the vast network of filaments, knots, walls, and voids that structure our Universe.
Using a real slice of the **IllustrisTNG100-1 simulation**, we:

* Extract galaxy (subhalo) properties from **HDF5 group catalogs**
* Construct a **k-Nearest Neighbour graph**
* Train a lightweight **GraphSAGE GNN** on a small subset
* Evaluate performance using classification metrics and visual projections
* Compare model predictions against the underlying dataset

The goal: demonstrate that **GNNs can capture relational, non-Euclidean cosmic structure** better than classical ML.

---

## 📦 Features

✔ Extract real galaxy data from IllustrisTNG (positions, mass, velocity, SFR)
✔ Build kNN graphs from spatial coordinates
✔ Train a scalable GraphSAGE model
✔ Visualize galaxy projections (XY, XZ)
✔ Visualize graph connections (kNN)
✔ Evaluate GNN using confusion matrix & accuracy
✔ Includes dataset sampler + utility notebook

---

## 🧪 Dataset

**Simulation:** IllustrisTNG100-1
**Snapshot:** 99 (z = 0)
**Files used:**

* `fof_subhalo_tab_099.*.hdf5` (subhalo catalog)
* Extracted CSV: `tng100_snapshot99_sample_2000.csv`
* Derived graph: `graph_data.pt`, `graph_data_npy.npz`

From these files, we used:

* Position (`SubhaloPos`)
* Velocity (`SubhaloVel`)
* Mass (`SubhaloMass`)
* Star Formation Rate (`SubhaloSFR`)
* Halo membership (`SubhaloGrNr`)

---

## 🧠 Methodology

1. **Data Extraction**: Read subhalo properties from `.hdf5` group catalogs
2. **Filtering**: Keep first 2000 subhalos for a lightweight pipeline
3. **Graph Construction**: Build kNN graph (k = 8)
4. **Model**: GraphSAGE (`SAGEConv`) with 3 layers
5. **Training**: Node classification task (binary mock labels)
6. **Evaluation**:

   * Accuracy
   * F1 score
   * Confusion matrix
7. **Visualization**:

   * Galaxy XY/XZ projections
   * Mass quartile color maps
   * kNN graph
   * GNN predictions

---

## 📊 Key Results

* **GNN Test Accuracy:** ~0.83
* **Macro F1 Score:** ~0.82
* Captures spatial clustering & structure well
* Predictions visually align with dataset distribution

See `/images/` folder for all plots:

* `plot_xy.png`
* `plot_xz.png`
* `knn_graph.png`
* `gnn_predictions_xy.png`
* `confusion_matrix.png`
* `plot_xy_mass_quartile.png`
* etc.

---

## 🔧 Tech Stack

* **Python 3**
* **PyTorch / PyTorch Geometric (PyG)**
* **NumPy, Pandas, Matplotlib**
* **h5py**
* **scikit-learn**

---

## 📘 How to Run

1. Clone the repo:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

2. Open the notebook:

```bash
jupyter notebook notebook/cosmic_web_gnn.ipynb
```

3. Run all cells.

---

## 🛰 Future Work

* Replace binary labels with **actual LSS classification** (voids, filaments, walls)
* Train on larger TNG100 and TNG300 samples
* Add dynamic GNNs using velocities
* Compare GraphSAGE vs GAT vs TNNs

---

## 🧑‍💻 Authors

**Jagdish Wagh**, **Gangotrinath Tripathi**, **(formerly Arpita Singh)**
Artificial Intelligence & Data Science,
Thakur College of Engineering and Technology, Mumbai

---

## ⭐ Acknowledgements

We thank the **IllustrisTNG Collaboration** for making the simulation data openly accessible.

---
