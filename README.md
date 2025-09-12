# Comprehensive Attribute Encoding and Dynamic LSTM HyperModels for Predictive Process Monitoring

***Predictive Process Monitoring (PPM) research library** featuring LSTM-based HyperModels with advanced attribute embeddings. Designed for real-world predictive process analytics, this toolkit balances flexibility, performance, and reproducibility.*

**Authors**: Fang Wang (Florence Wong), Paolo Ceravolo, Ernesto Damiani  
**Repository**: Code and Demos for the associated research article.

---

## 📖 Overview  
**[Download Full Paper](https://arxiv.org/abs/2506.03696)**  
This repository provides implementations of **attribute encoding techniques** and **LSTM-based HyperModels** for outcome-oriented predictive process monitoring. The models support multiple scenario setups, including:  

- Handling **both balanced and imbalanced datasets**.  
- Utilizing **pseudo-embedding matrices** for duration and feature correlations.  
- Supporting **hierarchical inputs** for sequence and event attributes.  
- Accommodating **simultaneous event inputs**.  

The framework is designed for flexibility and performance across diverse predictive process monitoring tasks.
---
## ⚙️ Installation & Requirements

This code has been developed and tested with the following versions:
- **Python:** 3.11.9
- **TensorFlow:** 2.16.1
- **Keras:** 3.11.3
- **Keras-turner:** 1.4.7
---
## 🧩 Repository Structure  

### 🔧 **Embedding and Encoding**  
| File | Description |
|------|-------------|
| `DurationEmbedding.py` | Duration pseudo-embedding matrix and encoding. |
| `FeatureEmbedding.py` | Co-relation pseudo-embedding matrix and encoding. |
| `DataEncoder.py` | Event-level and sequence-level attribute encoding, including multidimensional encoding. |

### 🤖 **HyperModels (LSTM Variants)**  
| Model | File | Description |
|-------|------|-------------|
| **B-LSTM** | `BaseLSTM.py` | Baseline LSTM for balanced datasets. |
| **B-LSTM (Imbalanced)** | `BaseLSTMIm.py` | Baseline LSTM for imbalanced datasets. |
| **D-LSTM** | `DurationEmbeddingLSTM.py` | LSTM with duration embeddings. |
| **D-LSTM (Imbalanced)** | `DurationEmbeddingLSTMIm.py` | Duration-embedded LSTM for imbalanced data. |
| **DC-LSTM** | `FeatureDurationEmbeddingLSTM.py` | LSTM with feature + duration embeddings. |
| **DC-LSTM (Imbalanced)** | `FeatureDurationEmbeddingLSTMIm.py` | DC-LSTM for imbalanced data. |
| **T-LSTM** | `TextFeatureDurationEmbeddingLSTM.py` | LSTM with text + feature + duration embeddings. |
| **T-LSTM (Imbalanced)** | `TextFeatureDurationEmbeddingLSTMIm.py` | T-LSTM for imbalanced data. |

### 🎛️ **Demos (Jupyter Notebooks)**  
| Notebook | Purpose |
|----------|---------|
| `FeatureEmbedding_call.ipynb` | Demo for feature embedding. |
| `DurationBin_call.ipynb` | Demo for duration bin encoding. |
| `PatientsBaseLSTM_call.ipynb` | Runs **B-LSTM** (adjustable for balanced/imbalanced datasets). |
| `PatientsTextEmbeddingLSTM_call.ipynb` | Runs **T-LSTM** (adjustable for balanced/imbalanced datasets). |
| `PatientsEmbeddingLSTM_call.ipynb` | Runs **D-LSTM** and **DC-LSTM** (adjustable for balanced/imbalanced datasets). |
| `BPI12ConcurrLSTM_call.ipynb` | Runs **M-B-LSTM** (balanced dataset demo, adjustable for imbalanced). |
| `BPI12EmbeddingLSTM_call.ipynb` | Runs **F-D-LSTM** (balanced dataset demo, adjustable for imbalanced). |

### 🔗 Explore More: HyperGNN Toolkit Now Available!

The **HyperGNN for Predictive Business Process Monitoring (PBPM)** is now live — extending this LSTM-based framework with a powerful GCN-based alternative.

Key features of the GCN toolkit include:
- Self-tuning HyperModels built on **GCNConv** and **GraphConv** layers  
- **Hierarchical input support** for sequence- and event-level attributes  
- **Duration-aware pseudo-embeddings** and **activity embeddings**  
- Robust performance across both **balanced and imbalanced** datasets  

📦 Check out the new repo here: **[HGCN (O): A Self-Tuning GCN HyperModel Toolkit for Outcome Prediction](https://github.com/skyocean/HGCN)**  
📄 [Read the preprint](https://arxiv.org/abs/2507.22524)

 
---

## 📜 Citation  
If you use this code, please cite the original paper:  

```bibtex
@article{Wang20205LSTMHyperPBPM,
         title={Comprehensive Attribute Encoding and Dynamic LSTM HyperModels for Outcome Oriented Predictive Business Process Monitoring}, 
         author={Fang Wang and Paolo Ceravolo and Ernesto Damiani},
         year={2025},
         eprint={2506.03696},
         archivePrefix={arXiv},
         primaryClass={cs.LG},
         url={https://arxiv.org/abs/2506.03696}, 
}
```

## 🔗 **About the Author**
This repository is maintained by Florence Wong, Ph.D. in Business Analytics and Applied Machine Learning.
For collaboration, contact via http://www.linkedin.com/in/florence-wong-fw
