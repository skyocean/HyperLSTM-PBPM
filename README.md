# Comprehensive Attribute Encoding and Dynamic LSTM HyperModels for Predictive Business Process Monitoring

**Authors**: Fang Wang (Florence Wong), Paolo Ceravolo, Ernesto Damiani  
**Repository**: Code and Demos for the associated research article.

---

## 📖 Overview  
This repository contains implementations of **attribute encoding techniques** and **LSTM-based HyperModels** for outcome-oriented predictive business process monitoring. The models support both **balanced and imbalanced datasets**, with dynamic embeddings for duration, feature correlations, and text data.

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

---

## 🚀 Quick Start  
1. **Clone the repository**:  
   ```bash
   git clone https://github.com/your-repo-url.git
   cd your-repo


## 📜 Citation  
If you use this code, please cite the original paper:  

```bibtex
@article{LSTMHyperPBPM2024,
  title={Comprehensive Attribute Encoding and Dynamic LSTM HyperModels for Outcome-Oriented Predictive Business Process Monitoring},
  author={Wang, Fang and Ceravolo, Paolo and Damiani, Ernesto},
  journal={Journal of Business Process Analytics},
  volume={X},
  pages={XX--XX},
  year={2023},
  publisher={Springer}
}```
