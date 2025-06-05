# Comprehensive Attribute Encoding and Dynamic LSTM HyperModels for Predictive Business Process Monitoring

**Authors**: Fang Wang (Florence Wong), Paolo Ceravolo, Ernesto Damiani  
**Repository**: Code and Demos for the associated research article.

---

## 📖 Overview  
**[Download Full Paper](https://arxiv.org/abs/2506.03696)**  
This repository provides implementations of **attribute encoding techniques** and **LSTM-based HyperModels** for outcome-oriented predictive business process monitoring. The models support multiple scenario setups, including:  

- Handling **both balanced and imbalanced datasets**.  
- Utilizing **pseudo-embedding matrices** for duration and feature correlations.  
- Supporting **hierarchical inputs** for sequence and event attributes.  
- Accommodating **simultaneous event inputs**.  

The framework is designed for flexibility and performance across diverse predictive process monitoring tasks.
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
}```
