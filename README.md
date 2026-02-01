# Neural Knowledge Distillation: BERT to Bi-LSTM

### A production-grade implementation of compressing a 110M parameter Transformer into a lightweight 2.5M parameter Recurrent Network for real-time edge inference.

![Project Status](https://img.shields.io/badge/Status-Complete-success)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Overview

Modern NLP is dominated by Transformers like BERT, which achieve state-of-the-art accuracy but are often too heavy (400MB+) and slow (100ms+ latency) for mobile devices or high-throughput APIs.

This project implements **Knowledge Distillation (KD)** to solve this deployment bottleneck. By treating a fine-tuned BERT model as a "Teacher" and a Bi-Directional LSTM as a "Student," I successfully transferred the teacher's nuanced understanding into a model that is **~20x smaller** and **~15x faster** while retaining high accuracy on the SST-2 sentiment analysis task.

---

## 📊 Benchmark Results

The core achievement of this project is the drastic reduction in computational cost with minimal loss in performance.

| Metric | Teacher (BERT-Base) |  Student (Bi-LSTM) | 🚀 Improvement |
| :--- | :--- | :--- | :--- |
| **Parameters** | 109,482,242 | 2,500,000 (Approx) | **~97% Smaller** |
| **Model Size** | ~420 MB | ~25 MB | **17x Compression** |
| **Inference Latency (CPU)** | ~166 ms | ~12 ms | **14x Speedup** |
| **Throughput** | ~6 req/sec | ~83 req/sec | **High Scalability** |

> *Benchmarks run on a standard Intel CPU environment to simulate edge/serverless deployment constraints.*

---

## Application Demo

I built an interactive dashboard using **Streamlit** to visualize the distillation process in real-time. It compares the prediction confidence and latency of both models side-by-side.

<p align = "center">
  <img src="./streamlit_screenshots/main_page.png" alt="Streamlit Dashboard" width="600">
  <img src="./streamlit_screenshots/results_page.png" alt="Streamlit Dashboard" width="600">
</p>


---

## Technical Architecture

### The Teacher: BERT (Bidirectional Encoder Representations from Transformers)
- **Role:** The Expert. Fine-tuned on the GLUE/SST-2 dataset.
- **Why:** Provides "Dark Knowledge" (soft logits) that contain information about class relationships, not just the final label.

### The Student: Bi-LSTM (Bidirectional Long Short-Term Memory)
- **Role:** The Apprentice. A custom PyTorch implementation.
- **Why:** LSTMs are sequentially much faster (`O(n)`) and require significantly less RAM than the `O(n^2)` attention mechanism of Transformers.
- The custom lightweight architecture optimized for CPU inference is used. Unlike the Teacher's 12-layer Transformer stack, this Student uses a shallow, recurrent approach.

    | Hyperparameter | Value | Description |
    | :--- | :--- | :--- |
    | **Vocabulary Size** | 30,522 | Matches BERT's `WordPiece` tokenizer for compatibility. |
    | **Embedding Dim** | 128 | Dense vector representation for input tokens. |
    | **Hidden Dimension** | 256 | The size of the LSTM's internal memory state. |
    | **Layers** | 2 | Stacked LSTM layers for capturing deeper semantic patterns. |
    | **Bidirectional** | True | Processes text Left-to-Right and Right-to-Left simultaneously. |
    | **Dropout** | 0.3 | Regularization applied to prevent overfitting during distillation. |
    | **Classifier Head** | Linear | Maps the concatenated hidden states ($256 \times 2 = 512$) to 2 classes. |

### The Distillation Process
Instead of training the Student on just hard labels (0 or 1), I trained it using a **Dual-Loss Function**:
1.  **Cross-Entropy Loss:** To match the ground truth labels.
2.  **KL-Divergence Loss:** To mimic the Teacher's probability distribution (Logits).

$$Loss = \alpha \cdot L_{soft}(Teacher, Student) + (1-\alpha) \cdot L_{hard}(Truth, Student)$$

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/RamuNalla/distillation-BERT-to-LSTM.git
cd distillation-BERT-to-LSTM
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Folder Structure
Ensure your local environment matches this structure:
```bash
├── models/
│   ├── teacher/          # Contains BERT config, model.safetensors, vocab.txt
│   └── student/          # Contains student_lstm.pth
├── notebooks/            # Jupyter notebooks used for training (Colab export)
├── model_archs.py        # The Bi-LSTM PyTorch class definition
├── benchmark.py          # Script to generate performance metrics
├── app.py                # Streamlit Dashboard source code
└── README.md
```

---

## 🚀 How to Run

### Run the Benchmark
Generate the expert metrics report (Latency, Parameter count, etc.) in your terminal.
```bash
python benchmark.py
```

### Launch the Dashboard
Start the web application to interact with the models.
```bash
streamlit run app.py
```

---

## 👨‍💻 Author
**Ramu Nalla** - Lead Data Scientist.