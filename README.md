# 📄 Code Plagiarism Detection System

A semantic code similarity detection tool built using PyTorch and Graph Neural Networks (GNNs). Designed to detect plagiarism in academic Java code submissions using graph-based representations and trained with triplet loss for precise similarity learning.

---

## 🚀 Features

- 🔍 Detects semantic plagiarism in Java code, even with structural obfuscations.
- 🤖 Leverages Graph Neural Networks to learn syntax-aware code embeddings.
- 📐 Trained using triplet loss for discriminative learning of similar/dissimilar pairs.
- 🛠️ Converts code into graph structures (AST or CFG) for rich semantic modeling.
- 📊 Outputs a plagiarism percentage score between code submissions.

---

## 🧱 Architecture

- **Language Input:** Java  
- **Backend:** PyTorch + PyTorch Geometric  
- **Learning Objective:** Triplet Loss  
- **Input Format:** Graph-based (e.g., Abstract Syntax Tree)  
- **Output:** Plagiarism score as a percentage (0% - 100%)

---

## 📂 Project Structure

```
code-plagiarism-detector/
│
├── data/                   # Raw and preprocessed Java files
├── graphs/                 # Code graph representations (AST/CFG)
├── models/                 # GNN architectures
├── utils/                  # Feature extraction and graph tools
├── training/               # Triplet generation and training scripts
├── inference/              # Similarity scoring and evaluation
├── feature_extraction.py   # Graph input preparation
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 📈 Output

The model returns a **plagiarism percentage score** (0% to 100%) indicating the semantic similarity between two code files.  
- **0%** → Completely different  
- **100%** → Highly similar or plagiarized

---

## 📚 Citation

If you use this work in your research, please cite:

```
Yash Shukla, "Advanced Hybrid Graph Neural Networks for Robust Code Plagiarism Detection: A Comparative Analysis of Data-Driven Techniques", ICCUBEA 2025 (Accepted).
```

---

## 🤝 Contributions

Pull requests are welcome!  
If you’d like to contribute or propose major changes, please open an issue first to initiate discussion.

---
