# Explaining Word Interactions using Integrated Directional Gradients
This repository contains the code developed for my Master's Thesis "Explaining Word Interactions using Integrated Directional Gradients", part of the Master in the Fundamental Principles of Data Science of the University of Barcelona.


- **Author**: Marc Ballestero Ribó.
- **Program**: Master's in the Fundamental Principles of Data Science.
- **Institution**: University of Barcelona
- **Advisors**: Dr. Daniel Ortiz-Martínez, Prof. Dr. Petia Radeva.
- **Thesis Period**: September 2024 - June 2025.
- **Qualification**: TBD.


## 🧾 Abstract
Explainability methods are key for understanding the decision-making processes behind complex text models. In this thesis, we theoretically and empirically explore Integrated Directional Gradients (IDG), a method that can attribute importance to both individual features and their high-order interactions. We introduce evaluation metrics to qualitatively assess the quality of the generated explanations, and propose a framework to adapt word-level evaluation methods to high-order phrase-level interactions. Applying IDG to a BERT-based hate speech detection model, we compare its performance at the word level against well-established methods such as Integrated Gradients (IG) and Shapley Additive Explanations (SHAP). Our results indicate that, while IDG's word-level attributions are less faithful than those of IG and SHAP, they are the best-scoring ones in terms of plausibility. On the other hand, IDG's high-order importance attributions exhibit high faithfulness metrics, indicating that IDG can consider hierarchical dependencies that traditional methods overlook. Qualitative analyses further support the interpretability of IDG explanations. Overall, this thesis highlights the potential of high-order explanation methods for improving transparency in text models.


## 📁 Repository Structure
```
.
├── CITATION.cff         # Citation metadata
├── data/                # Raw and preprocessed datasets
├── docs/                # Thesis report and defence slides
├── etc/                 # Extra data, report figures, etc.
├── LICENSE              # License file
├── models/              # Saved model checkpoints
├── notebooks/           # Jupyter notebooks for experiments
├── output/              # Generated results
├── README.md            # You're here
├── ruff.toml            # Linting configuration (ruff)
└── src/                 # Source code
```


## 🔍 Citation
Please cite this thesis using the following references:

### **Written Report**
```bibtex
TBD
```

### **Code**
Use the `CITATION.cff` file or the following BibTeX reference:
```bibtex
@misc{BallesteroRibo2025IDG_HateXplain,
  author       = {Marc Ballestero Ribó},
  title        = {IDG\_HateXplain},
  year         = {2025},
  howpublished = {\url{https://github.com/srmarcballestero/IDG_HateXplain}},
  institution  = {University of Barcelona},
  url          = {https://srmarcballestero.github.io}
}
```