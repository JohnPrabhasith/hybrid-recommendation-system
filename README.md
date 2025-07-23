#  Hybrid Recommendation System (TF-IDF + SVD)

This repository contains a Jupyter Notebook that implements a **hybrid recommendation system** combining:

-  **Content-Based Filtering** using **TF-IDF Vectorization**
-  **Collaborative Filtering** using **SVD (Singular Value Decomposition)** from the `scikit-surprise` library

By merging both approaches, the system generates smarter, more personalized recommendations that consider both **item metadata** and **user behavior**.

---

##  What It Does

###  Content-Based Filtering
- Uses `TfidfVectorizer` from `sklearn` to extract text-based features from item metadata (like descriptions or titles).
- Computes **cosine similarity** between items to suggest similar ones.

###  Collaborative Filtering
- Uses **`SVD`** from `surprise` to learn latent user and item features from historical interaction data (e.g., ratings).
- Predicts how a user would rate unseen items based on matrix factorization.

###  Hybrid Logic
- Combines both scores (e.g., via weighted average or rule-based merging).
- Allows fallback: if one method has insufficient data, the other can still provide recommendations.

---

## ⚙️ Dependencies

Make sure to install the following (if running in **Google Colab**, execute these in a code cell):

```python
!pip install numpy
!pip install scipy
!pip install scikit-learn
!pip install scikit-surprise
```

### How to Run
1. Clone the repository:
```bash 
git clone https://github.com/your-username/hybrid-recommender.git
cd hybrid-recommender
```

2. Open the hybrid_recommender.ipynb notebook using:
  Jupyter Notebook on your local system

3. Run all cells to:
  -> Load data
  -> Build TF-IDF and SVD models
  -> Generate hybrid recommendations
