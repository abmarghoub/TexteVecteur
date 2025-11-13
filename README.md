
# TP 2: Du texte au vecteur : comprendre la logique de la représentation textuelleF)

---
### Réalisé par

**Abla MARGHOUB**

### Encadré par

**Pr. Oumaima STITINI**

### Module

**Inteligencia Artificial**

### Établissement

**École Normale Supérieure - Université Cadi Ayyad**

---

**Objectif :**

- Convertir plusieurs phrases en vecteurs numériques à l’aide de `CountVectorizer`.
- Transformer les phrases en matrice BoW.
- Appliquer la pondération TF-IDF pour valoriser les mots importants.
- Visualiser les poids TF-IDF pour mieux comprendre l’importance de chaque mot.

## Partie 5 — Représentation Bag of Words (BoW)

### Étapes réalisées

**1. Installation et import des bibliothèques :**
```
from sklearn.feature_extraction.text import CountVectorizer
```
**2. Code Exemple :**
```
sample_texts = df['cleaned_review'].head(3).tolist()

vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(sample_texts)

print("Matrice BoW :")
print(X_bow.toarray())
print("\nListe des mots extraits :")
print(vectorizer.get_feature_names_out())
```
<img width="782" height="268" alt="image" src="https://github.com/user-attachments/assets/0b87fcf3-26ff-4faa-8766-5f9a60dd49e9" />

**3. Questions :**

* 1. Chaque colonne de la matrice correspond à un mot unique.
* 2. Les valeurs représentent le **nombre d’occurrences** du mot dans la phrase.
* 3. Limites de BoW : perte de l’ordre des mots et du contexte.

---

## Partie 6 — Représentation TF-IDF

### Étapes réalisées

**1. Installation et import des bibliothèques :**
```
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

```
**2. Code Exemple :**

```

tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(sample_texts)

# DataFrame pour visualiser
tfidf_df = pd.DataFrame(
    X_tfidf.toarray(),
    index=[f"Review {i+1}" for i in range(len(sample_texts))],
    columns=tfidf_vectorizer.get_feature_names_out()
)
print(tfidf_df)

# Heatmap pour mieux visualiser
plt.figure(figsize=(12,4))
sns.heatmap(tfidf_df, annot=True, cmap='Blues')
plt.title("TF-IDF des mots pour 3 reviews")
plt.show()
```
<img width="797" height="421" alt="image" src="https://github.com/user-attachments/assets/9858ccb4-71e5-4375-b19c-c113514422f5" />

**3. Questions :**

* 1.  La différence principale entre BoW et TF-IDF : TF-IDF **pèse les mots importants et rares**, tandis que BoW ne compte que les occurrences.
* 2. Les mots très fréquents dans toutes les phrases ont un poids faible.
* 3.  Exemple concret : pour classifier des reviews Twitter, les mots comme “love”, “hate”, “amazing” auront un poids élevé avec TF-IDF.

---

**Résumé attendu pour ces parties :**

* `X_bow` : matrice des occurrences de mots
* `X_tfidf` : matrice des poids TF-IDF
* Visualisation claire avec DataFrame et heatmap pour montrer l’importance des mots

