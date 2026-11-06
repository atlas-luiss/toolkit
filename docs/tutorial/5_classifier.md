# 5. Choose the classifier

A model has two parts: a **backbone** that turns an input into a feature vector, and a
**classifier** (the "head") that turns the feature vector into a class. In continual learning the
classifier is often where forgetting shows: a trained head favours the newest classes.

There are three kinds of classifier, chosen with `classifier=`:

| kind | how it works | names |
|---|---|---|
| **trained** (default) | a layer trained with the rest, by gradient descent | `linear` (default), `cosine` |
| **statistical** | no training: after each task it stores simple statistics of each class's features (e.g. the mean) and classifies new samples by them; nothing to forget | `ncm`, `lda`, `klda`, `fecam`, `ranpac` |
| **correction** | keeps the trained head and fixes its bias towards new classes after each task | `slca`, `wa` |

```python
import cl_toolkit as clt

quick = dict(data="toy", model="mlp", num_tasks=5)
```

## A statistical head on a frozen model

With a good pre-trained model this is a strong baseline that needs no training at all:
`strategy="frozen"` (do not train) and `epochs=0`:

```python
for head in ["ncm", "lda", "fecam", "ranpac"]:
    r = clt.run(strategy="frozen", classifier=head, epochs=0, **quick)
    print(head, round(r["avg_accuracy"], 1))
```

Here the "pre-trained" model is the small random `mlp`; with a real pre-trained ViT on CIFAR-100,
RanPAC reaches 89.7% this way.

| head | idea |
|---|---|
| `ncm` | nearest class mean (cosine similarity to each class's average feature) |
| `lda` | class means + one shared covariance (a Gaussian classifier) |
| `klda` | LDA on random kernel features (more flexible boundaries) |
| `fecam` | one covariance per class (Mahalanobis distance) |
| `ranpac` | random projection + ridge regression |

## A head on top of a trained method

Any method can use any head. The method trains the backbone; the head classifies:

```python
r = clt.run(strategy="er", classifier="ncm", **quick)          # replay + nearest class mean
print("er + ncm", round(r["avg_accuracy"], 1))

r = clt.run(strategy="naive", classifier="slca", **quick)      # correct the trained head
print("naive + slca", round(r["avg_accuracy"], 1))
```

## Settings

Each head has options, passed with `classifier_kwargs`:

```python
r = clt.run(strategy="frozen", classifier="ranpac", classifier_kwargs={"proj_dim": 2000}, epochs=0, **quick)
r = clt.run(strategy="frozen", classifier="ncm", classifier_kwargs={"metric": "euclidean"}, epochs=0, **quick)
```

**Next:** [6. Routing](6_routing.md)
