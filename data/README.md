# Dataset

Download the **ISOT Fake News Dataset** from Kaggle:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Place the two files here:

```
data/Fake.csv
data/True.csv
```

Each file has the columns `title, text, subject, date`.

Any other dataset works too: use `--data my.csv` with columns `text` (and optionally `title`) and `label` (1 = fake, 0 = real).
