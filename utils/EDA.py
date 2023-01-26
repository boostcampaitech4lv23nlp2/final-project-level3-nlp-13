from collections import Counter
from pprint import pprint

import pandas as pd
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords

kiwi = Kiwi()
stopwords = Stopwords()
c = Counter()
path = "data/preprocessed_data/naver/naver_corpus_1st_size_1337.csv"
df = pd.read_csv(path)

for (i, row) in df.iterrows():
    text = row["title"] + " " + row["body"]
    tokens = [token.form for token in kiwi.tokenize(text, stopwords=stopwords)]
    c.update(tokens)

counts = c.most_common()
pprint(counts[:100])
new_df = pd.DataFrame(columns=["word", "count"])
for (w, count) in counts:
    new_df.update(
        pd.DataFrame(
            {
                "word": w,
                "count": count,
            }
        )
    )

new_df.to_csv("vocab_counts.csv", index=False)
