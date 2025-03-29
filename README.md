# Neural Machine Translation Project


We're doing a deep learning project. The title is "Design a deep learning model for Deep Learning-based Language Translation". For the dataset we'll be using WMT 2014 English-German dataset. We need to train three models. I'm thinking to start from basic encoder decoder RNN LSTM, then the base transformer and one more approach. 

The deliverables are as below:

1. A comprehensive report of 4 pages including all the results.
2. A demonstration UI(We'll be using the best model weights for demo)
3. All the code files in a repository



## Dataset

**Dataset Name:** WMT 2014 English-German

**Description:** A parallel corpus for machine translation tasks, containing English and German sentence pairs.

**Languages:** English (en), German (de)

**Size:**
- **Training Data:** Approximately 4.5 million sentence pairs.
- **Validation Data:** Commonly uses "newstest2013" with 3,000 sentence pairs.
- **Test Data:** "newstest2014" comprising 3,003 sentence pairs.

**Data Format:** Plain text files with aligned sentences; English and German sentences are provided in separate files.

**Source:** Derived from the Europarl corpus, News Commentary corpus, and Common Crawl corpus.

**Access:**
- Official Website: [http://www.statmt.org/wmt14/translation-task.html](http://www.statmt.org/wmt14/translation-task.html)
- Kaggle: [https://www.kaggle.com/datasets/mohamedlotfy50/wmt-2014-english-german](https://www.kaggle.com/datasets/mohamedlotfy50/wmt-2014-english-german)

**Preprocessing Recommendations:**
- **Tokenization:** Segment text into tokens using tools like Moses tokenizer.
- **Truecasing:** Normalize text casing to maintain consistency.
- **Subword Segmentation:** Apply Byte Pair Encoding (BPE) to handle rare and compound words.

**Usage:** Widely used for training and evaluating machine translation models, including RNNs, LSTMs, and Transformers.

**Licensing:** Usage terms are specified on the official website; users should review and comply with these terms.

**References:**
- Official WMT 2014 Translation Task Overview: [http://www.statmt.org/wmt14/translation-task.html](http://www.statmt.org/wmt14/translation-task.html)
- Kaggle Dataset Page: [https://www.kaggle.com/datasets/mohamedlotfy50/wmt-2014-english-german](https://www.kaggle.com/datasets/mohamedlotfy50/wmt-2014-english-german) 


