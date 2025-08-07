# sentiment-analyzer-from-scratch

## Description
This is a simple sentiment analysis model made entirely from numPy, re, and NLTK alone. During development, pandas and matplotlib were also used. The model uses self-made embeddings, positional encodings, transformers, and logits. It employs several concepts from calculus and linear algebra to perform the forward and backward passes. 

The model was trained on 417 entries of [this dataset](https://www.kaggle.com/datasets/nursyahrina/chat-sentiment-dataset) from KaggleHub. It takes a test and classifies it to one of three sentiments - negative, neutral, or positive.

Given that this model was developed from scratch using limited samples, it is primitive and makes the best predictions from simple, straightforward sentences, similar to the ones you can see in the dataset.

## How it Works
[Will work on this later]

## Model Performance with Different Learning Rates

### learning_rate = 1e-03
Within the 92 samples in the test dataset, this model had an average accuracy of 72.77%.
![metrics for 1e-03](assets/1e_03_1.png)
![epoch loss for 1e-03](assets/1e_03_2.png)
![accuracy by class for 1e-03](assets/1e_03_3.png)

[The rest are in the works].

## Installation & Running the Model
Clone the repository:
```
https://github.com/addinar/sentiment-analyzer-from-scratch.git
```

After cloning the repository, simply follow the instructions in the demo notebook, saved as `notebooks/sentiment_analyzer_demo.ipynb`.

## License
Distributed under MIT License.
