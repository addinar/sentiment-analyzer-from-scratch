import numpy as np
import nltk 
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk import WordNetLemmatizer
import re

class SimpleSentimentAnalyzer:
  def __init__(self, word2idx, embedding_matrix, transformer_blocks, classifier_head, label2idx, idx2label):
    self.dim = 128
    self.lemmatizer = WordNetLemmatizer()
    self.word2idx = word2idx
    self.embedding_matrix = embedding_matrix
    self.transformer_blocks = transformer_blocks
    self.classifier_head = classifier_head
    self.label2idx = label2idx
    self.idx2label = idx2label

  def tokenize(self, text):
    text = re.sub(r"http\\S+|www\\S+", "", text)
    text = re.sub(r"@[\\w_]+", "", text)
    text = re.sub(r"#(\\w+)", r"\\1", text)
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    tokens = text.split()
    tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
    return tokens

  def generate_embeddings(self, text):
    tokens = self.tokenize(text)
    indices = [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]
    embeddings = []

    for index in indices:
      embeddings.append(self.embedding_matrix[index])

    return np.array(embeddings)

  def positional_encode(self, embeddings): #input an embedded sentence
    for pos, embedding in enumerate(embeddings):
      for i, _ in enumerate(embedding):
        exp = (2 * i) / self.dim
        den = 10000 ** exp
        val = pos / den
        if i % 2 == 0:
          embedding[i] += np.sin(val)
        else:
          embedding[i] += np.cos(val)
      embeddings[pos] = embedding

    return embeddings

  def softmax(self, x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
  
  def multi_head_attention(self, X, W_q_heads, W_k_heads, W_v_heads, W_o):
    outputs = []
    A_list = []
    Q_list = []
    K_list = []
    V_list = []

    for idx, (W_q, W_k, W_v), in enumerate(zip(W_q_heads, W_k_heads, W_v_heads)):
      Q = X @ W_q
      K = X @ W_k
      V = X @ W_v

      scores = Q @ K.T
      head_dim = Q.shape[-1]
      scores_scaled = scores / np.sqrt(head_dim)
      A = self.softmax(scores_scaled)

      output = A @ V

      outputs.append(output)
      A_list.append(A)
      Q_list.append(Q)
      K_list.append(K)
      V_list.append(V)

    concat = np.concatenate(outputs, axis=-1)

    out = concat @ W_o

    return out, A_list, Q_list, K_list, V_list

  def add_substep(self, X, substep): 
    return X + substep

  def LayerNorm(self, X, gamma, beta, eps=1e-5):
    mean = X.mean(axis=1, keepdims=True)
    var = X.var(axis=1, keepdims=True)
    norm = (X - mean) / np.sqrt(var + eps)
    output = gamma * norm + beta

    return output
  
  def relu(self, x):
    return np.maximum(0, x)
  
  def ffn(self, X, W_1, W_2, b_1, b_2):
    h1 = X @ W_1 + b_1
    a1 = self.relu(h1)

    output = a1 @ W_2 + b_2

    return output, a1, h1
  
  def forward(self, text):
    X = self.generate_embeddings(text)
    X = self.positional_encode(X)

    for block in self.transformer_blocks:
      X = block.forward(X, self)

    x_cls = X[0]
    probs, logits = self.classifier_head.probabilities(x_cls)

    self.classifier_head.cached['logits'] = logits
    self.classifier_head.cached['x_cls'] = x_cls
    self.classifier_head.cached['probs'] = probs

    return probs, logits
  
  def predict(self, text):
    self.forward(text)
    probs = self.classifier_head.cached['probs']
    pred_idx = np.argmax(probs)
    return self.idx2label[pred_idx]