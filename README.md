
# Next Word or Phrase Prediction using Bi-LSTM

## 📜 Overview
This project is focused on building a **Next Word Prediction System** using deep learning techniques. Given a sequence of words, the model predicts the most probable next word, which has practical applications in **smart keyboards, chatbots, virtual assistants**, and **text generation systems**.

We utilize **Natural Language Processing (NLP)** combined with a **Bi-LSTM neural network** to model the sequential nature of language and predict contextually relevant next words.

## 📝 Dataset
- **Source**: Sherlock Holmes text corpus
- **Content**: Thousands of English sentences with diverse linguistic structures.
- **Structure**:
  - **Inputs**: Sequences of words (e.g., “I love to”)
  - **Target Output**: Next word in the sequence (e.g., “learn”)

### Example Training Pairs:
| Input Sequence | Target Word |
|----------------|------------:|
| "I"            | "love"      |
| "I love"       | "to"        |
| "I love to"    | "learn"     |

## 🔄 Data Preprocessing Pipeline
1. **Text Cleaning**: Lowercasing, punctuation removal, special character filtering.
2. **Tokenization**: Mapping words to unique integer IDs.
3. **Sequencing**: Creating input-output pairs of varying sequence lengths.
4. **Padding**: Ensuring uniform input length for model training.

## 🧠 Model Architecture
- **Embedding Layer**: Converts word indices into dense vectors.
- **Bi-LSTM Layer**: Captures forward and backward context in sequences.
- **Dense Softmax Layer**: Outputs probabilities for predicting the next word.

### Training Configuration:
| Parameter     | Value                   |
|---------------|-------------------------|
| Loss Function | Categorical Crossentropy |
| Optimizer     | Adam                    |
| Metrics       | Accuracy, Top-5 Accuracy |
| Batch Size    | Based on system resources |
| Epochs        | Tuned via experimentation |

## 📊 Evaluation Metrics
- **Accuracy**: Measures correct predictions.
- **Top-5 Accuracy**: Checks if the correct next word is among the top 5 suggestions.
- **Perplexity** *(Optional)*: Evaluates model uncertainty (lower is better).

## ⚙️ Tools & Libraries
- **Python** (Jupyter Notebook)
- **TensorFlow / Keras**
- **NumPy, Pandas**
- **Matplotlib**

## 🚀 Workflow
1. Load & preprocess the dataset.
2. Tokenize and create training sequences.
3. Build and compile the Bi-LSTM model.
4. Train and evaluate the model.
5. Predict next words given partial input text.

## 🏆 Results
- Achieved **Top-1 Accuracy of ~96%** and **Top-5 Accuracy of ~99%** on the test set.
- Model effectively predicts contextually correct next words in given sequences.

## 🔮 Future Enhancements
- Upgrade to **Transformer-based architectures (e.g., GPT, BERT)**.
- Train on larger and more diverse corpora for better generalization.
- Deploy as an **API** or integrate with **smart keyboards & chatbots**.

## 📂 Project Structure
```
├── data/
│   └── sherlock_holmes.txt
├── notebooks/
│   └── next_word_prediction.ipynb
├── models/
│   └── next_word_model.h5
├── README.md
└── requirements.txt
```

## 📬 Contact
For any queries, reach out at:  
📧 [sahithkumarsangi1807@gmail.com](mailto:sahithkumarsangi1807@gmail.com)
