import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D,MaxPooling1D,Embedding
from keras.models import Model
from sklearn.metrics import roc_auc_score


MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 10

print("Loading word vectors...")
word2vec = {}
with open(os.path.join("glove.6B.%sd.txt" % EMBEDDING_DIM), encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:],dtype="float32")
        word2vec[word] = vec
print("Fount %s word vectors."  % len(word2vec))

print("Loading in comments...")
train = pd.read_csv("train.csv")
sentences = train["comment_text"].fillna("DUMMY_VALUE").values
possible_labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
targets = train[possible_labels].values


print("max sequence length:", max(len(s) for s in sentences))
print("max sequence length:", min(len(s) for s in sentences))
s = sorted(len(s) for s in sentences)
print("median sequence length:", s[len(s) // 2])

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

word2idx = tokenizer.word_index
print("Found %s inque tokens." % len(word2idx))


data = pad_sequences(sequences, maxlen= MAX_SEQUENCE_LENGTH)
print("Shape of data tensor:", data.shape)

print("Filling pre-trained embeddings...")
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


embedding_layer = Embedding(
    num_words,
    EMBEDDING_DIM,
    weights = [embedding_matrix],
    input_length = MAX_SEQUENCE_LENGTH,
    trainable = False
)

print("Building model...")
input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
x = embedding_layer(input_)
x = Conv1D(128,3,activation="relu")(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128,3,activation="relu")(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128,3,activation="relu")(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128,activation="relu")(x)
output = Dense(len(possible_labels), activation="sigmoid")(x)

model = Model(input_, output)
model.compile(loss="binary_crossentropy", 
              optimizer="rmsprop", 
              metrics=["accuracy"])

print("Training model...")
r = model.fit(
    data,
    targets,
    batch_size = BATCH_SIZE,
    epochs= EPOCHS,
    validation_split=VALIDATION_SPLIT
)

plt.plot(r.history["loss"],label="loss")
plt.plot(r.history["val_loss"],label="val_loss")
plt.legend()
plt.show()

plt.plot(r.history["accuracy"],label="acc")
plt.plot(r.history["val_accuracy"],label="val_acc")
plt.legend()
plt.show()

p = model.predict(data)
aucs = []
for j in range(6):
    auc = roc_auc_score(targets[:,j], p[:,j])
    aucs.append(auc)

# Guardar el modelo
model.save('modelo_entrenado.h5')
print("Modelo guardado correctamente.")

print(np.mean(aucs))