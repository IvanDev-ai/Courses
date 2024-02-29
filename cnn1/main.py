import streamlit as st
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import gdown

MAX_SEQUENCE_LENGTH = 100

# Descargar el archivo CSV desde Google Drive
url = 'https://drive.google.com/uc?id=1Sf7AjBoEC1-ZtkrmAnq0frTWMAuh5osb'
output = 'train.csv'
gdown.download(url, output, quiet=False)

train = pd.read_csv(output)
sentences = train["comment_text"].fillna("DUMMY_VALUE").values

# Cargar el modelo
def load_h5_model():
    return load_model("modelo_entrenado.h5")

model = load_h5_model()

MAX_SEQUENCE_LENGTH = 100

st.title('Clasificador de Comentarios')

st.write("Solo responde en ingles ya que fue entrenado con una base de datos anglosajona!")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

# Procesamiento de la entrada del usuario
input_text = st.text_input('Ingrese un comentario:', 'you are an idiot')
input_sequence = tokenizer.texts_to_sequences([input_text])
input_sequence = pad_sequences(input_sequence, maxlen=MAX_SEQUENCE_LENGTH)

# Predicción
predictions = model.predict(input_sequence)

possible_labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

# Mostrar el resultado de la predicción
st.write('Predicciones:')
for label, prob in zip(possible_labels, predictions[0]):
    st.write(f"{label}:")
    st.progress(prob.astype(float))  # Convertir a tipo float
