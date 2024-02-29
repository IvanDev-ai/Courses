import streamlit as st
import pandas as pd 
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from io import BytesIO

model = None
train = None
uploaded_file = st.file_uploader("modelo_entrenado", type=["h5"])
model_file = BytesIO(uploaded_file.read())
model = load_model(model_file)

MAX_SEQUENCE_LENGTH = 100

st.title('Clasificador de Comentarios')

st.write("Solo responde en ingles ya que fue entrenado con una base de datos anglosajona!")

tokenizer = Tokenizer()
uploaded_file = st.file_uploader("train", type=["csv"])
train_file = BytesIO(uploaded_file.read())
train = pd.read_csv(train_file)
sentences = train["comment_text"].fillna("DUMMY_VALUE").values
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
