import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input,LSTM, Bidirectional, GlobalMaxPooling1D, Lambda, Concatenate, Dense
import keras.backend as K
from keras.preprocessing.image import img_to_array
from PIL import Image

def get_train(Limit=None):
    # Dataset: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
    df = pd.read_csv("mnist_train.csv")
    data = df.values
    np.random.shuffle(data)
    X = data[:,1:].reshape(-1,28,28) / 255
    Y = data[:,0]
    if Limit is not None:
        X,Y = X[:Limit], Y[:Limit]
    return X, Y

X, Y = get_train()

D = 28
M = 15

input_ = Input(shape= (D,D))
rnn1 = Bidirectional(LSTM(M, return_sequences=True))
x1 = rnn1(input_)
x1 = GlobalMaxPooling1D()(x1)

rnn2 = Bidirectional(LSTM(M, return_sequences=True))

permutor = Lambda(lambda t: K.permute_dimensions(t, pattern=(0,2,1)))

x2 = permutor(input_)
x2 = rnn2(x2)
x2 = GlobalMaxPooling1D()(x2)

concatenator = Concatenate(axis=1)
x = concatenator( [ x1, x2 ] )
output = Dense(10, activation= 'softmax')(x)

model = Model(inputs = input_, outputs = output)


model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer= 'Adam',
              metrics=['accuracy'])

print('Training model...')
r = model.fit(X,Y,batch_size=32,epochs=10,validation_split=0.3)


def predict_digit(model, image_path):
    img = Image.open(image_path).convert('L')

    img = img.resize((28, 28))

    img_array = img_to_array(img)

    img_array = img_array / 255.0

    img_array = img_array.reshape(1, 28, 28)

    prediction = model.predict(img_array)

    return np.argmax(prediction)

# Uso de la función
image_path = 'ej5.png'
print(predict_digit(model, image_path))