import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from PIL import Image
import pandas as pd
import io

# Завантаження даних
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Завантаження моделі без оптимізатора
model = load_model('model.h5', compile=False)

# Повторна компіляція моделі
model.compile(optimizer=Adam(learning_rate=0.005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Оцінка моделі на тестових даних
test_loss, test_accuracy = model.evaluate(x_test, y_test)
st.write(f'Test accuracy: {test_accuracy:.4f}')

# Побудова графіків втрат та точності
def plot_history(history):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].plot(history.history['loss'], label='Train Loss')
    axs[0].plot(history.history['val_loss'], label='Validation Loss')
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(history.history['accuracy'], label='Train Accuracy')
    axs[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    st.pyplot(fig)

# Навчання моделі
history = model.fit(x_train, y_train,
                    epochs=50,
                    batch_size=64,
                    validation_data=(x_test, y_test),
                    callbacks=[
                        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
                    ],
                    verbose=2)

plot_history(history)

# Прогнозування на тестових даних
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Класифікаційний звіт
classification_report_str = classification_report(y_true, y_pred_classes, target_names=[
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
])

st.text(classification_report_str)

# Завантаження та відображення вхідного зображення
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((28, 28))
    image_np = np.array(image).reshape(-1, 28, 28, 1).astype('float32') / 255.0

    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Прогнозування на завантаженому зображенні
    pred = model.predict(image_np)
    pred_class = np.argmax(pred, axis=1)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    st.write(f'Predicted class: {class_names[pred_class[0]]}')

    # Відображення ймовірностей для кожного класу
    st.write('Class probabilities:')
    prob_df = pd.DataFrame(pred, columns=class_names)
    st.dataframe(prob_df.T)

    # Збереження результатів у буфер
    result_image = io.BytesIO()
    image.save(result_image, format='PNG')
    result_image.seek(0)

    result_csv = io.StringIO()
    prob_df.to_csv(result_csv)
    result_csv.seek(0)

    # Додавання посилань для завантаження результатів
    st.download_button(label='Download Image', data=result_image, file_name='result_image.png', mime='image/png')
    st.download_button(label='Download Probabilities', data=result_csv, file_name='result_probabilities.csv', mime='text/csv')

if __name__ == '__main__':
    st.title('Neural Network Visualization')
    st.write('This app visualizes the performance of a neural network model on the Fashion MNIST dataset.')
    st.write('## Model Accuracy')
    st.write(f'Test accuracy: {test_accuracy:.4f}')
    st.write('## Training History')
    plot_history(history)
    st.write('## Classification Report')
    st.text(classification_report_str)
    st.write('## Upload an Image for Prediction')

# Додати функцію для завантаження файлів
uploaded_file = st.file_uploader("Оберіть файл", type=["csv", "txt"])

if uploaded_file is not None:
    # Обробка завантаженого файлу, наприклад, можна зберегти його на сервері
    with open(uploaded_file.name, 'wb') as f:
        f.write(uploaded_file.getbuffer())

   # st.success('Файл успішно завантажено!')