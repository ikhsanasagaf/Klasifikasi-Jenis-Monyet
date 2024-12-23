import os
import keras
from keras.models import load_model
import streamlit as st 
import tensorflow as tf
import numpy as np
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.colored_header import colored_header
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

base_dir = 'Images/'
img_size = 180
batch = 32

train_ds = tf.keras.utils.image_dataset_from_directory( base_dir,
                                                       seed = 123,
                                                       validation_split=0.2,
                                                       subset = 'training',
                                                       batch_size=batch,
                                                       image_size=(img_size,img_size))

val_ds = tf.keras.utils.image_dataset_from_directory( base_dir,
                                                       seed = 123,
                                                       validation_split=0.2,
                                                       subset = 'validation',
                                                       batch_size=batch,
                                                       image_size=(img_size,img_size))

AUTOTUNE = tf.data.AUTOTUNE

val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)

def Header():
    colored_header(
        label="Klasifikasi Jenis Monyet Menggunakan Algoritma CNN",
        description="Program Klasifikasi 10 Jenis Monyet dengan Menggunakan Basic Keras Model - Dataset yang digunakan pada Program ini terdapat sekitar 1098 Gambar dengan 80% Training & 20% Validation ",
        color_name="green-70",
    )

Header()
monkey_names = ['Bald Uakari', 'Black Headed Night Monkey', 'Common Squirrel Monkey','Japanese Macaque','Mantled Howler','Nilgiri Langur', 'Patas Monkey', 'Pygmy Marmoset','Silvery Marmoset','White Headed Capuchin']

model = load_model('Model_Monyet.h5')

def classify_images(image_path): 
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    predicted_label = np.argmax(predictions)
    result = tf.nn.softmax(predictions[0])

    y_pred = model.predict(val_ds)
    y_pred = np.argmax(y_pred, axis=1)
    
    y_true = np.concatenate([y for x, y in val_ds], axis=0)
    
    pred_filter = y_true==predicted_label
    acc = accuracy_score(y_true[pred_filter], y_pred[pred_filter])
    prec = precision_score(y_true[pred_filter], y_pred[pred_filter], average='weighted')
    recc = recall_score(y_true[pred_filter], y_pred[pred_filter], average='weighted')
    f1 = f1_score(y_true[pred_filter], y_pred[pred_filter], average='weighted')
        
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{acc:.2%}")
        st.metric("Precision", f"{prec:.2%}")
    with col2:
        st.metric("Recall", f"{recc:.2%}")
        st.metric("F1-Score", f"{f1:.2%}")

    outcome = 'Gambar diatas termasuk dalam Jenis ' + monkey_names[predicted_label] + ' dengan skor confidence ' + str(np.max(result) * 100)
    return outcome

uploaded_file = st.file_uploader('Silahkan Masukkan Gambar Monyet')
if uploaded_file is not None:
    with open(os.path.join('upload', uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, width = 200)

    with stylable_container(
        key="container_with_border",
        css_styles="""
            {
                border: 1px solid rgba(49, 51, 63, 0.2);
                border-radius: 0.5rem;
                background-color: #c9ffd2;
                padding: calc(1em - 1px)
            }
            """,
    ):
        st.markdown(classify_images(uploaded_file))





