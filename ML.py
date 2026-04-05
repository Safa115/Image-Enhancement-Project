import streamlit as st
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# =========================
# Constants
# =========================
classes = ['Normal', 'Viral Pneumonia', 'Lung_Opacity']
IMG_SIZE = 96   # small size to reduce computation
EPOCHS = 2      # keep small for Streamlit
BATCH_SIZE = 32

# =========================
# Cached Dataset Loader
# =========================
@st.cache_data
def load_dataset(path):
    X, y = [], []

    for label in range(3):
        folder = os.path.join(path, classes[label])
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0

            X.append(img)
            y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = to_categorical(y, 3)

    return X, y

# =========================
# CNN Model
# =========================
def build_cnn():
    model = Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,1)),
        MaxPooling2D(2,2),

        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# =========================
# Confusion Matrix Plot
# =========================
def plot_cm(cm, title):
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)
    ax.set_title(title)

    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    return fig

# =========================
# Streamlit UI
# =========================
st.title("CNN Performance Before & After Image Preprocessing")

original_path = r"C:\DataSet\Lung X-Ray Image\Lung X-Ray Image"
processed_path = r"C:\DataSet\Lung X-Ray Image\Lung X-Ray Image_Enhanced"

if st.button("Run CNN Comparison"):
    with st.spinner("Training CNN models... please wait ⏳"):

        for name, path in {
            "Original Images": original_path,
            "Preprocessed Images": processed_path
        }.items():

            # Load data (cached)
            X, y = load_dataset(path)

            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = build_cnn()
            model.fit(
                Xtr, ytr,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0
            )

            preds = np.argmax(model.predict(Xte, verbose=0), axis=1)
            true = np.argmax(yte, axis=1)

            acc = accuracy_score(true, preds) * 100
            cm = confusion_matrix(true, preds)

            st.subheader(f"CNN – {name}")
            st.write(f"Accuracy: **{acc:.2f}%**")
            st.pyplot(plot_cm(cm, f"Confusion Matrix – {name}"))
