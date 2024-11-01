import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Titre de l'application
st.title("Segmentation de Document par Seuillage Adaptatif")

# Chargement de l'image
uploaded_file = st.file_uploader("Choisissez une image de document", type=["jpg", "jpeg", "png"])

# Bouton pour segmenter l'image
if uploaded_file is not None:
    # Lire l'image en niveaux de gris
    image = Image.open(uploaded_file)
    image = np.array(image.convert('L'))  # Convertir en niveaux de gris (échelle de gris)

    # Affichage de l'image originale
    st.subheader("Image Originale")
    st.image(image, use_column_width=True, caption="Image originale")

    # Bouton pour appliquer la segmentation
    if st.button("Segmenter l'Image"):
        # Application d'un flou gaussien pour réduire le bruit
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # Application du seuillage adaptatif
        adaptive_threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY, 11, 2)

        # Affichage de l'image segmentée
        st.subheader("Image Segmentée (Seuillage Adaptatif)")
        st.image(adaptive_threshold, use_column_width=True, caption="Image segmentée")

        # Option pour télécharger le résultat
        result = Image.fromarray(adaptive_threshold)
        st.download_button("Télécharger l'image segmentée", 
                           data=result.tobytes(), 
                           file_name="image_segmentee.png", 
                           mime="image/png")
