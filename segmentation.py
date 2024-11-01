import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def process_image(image, block_size, C):
    # Convertir l'image PIL en array numpy
    img_array = np.array(image)
    
    # Convertir en niveaux de gris si l'image est en couleur
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
        
    # Appliquer le seuillage adaptatif
    # block_size doit être impair
    if block_size % 2 == 0:
        block_size += 1
        
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C
    )
    
    return binary

def main():
    st.title("Segmentation d'Image par Seuillage Adaptatif")
    
    # Zone de téléchargement de l'image
    uploaded_file = st.file_uploader("Choisissez une image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Lire et afficher l'image originale
        image = Image.open(uploaded_file)
        st.subheader("Image Originale")
        st.image(image, use_column_width=True)
        
        # Paramètres ajustables
        st.sidebar.header("Paramètres de Segmentation")
        block_size = st.sidebar.slider(
            "Taille du bloc",
            3, 99, 11, 2,
            help="Taille du voisinage pour le calcul du seuil. Doit être impair."
        )
        C = st.sidebar.slider(
            "Constante C",
            -30, 30, 2,
            help="Constante soustraite de la moyenne. Ajuste la sensibilité du seuillage."
        )
        
        # Bouton pour lancer la segmentation
        if st.button("Segmenter l'image"):
            # Traiter l'image
            segmented = process_image(image, block_size, C)
            
            # Afficher le résultat
            st.subheader("Image Segmentée")
            st.image(segmented, use_column_width=True)
            
            # Option de téléchargement du résultat
            buf = io.BytesIO()
            Image.fromarray(segmented).save(buf, format='PNG')
            st.download_button(
                label="Télécharger l'image segmentée",
                data=buf.getvalue(),
                file_name="image_segmentee.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()
