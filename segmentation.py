import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import io

def simple_adaptive_threshold(image, block_size, constant):
    # Convertir l'image en numpy array
    img_array = np.array(image)
    
    # S'assurer que l'image est en niveaux de gris
    height, width = img_array.shape
    
    # Créer une image de sortie
    output = np.zeros((height, width), dtype=np.uint8)
    
    # Padding pour gérer les bords
    pad = block_size // 2
    padded_image = np.pad(img_array, pad, mode='edge')
    
    # Appliquer le seuillage adaptatif
    for i in range(height):
        for j in range(width):
            # Extraire le bloc local
            block = padded_image[i:i+block_size, j:j+block_size]
            # Calculer la moyenne locale
            threshold = np.mean(block) - constant
            # Appliquer le seuil
            output[i, j] = 255 if img_array[i, j] > threshold else 0
            
    return output

def main():
    st.title("Segmentation d'Image Simple")
    
    # Zone de téléchargement de l'image
    uploaded_file = st.file_uploader("Choisissez une image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        try:
            # Lire l'image
            image = Image.open(uploaded_file)
            
            # Convertir en niveaux de gris
            gray_image = ImageOps.grayscale(image)
            
            # Afficher l'image originale
            st.subheader("Image Originale")
            st.image(gray_image, use_column_width=True)
            
            # Paramètres ajustables
            st.sidebar.header("Paramètres de Segmentation")
            block_size = st.sidebar.slider(
                "Taille du bloc",
                3, 31, 11, 2,
                help="Taille du bloc pour le calcul de la moyenne locale. Valeur impaire recommandée."
            )
            constant = st.sidebar.slider(
                "Constante",
                0, 50, 10,
                help="Valeur soustraite de la moyenne. Ajuste le contraste."
            )
            
            # Bouton pour lancer la segmentation
            if st.button("Segmenter l'image"):
                with st.spinner('Segmentation en cours... Cela peut prendre quelques instants.'):
                    # Réduire la taille de l'image si elle est trop grande
                    max_size = 800
                    if gray_image.size[0] > max_size or gray_image.size[1] > max_size:
                        ratio = max_size / max(gray_image.size)
                        new_size = tuple(int(dim * ratio) for dim in gray_image.size)
                        gray_image = gray_image.resize(new_size, Image.Resampling.LANCZOS)
                    
                    # Traiter l'image
                    img_array = np.array(gray_image)
                    segmented = simple_adaptive_threshold(img_array, block_size, constant)
                    
                    # Afficher le résultat
                    st.subheader("Image Segmentée")
                    st.image(segmented, use_column_width=True)
                    
                    # Option de téléchargement du résultat
                    result_image = Image.fromarray(segmented)
                    buf = io.BytesIO()
                    result_image.save(buf, format='PNG')
                    st.download_button(
                        label="Télécharger l'image segmentée",
                        data=buf.getvalue(),
                        file_name="image_segmentee.png",
                        mime="image/png"
                    )

        except Exception as e:
            st.error(f"Une erreur s'est produite : {str(e)}")
            st.error("Conseils de dépannage : Essayez avec une image plus petite ou ajustez les paramètres.")

if __name__ == "__main__":
    main()
