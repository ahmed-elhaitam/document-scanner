import streamlit as st
import numpy as np
from PIL import Image
import io
from skimage import filters
from skimage.filters import threshold_local
from skimage import img_as_ubyte

def process_image(image, block_size, offset):
    # Convertir l'image PIL en array numpy
    img_array = np.array(image)
    
    # Convertir en niveaux de gris si l'image est en couleur
    if len(img_array.shape) == 3:
        # Convertir RGB en niveaux de gris en prenant la moyenne des canaux
        gray = np.mean(img_array, axis=2).astype(np.uint8)
    else:
        gray = img_array
        
    # Appliquer le seuillage adaptatif
    if block_size % 2 == 0:
        block_size += 1
    
    local_thresh = threshold_local(gray, block_size=block_size, offset=offset, method='gaussian')
    binary = gray > local_thresh
    
    # Convertir en uint8 pour l'affichage
    return img_as_ubyte(binary)

def main():
    st.title("Segmentation d'Image par Seuillage Adaptatif")
    
    # Instructions d'installation
    st.markdown("""
    ### Installation requise :
    ```bash
    pip install streamlit scikit-image pillow numpy
    ```
    """)
    
    # Zone de téléchargement de l'image
    uploaded_file = st.file_uploader("Choisissez une image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        try:
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
            offset = st.sidebar.slider(
                "Offset",
                -30.0, 30.0, 0.0,
                help="Valeur soustraite du seuil calculé. Ajuste la sensibilité."
            )
            
            # Bouton pour lancer la segmentation
            if st.button("Segmenter l'image"):
                with st.spinner('Segmentation en cours...'):
                    # Traiter l'image
                    segmented = process_image(image, block_size, offset)
                    
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
                    
        except Exception as e:
            st.error(f"Une erreur s'est produite lors du traitement de l'image: {str(e)}")

if __name__ == "__main__":
    main()
