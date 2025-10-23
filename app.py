import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# ------------------ CONFIGURACIÓN DE LA APP ------------------
st.set_page_config(
    page_title="🌌 Explorador Galáctico IA",
    page_icon="✨",
    layout="wide"
)

# ------------------ FUNCIÓN PARA CARGAR EL MODELO ------------------
@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        import yolov5
        try:
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            try:
                model = yolov5.load(model_path)
                return model
            except Exception:
                st.warning("Intentando método alternativo de carga...")
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if current_dir not in sys.path:
                    sys.path.append(current_dir)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.info("""
        💡 Recomendaciones:
        1. Instala las versiones recomendadas de PyTorch y YOLOv5:
           ```
           pip install torch==1.12.0 torchvision==0.13.0
           pip install yolov5==7.0.9
           ```
        2. Verifica la ubicación del modelo.
        3. Si persiste el error, usa el modelo desde Torch Hub.
        """)
        return None


# ------------------ INTERFAZ PRINCIPAL ------------------
st.markdown("<h1 style='text-align: center; color: #D9B3FF;'>🌠 Explorador Galáctico de Objetos</h1>", unsafe_allow_html=True)

st.markdown("""
Bienvenido al **Explorador Galáctico**, una app creada para **identificar y analizar elementos del universo visual** ✨  
Con ayuda de la inteligencia artificial, podrás reconocer objetos —como planetas, estrellas o texturas cósmicas— a partir de tus imágenes o fotos en tiempo real.  
""")

# // imagen aquí (imagen principal o banner del universo)

with st.spinner("🚀 Cargando el modelo galáctico..."):
    model = load_yolov5_model()

# ------------------ CONFIGURACIÓN LATERAL ------------------
if model:
    st.sidebar.markdown("<h2 style='color: #D9B3FF;'>⚙️ Panel de Control Estelar</h2>", unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Ajustes del escáner cósmico")
        model.conf = st.slider('Nivel de confianza mínima', 0.0, 1.0, 0.25, 0.01)
        model.iou = st.slider('Umbral IoU (superposición)', 0.0, 1.0, 0.45, 0.01)
        st.caption(f"Confianza: {model.conf:.2f} | IoU: {model.iou:.2f}")

        st.subheader("Opciones del universo")
        try:
            model.agnostic = st.checkbox('Modo sin distinción de galaxias', False)
            model.multi_label = st.checkbox('Permitir múltiples astros', False)
            model.max_det = st.number_input('Máx. cuerpos celestes detectados', 10, 2000, 1000, 10)
        except:
            st.warning("Algunas opciones avanzadas no están disponibles en esta versión del cosmos.")


    # ------------------ SECCIÓN PRINCIPAL ------------------
    main_container = st.container()
    with main_container:
        picture = st.camera_input("📸 Captura un objeto o textura galáctica", key="camera")

        # // imagen aquí (fondo o decoración tipo galaxia)
        
        if picture:
            bytes_data = picture.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            with st.spinner("🪐 Analizando composición estelar..."):
                try:
                    results = model(cv2_img)
                except Exception as e:
                    st.error(f"Error durante la detección: {str(e)}")
                    st.stop()

            try:
                predictions = results.pred[0]
                boxes = predictions[:, :4]
                scores = predictions[:, 4]
                categories = predictions[:, 5]

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("<h3 style='color: #D9B3FF;'>🔭 Mapa Galáctico Detectado</h3>", unsafe_allow_html=True)
                    results.render()
                    st.image(cv2_img, channels='BGR', use_container_width=True)

                with col2:
                    st.markdown("<h3 style='color: #D9B3FF;'>🌌 Cuerpos Celestes Identificados</h3>", unsafe_allow_html=True)

                    label_names = model.names
                    category_count = {}

                    for category in categories:
                        idx = int(category.item()) if hasattr(category, 'item') else int(category)
                        category_count[idx] = category_count.get(idx, 0) + 1

                    data = []
                    for category, count in category_count.items():
                        label = label_names[category]
                        confidence = scores[categories == category].mean().item() if len(scores) > 0 else 0
                        data.append({
                            "Objeto Galáctico": label,
                            "Cantidad": count,
                            "Confianza promedio": f"{confidence:.2f}"
                        })

                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df, use_container_width=True)
                        st.bar_chart(df.set_index('Objeto Galáctico')['Cantidad'])
                    else:
                        st.info("No se detectaron elementos galácticos. Intenta ajustar los parámetros del escáner.")

            except Exception as e:
                st.error(f"Error al procesar los resultados: {str(e)}")
                st.stop()
else:
    st.error("No se pudo conectar con el universo (modelo no cargado).")
    st.stop()


# ------------------ PIE DE PÁGINA ------------------
st.markdown("---")
st.caption("""
💫 **Explorador Galáctico** — Proyecto inspirado en la conexión entre el arte, la ciencia y el universo.  
Esta aplicación combina visión por computadora con una estética cósmica para **interpretar los objetos como si fueran cuerpos estelares**.  
Desarrollada con **Streamlit**, **YOLOv5** y mucha imaginación estelar. 🌠
""")

# // imagen aquí (logo o ilustración final de galaxia)
