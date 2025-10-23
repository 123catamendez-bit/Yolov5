import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# Configuración de página Streamlit
st.set_page_config(
    page_title="♻️ Detección de Materiales Sostenibles",
    page_icon="🌍",
    layout="wide"
)

# Función para cargar el modelo YOLOv5 (detección de objetos)
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
        1. Instala una versión compatible de PyTorch y YOLOv5:
           ```
           pip install torch==1.12.0 torchvision==0.13.0
           pip install yolov5==7.0.9
           ```
        2. Verifica que el archivo del modelo esté en la ubicación correcta.
        3. Si el error continúa, descarga el modelo desde torch hub.
        """)
        return None


# ------------------ INTERFAZ ------------------

st.title("🌱 Detección de Materiales Sostenibles")
st.markdown("""
Esta aplicación te permite **identificar materiales reciclados o sostenibles** en imágenes.
Puedes usar tu cámara para analizar objetos y ver si podrían ser aptos para procesos de diseño sostenible.
""")

# // imagen de encabezado aquí (por ejemplo, una de tus fotos de materiales reciclados)

# Cargar modelo
with st.spinner("🔄 Cargando modelo de detección..."):
    model = load_yolov5_model()

if model:
    # Configuración lateral
    st.sidebar.title("⚙️ Parámetros de Detección")
    with st.sidebar:
        st.subheader('Ajustes del modelo')
        model.conf = st.slider('Nivel de confianza mínima', 0.0, 1.0, 0.25, 0.01)
        model.iou = st.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)
        st.caption(f"Confianza: {model.conf:.2f} | IoU: {model.iou:.2f}")
        st.subheader('Opciones avanzadas')
        try:
            model.agnostic = st.checkbox('Detección sin distinción de clases', False)
            model.multi_label = st.checkbox('Permitir múltiples etiquetas', False)
            model.max_det = st.number_input('Máx. objetos detectados', 10, 2000, 1000, 10)
        except:
            st.warning("Algunas opciones avanzadas no están disponibles.")

    # Cámara o imagen
    main_container = st.container()
    with main_container:
        picture = st.camera_input("📸 Toma una foto del material o producto", key="camera")

        if picture:
            bytes_data = picture.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            with st.spinner("🔍 Analizando sostenibilidad del material..."):
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
                    st.subheader("📷 Imagen Analizada")
                    results.render()
                    st.image(cv2_img, channels='BGR', use_container_width=True)

                with col2:
                    st.subheader("🔎 Resultados de Detección")
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
                            "Material": label,
                            "Cantidad detectada": count,
                            "Confianza promedio": f"{confidence:.2f}"
                        })

                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df, use_container_width=True)
                        st.bar_chart(df.set_index('Material')['Cantidad detectada'])
                    else:
                        st.info("No se detectaron materiales. Intenta con otra imagen o baja el nivel de confianza.")

            except Exception as e:
                st.error(f"Error al procesar los resultados: {str(e)}")
                st.stop()

else:
    st.error("No se pudo cargar el modelo. Revisa las dependencias y vuelve a intentarlo.")
    st.stop()


# ------------------ PIE DE PÁGINA ------------------
st.markdown("---")
st.caption("""
🌿 **Acerca del proyecto:**  
Esta app fue adaptada para explorar cómo la IA puede apoyar el **diseño sostenible**,  
ayudando a reconocer materiales reciclados, reutilizables o ecológicos a partir de imágenes.  
Desarrollada con **Streamlit**, **YOLOv5** y una visión creativa hacia un futuro más verde ♻️
""")

# // imagen final aquí (por ejemplo, tu logo o una composición de tus piezas sostenibles)
