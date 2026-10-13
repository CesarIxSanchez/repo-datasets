import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Práctica 4. Modelos de machine learning", page_icon="🤖", layout="wide")

# --- ESTILO MINIMALISTA MODIFICADO ---
# Se cambió el color de fondo y texto de las instrucciones para mejor contraste en tema oscuro/claro.
st.markdown("""
    <style>
    .stButton>button {background-color: #3498db; color: white; border-radius: 5px; border: none; padding: 10px 20px;}
    .stButton>button:hover {background-color: #2980b9;}
    .instrucciones {
        background-color: #1e3a5f; /* Azul marino oscuro */
        color: #ffffff; /* Texto blanco puro */
        padding: 15px; 
        border-radius: 8px; 
        border-left: 5px solid #00d2ff; /* Borde cyan brillante */
        margin-bottom: 20px;
        font-size: 1.05em;
    }
    .dataset-subtitle {
        color: #888888;
        font-style: italic;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# --- CABECERA ---
st.title(" 🤖 Práctica 4. Modelos de machine learning")
st.markdown("**Alumno:** César Antonio Ix Sánchez. 66271")
st.markdown("---")

# --- NAVEGACIÓN MEDIANTE PESTAÑAS ---
tab1, tab2, tab3 = st.tabs(["1. Pattern Recognition (Clasificación)", "2. Fitting (Regresión)", "3. Series de Tiempo"])

# ==========================================
# PESTAÑA 1: PATTERN RECOGNITION (CLASIFICACIÓN)
# ==========================================
with tab1:
    st.header("Pattern Recognition")
    st.markdown("<div class='dataset-subtitle'>Dataset: predictive_maintenance_dataset.csv</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="instrucciones">
        <b>Instrucciones de uso:</b><br>
        1. Observa la vista previa del dataset industrial cargado desde GitHub.<br>
        2. Presiona el botón <b>"Entrenar y Procesar Modelo"</b>.<br>
        3. El programa aplicará balanceo de clases (SMOTE), entrenará un Perceptrón Multicapa (MLP) y mostrará las métricas y la matriz de confusión. (Puede tardar entre 1 y 2 minutos).
    </div>
    """, unsafe_allow_html=True)

    url_class = "https://raw.githubusercontent.com/CesarIxSanchez/repo-datasets/refs/heads/main/pattern-recognition/predictive_maintenance_dataset.csv"
    
    @st.cache_data
    def load_class_data():
        return pd.read_csv(url_class)
    
    try:
        df_class = load_class_data()
        st.subheader("Vista Previa del Dataset (parte del Dataset)")
        st.dataframe(df_class.head(100), use_container_width=True) 
        
        if st.button("Entrenar y Procesar Modelo", key="btn_class"):
            with st.spinner("Entrenando red neuronal..."):
                columnas_features = ['metric1', 'metric2', 'metric3', 'metric4', 'metric5', 'metric6', 'metric7', 'metric8', 'metric9']
                X = df_class[columnas_features]
                y = df_class['failure']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                smote = SMOTE(random_state=42)
                X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
                
                mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
                mlp.fit(X_train_smote, y_train_smote)
                
                y_pred = mlp.predict(X_test_scaled)
                
                st.success("¡Entrenamiento completado!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Métricas de Rendimiento")
                    st.info(f"**Accuracy (Precisión Global):** {accuracy_score(y_test, y_pred) * 100:.2f}%")
                    st.text("Reporte de Clasificación:")
                    st.text(classification_report(y_test, y_pred))
                
                with col2:
                    st.subheader("Matriz de Confusión")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(5,3))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
                    ax.set_xlabel('Predicción')
                    ax.set_ylabel('Real')
                    st.pyplot(fig)
    except Exception as e:
        st.error(f"Error al cargar el dataset: {e}")

# ==========================================
# PESTAÑA 2: FITTING (REGRESIÓN)
# ==========================================
with tab2:
    st.header("Fitting")
    st.markdown("<div class='dataset-subtitle'>Dataset: medical_cost_personal_datasets.csv</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="instrucciones">
        <b>Instrucciones de uso:</b><br>
        1. Verifica los datos comerciales de seguros médicos.<br>
        2. Presiona <b>"Entrenar y Procesar Modelo"</b> para que la red neuronal estime los costos ('charges').<br>
        3. Observa la gráfica de dispersión para comparar visualmente qué tan cerca están las estimaciones del modelo frente a las facturas médicas reales.
    </div>
    """, unsafe_allow_html=True)

    url_reg = "https://raw.githubusercontent.com/CesarIxSanchez/repo-datasets/refs/heads/main/fitting/medical_cost_personal_datasets.csv"
    
    @st.cache_data
    def load_reg_data():
        return pd.read_csv(url_reg)
    
    try:
        df_reg = load_reg_data()
        st.subheader("Vista Previa del Dataset (parte del Dataset)")
        st.dataframe(df_reg.head(100), use_container_width=True)
        
        if st.button("Entrenar y Procesar Modelo", key="btn_reg"):
            with st.spinner("Entrenando red neuronal de regresión..."):
                df_reg_encoded = pd.get_dummies(df_reg, columns=['sex', 'smoker', 'region'], drop_first=True)
                
                X = df_reg_encoded.drop('charges', axis=1)
                y = df_reg_encoded['charges']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                
                X_train_scaled = scaler_X.fit_transform(X_train)
                X_test_scaled = scaler_X.transform(X_test)
                y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
                
                mlp_reg = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000, random_state=42)
                mlp_reg.fit(X_train_scaled, y_train_scaled)
                
                y_pred_scaled = mlp_reg.predict(X_test_scaled)
                y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                
                st.success("¡Entrenamiento completado!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Métricas de Regresión")
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                    
                    st.info(f"**Raíz del Error Cuadrático Medio (RMSE):** ${rmse:.2f}")
                    st.info(f"**Coeficiente de Determinación (R²):** {r2:.4f}")
                    st.write("Un R² cercano a 1 indica que el modelo se ajusta bien a los datos comerciales.")
                
                with col2:
                    st.subheader("Valores Reales vs Predichos")
                    fig, ax = plt.subplots(figsize=(5,4))
                    ax.scatter(y_test, y_pred, alpha=0.5, color='#3498db')
                    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                    ax.set_xlabel('Costo Real')
                    ax.set_ylabel('Costo Predicho')
                    st.pyplot(fig)
    except Exception as e:
        st.error(f"Error al cargar el dataset: {e}")

# ==========================================
# PESTAÑA 3: SERIES DE TIEMPO
# ==========================================
with tab3:
    st.header("Series de Tiempo")
    st.markdown("<div class='dataset-subtitle'>Dataset: human_activity_recognition_with_smartphones.csv</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="instrucciones">
        <b>Instrucciones de uso:</b><br>
        1. Este dataset contiene señales temporales de acelerómetros y giroscopios.<br>
        2. Presiona <b>"Entrenar y Procesar Modelo"</b> para clasificar la serie de tiempo en actividades físicas.<br>
        3. Dado el gran número de variables, el entrenamiento tomará un par de minutos.
    </div>
    """, unsafe_allow_html=True)

    url_ts = "https://raw.githubusercontent.com/CesarIxSanchez/repo-datasets/refs/heads/main/series-de-tiempo/human_activity_recognition_with_smartphones.csv"
    
    @st.cache_data
    def load_ts_data():
        return pd.read_csv(url_ts)
    
    try:
        df_ts = load_ts_data()
        st.subheader("Vista Previa del Dataset (parte del Dataset)")
        st.dataframe(df_ts.iloc[:, list(range(5)) + list(range(-3, 0))].head(50), use_container_width=True)
        
        if st.button("Entrenar y Procesar Modelo", key="btn_ts"):
            with st.spinner("Entrenando red neuronal para series de tiempo..."):
                df_ts = df_ts.drop('subject', axis=1, errors='ignore')
                
                X = df_ts.drop('Activity', axis=1)
                y_text = df_ts['Activity']
                
                le = LabelEncoder()
                y = le.fit_transform(y_text)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                mlp_ts = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam', max_iter=300, random_state=42)
                mlp_ts.fit(X_train_scaled, y_train)
                
                y_pred = mlp_ts.predict(X_test_scaled)
                
                st.success("¡Entrenamiento completado!")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.subheader("Métricas de Clasificación")
                    st.info(f"**Accuracy:** {accuracy_score(y_test, y_pred) * 100:.2f}%")
                    st.text("Reporte:")
                    st.text(classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred)))
                
                with col2:
                    st.subheader("Matriz de Confusión")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(6,4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', cbar=False, 
                                xticklabels=le.classes_, yticklabels=le.classes_)
                    plt.xticks(rotation=45)
                    plt.yticks(rotation=0)
                    ax.set_xlabel('Predicción')
                    ax.set_ylabel('Real')
                    st.pyplot(fig)
    except Exception as e:
        st.error(f"Error al cargar el dataset: {e}")