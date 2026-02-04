# Evaluación del Rendimiento de Algoritmos de Clustering Online de Imágenes con Restricciones de Tamaño

**Proyecto Integrador - Ingeniería en Ciencias de la Computación**

Universidad Politécnica Salesiana - Sede Quito

---

## 📋 Información General

| Aspecto | Detalle |
|--------|--------|
| **Carrera** | Ingeniería en Ciencias de la Computación |
| **Nivel** | Séptimo Semestre |
| **Período** | 67 |
| **Fecha** | Enero - Febrero 2026 |

---

## 👨‍💼 Integrantes del Proyecto

1. **Barahona Guzmán, Damian Joshua** - Líder del proyecto
2. **Flores Chamorro, Matheo Gerson**
3. **Macas Moreno, Andy Joel**
4. **Licto Freire, Cristian Alexis**
5. **Tandazo Pineda, Juan Francisco**
6. **Torres Calero, Joseph Alexander**

---

## 🎯 Objetivo General

Crear un sistema inteligente para agrupamiento online de imágenes con restricciones de tamaño, evaluando el rendimiento de algoritmos de clustering online mediante diferentes técnicas de extracción de características.

---

## 📝 Descripción del Proyecto

El proyecto consiste en desarrollar e implementar un algoritmo de clustering online con restricciones de tamaño para agrupar imágenes utilizando diferentes descriptores y embeddings. 

### Características Principales:

- **Tres algoritmos de extracción de características:**
  - Momentos, HU, Zernike
  - SIFT o SURF
  - HOG (Histograma de Gradientes Orientados)

- **Embedding con red neuronal profunda** para representaciones de mayor nivel

- **Clustering online con restricciones de tamaño** para agrupamiento dinámico de imágenes

- **Evaluación:** Métricas de validación interna y externa con verificación de restricciones de tamaño

---



## 🛠️ Herramientas Tecnológicas

| Herramienta | Uso |
|------------|-----|
| **Python** | Lenguaje principal de desarrollo |
| **OpenCV** | Procesamiento y manipulación de imágenes |
| **TensorFlow/Keras** | Generación de embeddings con redes neuronales profundas |
| **scikit-learn** | Algoritmos de clustering y métricas de evaluación |
| **NumPy & Pandas** | Procesamiento de datos |
| **Matplotlib & Seaborn** | Visualización de resultados |
| **Google Colaboratory** | Entrenamiento en la nube (opcional) |
| **GitHub** | Control de versiones |

---

## 📊 Datasets Utilizados

### 1. E-commerce Products Image Dataset
- **Origen:** [Kaggle - E-commerce Products Image Dataset](https://www.kaggle.com/datasets/sunnykusawa/ecommerce-products-image-dataset)
- **Acceso:** https://www.kaggle.com/datasets/sunnykusawa/ecommerce-products-image-dataset
- **Instancias:** 796 imágenes en total
- **Clases:** 4 (Jeans, Sofá, Camiseta, TV)
- **Distribución:** 199 imágenes por clase
- **Formato:** JPG
- **Características:** Fondo blanco, objetos claramente definidos, alto contraste

### 2. Mechanical Tools Classification Dataset
- **Origen:** [Kaggle - Mechanical Tools Classification Dataset](https://www.kaggle.com/datasets/salmaneunus/mechanical-tools-dataset)
- **Acceso:** https://www.kaggle.com/datasets/salmaneunus/mechanical-tools-dataset
- **Instancias:** 1214 imágenes (seleccionadas de 7527 totales)
- **Clases:** 3 (Rope, Toolbox, Pliers)
- **Distribución:** 335 (Rope), 482 (Toolbox), 397 (Pliers)
- **Formato:** JPG
- **Características:** Fondos variados, objetos diversos con formas y texturas complejas

---

## 📈 Etapas del Proyecto

### **Etapa 1: Análisis y Preparación**

En esta etapa inicial se realiza:
- **Selección de datasets:** Descarga y análisis de conjuntos de imágenes, verificación de número de instancias y clases
- **Preparación de imágenes:** Aplicación de técnicas de mejora de contraste, eliminación de ruido y umbralización para garantizar la calidad de las imágenes

### **Etapa 2: Extracción de Características**

Durante esta etapa se llevan a cabo los procesos de:
- **Extracción de características:** Aplicación de tres algoritmos diferentes (Momentos/HU/Zernike, SIFT/SURF, HOG) para crear 3 conjuntos de datos vectorizados
- **Generación de embeddings:** Implementación de una red neuronal profunda para generar representaciones de embeddings de las imágenes

### **Etapa 3: Clustering y Evaluación**

En esta fase se realiza:
- **Aplicación del algoritmo:** Implementación de clustering online con restricciones de tamaño, pruebas iniciales con dataset de Iris y posterior aplicación a los datasets principales
- **Evaluación y comparación:** Cálculo de métricas internas (Silhouette, Dunn) y externas (NMI, ARI, AMI), verificación de restricciones de tamaño y análisis comparativo de resultados


## 🌐 Despliegue Web

### Objetivo
Crear una aplicación web interactiva que permita visualizar y probar el sistema de clustering online de imágenes.


---

## 🚀 Cómo clonar y ejecutar el proyecto

### 1. Clonar el repositorio

```bash
git clone https://github.com/Damian2044/G5_ProyectoIntegrador_ML_VC.git
cd G5_ProyectoIntegrador_ML_VC
```

---

### 2. Backend (FastAPI)

Requisitos previos:
- Python 3.10+ instalado
- pip disponible en la línea de comandos

Desde la carpeta raíz del proyecto:

```bash
cd backend

# (Opcional) Crear y activar entorno virtual en Windows PowerShell
python -m venv .venv
.venv\\Scripts\\activate

# Instalar dependencias
pip install -r requirements.txt


# Ejecutar servidor FastAPI con Uvicorn
uvicorn app.app:app --reload --host 0.0.0.0 --port 8000
```

El backend quedará disponible en:
- API base: http://localhost:8000
- Documentación interactiva: http://localhost:8000/docs

---

### 3. Frontend (aplicación web)

Requisitos previos:
- Node.js 18+ y npm instalados

En una nueva terminal, desde la carpeta raíz del proyecto:

```bash
cd frontend

# Instalar dependencias
npm install

# Ejecutar en modo desarrollo
npm run dev -- --host
```

Por defecto, la aplicación web quedará disponible en una URL similar a:

- Vite/React: http://localhost:5173




