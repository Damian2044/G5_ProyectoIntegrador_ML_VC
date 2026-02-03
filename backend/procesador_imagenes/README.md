# Procesador de imágenes

Este módulo implementa un pipeline completo de procesamiento de imágenes y extracción de características para diferentes conjuntos de datos. Los resultados se utilizan posteriormente en los experimentos de clustering (por ejemplo, en `backend/clustering/scripts/pruebas_imagenes.py`).

## Estructura general

Archivos principales:

- `backend/procesador_imagenes/main.py`: script principal de procesamiento.
- `backend/procesador_imagenes/config.py`: configuración de datasets, rutas y parámetros de preprocesamiento y extracción.
- Carpeta `backend/procesador_imagenes/src/`: código de soporte (carga de datos, preprocesamiento, extracción de características y gestión de resultados).

## Configuración (`config.py`)

El archivo `config.py` define:

- `CONJUNTOS_DATOS`: describe cada dataset disponible.
  - `ecommerce`: dataset de productos de comercio electrónico (`ecommerce-products-image-dataset`).
  - `mechanical_tools`: dataset de herramientas mecánicas (`mechanical-tools-dataset`).
  - Cada entrada incluye el identificador de Kaggle (`idKaggle`), tamaño objetivo (`tamanoObjetivo`), categorías a usar y subcarpeta interna si aplica.
- `CONFIGURACION_PREPROCESAMIENTO`: parámetros globales de preprocesamiento (espacio de color, normalización, corrección automática, etc.).
- `DIR_RESULTADOS`: carpeta donde se almacenan los resultados del procesamiento (`backend/procesador_imagenes/results`).
- `PARAMETROS_CARACTERISTICAS`: parámetros específicos para cada extractor (momentos, SIFT, HOG, etc.).

## Script principal (`main.py`)

El archivo `main.py` orquesta el flujo completo:

1. Interpreta los argumentos de línea de comandos (dataset a procesar, algoritmos de características, tamaño de muestra, directorio de salida, etc.).
2. Para cada conjunto de datos seleccionado:
   - Descarga el dataset desde Kaggle si es necesario (`CargadorKaggle`).
   - Carga y filtra las imágenes (`GestorConjuntoDatos`).
   - Preprocesa cada imagen (`PreprocesadorImagen`), aplicando normalización, binarización y correcciones opcionales.
   - Extrae características (momentos, SIFT, HOG, embeddings) mediante `ProcesadorCaracteristicas`.
   - Opcionalmente guarda las imágenes procesadas en disco (original, preprocesada, binaria y, si se activa, el histograma).
   - Calcula estadísticas agregadas y guarda los resultados en CSV, JSON y ficheros de vectores (`GestorResultados`).

### Uso desde línea de comandos

Desde la raíz del proyecto:
```bash
python backend/procesador_imagenes/main.py
```
O con parámetros personalizados, por ejemplo:
```bash
python backend/procesador_imagenes/main.py \
  --conjunto ecommerce \
  --algoritmos momentos sift hog embeddings \
  --muestra 200
```

Parámetros principales:

- `--conjunto`: `ecommerce`, `mechanical_tools` o `todos`.
- `--algoritmos`: lista de extractores a aplicar (`momentos`, `sift`, `hog`, `embeddings`).
- `--muestra`: número de imágenes a procesar (si se omite, procesa todas).
- `--salida`: directorio base de salida (por defecto, `backend/procesador_imagenes/results`).
- `--correccion-automatica`: activa la corrección automática de imágenes (sobrescribe el valor de configuración).
- `--fondo-blanco`: indica si las imágenes tienen fondo blanco (útil para el análisis automático).
- `--guardar-imagenes`: guarda las imágenes procesadas en disco.
- `--graficar-histograma`: genera y guarda histogramas de intensidad para las imágenes procesadas.

## Resultados generados

Para cada conjunto procesado se crea una carpeta dentro de `backend/procesador_imagenes/results`, por ejemplo:

- `backend/procesador_imagenes/results/ecommerce/`
- `backend/procesador_imagenes/results/mechanical_tools/`

Dentro de cada carpeta se generan, entre otros:

- `caracteristicas.csv`: tabla plana con vectores de características por imagen.
- `caracteristicas.json`: estructura detallada con metadatos y resultados por imagen.
- `vectoresCaracteristicas.npz`: fichero comprimido con los vectores de características.
- Carpeta `imagenes_procesadas/` (si se activa el guardado de imágenes), con subcarpetas por clase e imagen.

Estos archivos son los que consumen los scripts de clustering para realizar los experimentos sobre imágenes.
