# Módulo de clustering con restricciones (CluORT)

Este paquete implementa un algoritmo de clustering **online**, con **restricciones de tamaño por cluster**, y varios scripts de experimentación sobre Iris e imágenes procesadas.

## Clase principal: `CluORT`

Archivo: `backend/clustering/clusteringRestricciones.py`

**Idea general**

- Procesa los datos en **streaming**, punto a punto.
- Mantiene como máximo `numClusters` centroides activos.
- Cada cluster tiene una **capacidad máxima** (`tamaniosMaximos[i]`).
- Decide a qué cluster asignar cada punto combinando: distancia al centroide, balance de tamaños y capacidad restante.
- Puede calcular métricas internas (Silueta, Dunn) de forma **aproximada** (rápido) o **exacta** (usando todos los puntos guardados).

**Constructor básico**

```python
from clusteringRestricciones import CluORT

clustering = CluORT(
    numClusters=3,                  # número máximo de clusters
    tamaniosMaximos=[50, 50, 50],   # capacidad máxima por cluster
    metricasAproximadas=True,       # True: usa métricas internas aproximadas
    escalarDatos=True,              # escalado online incremental
    guardarPuntos=False,            # True para obtener métricas exactas al final
)
```

**Flujo online típico**

```python
ultimo_resultado = None
for x, etiqueta_real in zip(X, y):
    ultimo_resultado = clustering.asignarPunto(x, etiquetaReal=etiqueta_real)

# Resumen final (selecciona automáticamente métricas aproximadas o exactas
# según la configuración interna)
resumen = clustering.obtenerResumenFinal()
metricas_internas = resumen["metricasInternas"]  # silueta, dunn
metricas_externas = resumen["metricasExternas"]  # ari, ami, nmi
distribucion = resumen["distribucion"]          # puntos y clases por cluster
```

Métodos clave:
- `asignarPunto(nuevoP, etiquetaReal=None)`: asigna un nuevo punto respetando las capacidades configuradas.
- `obtenerResumenFinal(usarAproximadas=None)`: devuelve métricas internas/externas, distribución y tamaños actuales y máximos.

## Experimentos con IRIS

Script: `backend/clustering/scripts/pruebas_iris.py`

Este script compara **CluORT** con **STREAMKMeans** (de `river`) sobre el dataset Iris y genera:
- Tabla CSV con todas las métricas.
- Tablas en PNG.
- Gráficos de evolución de métricas y distribuciones por cluster.

### Ejecución

Desde la raíz del proyecto se puede ejecutar:

```bash
python backend/clustering/scripts/pruebas_iris.py
```

Los resultados se guardan en la carpeta:

- `backend/clustering/resultados_iris/`

## Experimentos con imágenes

Script: `backend/clustering/scripts/pruebas_imagenes.py`

Este script aplica **CluORT** sobre las características ya extraídas por el módulo de procesamiento de imágenes (`procesador_imagenes/results`), para distintos tipos de descriptores (momentos, SIFT, HOG, embeddings, etc.).

### Ejecución

Antes de ejecutar este script es necesario haber generado previamente las características de imágenes mediante el pipeline de `procesador_imagenes`. A continuación se puede lanzar:

```bash
python backend/clustering/scripts/pruebas_imagenes.py
```

Los resultados se guardan en:

- `backend/clustering/resultados_imagenes/`

## Dependencias

El archivo `backend/requirements.txt` enumera los paquetes necesarios (principalmente `numpy`, `scikit-learn`, `pandas`, `matplotlib`, `dataframe_image` y `river` para STREAMKMeans).

La instalación puede realizarse con:

```bash
pip install -r backend/requirements.txt
```

Con esta información es posible comprender rápidamente la clase `CluORT` y reproducir los experimentos sobre Iris e imágenes.
