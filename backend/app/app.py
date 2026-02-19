from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import numpy as np
import random
import uuid
import os
from sklearn.decomposition import IncrementalPCA

from clustering.clusteringRestricciones import CluORT
from procesador_imagenes.src.preprocesamiento.preprocesadorImagen import PreprocesadorImagen
from procesador_imagenes.src.extraccionCaracteristicas.extractorMomentos import ExtractorMomentos
from procesador_imagenes.src.extraccionCaracteristicas.extractorSift import ExtractorSift
from procesador_imagenes.src.extraccionCaracteristicas.extractorHog import ExtractorHog
from procesador_imagenes.src.extraccionCaracteristicas.extratorEmbeddingsRedProfunda import (
    ExtractorEmbeddingsRedProfunda,
)
from procesador_imagenes.config import PARAMETROS_CARACTERISTICAS, CONFIGURACION_PREPROCESAMIENTO


##########################
# Configuración global
##########################

semillaGlobal = 42
TAMANIO_OBJETIVO = (224, 224)

# Controlar si se calcula/proyecta PCA 2D para visualización.
# Variable de entorno HABILITAR_PCA: "1", "true", "si", "yes" -> activa.
HABILITAR_PCA = os.getenv("HABILITAR_PCA", "true").strip().lower() in {"1", "true", "si", "yes"}


##########################
# Modelos Pydantic
##########################


class EstadoSesionClustering(BaseModel):
    id: str
    k: int
    tamanosMaximos: List[int]
    metodo: str


class ParametrosInicioClustering(BaseModel):
    k: int
    tamanosMaximos: List[int]
    metodo: str
    semilla: Optional[int] = None


class ParametrosActualizarTamanios(BaseModel):
    sesionId: str
    tamanosNuevos: List[int]


##########################
# Gestión de sesiones de clustering
##########################


sesionesClustering: Dict[str, Dict] = {}


def crearSesionClustering(k: int, tamanosMaximos: List[int], metodo: str) -> EstadoSesionClustering:
    if len(tamanosMaximos) != k:
        raise ValueError("La longitud de tamanosMaximos debe coincidir con k")

    sesion_id = uuid.uuid4().hex
    cluort = CluORT(
        numClusters=k,
        tamaniosMaximos=tamanosMaximos,
        metricasAproximadas=True,
        escalarDatos=True,
        guardarPuntos=False,
    )
    preprocesador = PreprocesadorImagen()

    sesionesClustering[sesion_id] = {
        "estado": EstadoSesionClustering(id=sesion_id, k=k, tamanosMaximos=tamanosMaximos, metodo=metodo),
        "cluort": cluort,
        "preprocesador": preprocesador,
    }

    return sesionesClustering[sesion_id]["estado"]


def obtenerSesion(sesionId: str) -> Dict:
    sesion = sesionesClustering.get(sesionId)
    if not sesion:
        raise HTTPException(status_code=404, detail="Sesión de clustering no encontrada")
    return sesion


##########################
# Utilidades varias
##########################


def sanitizarParaJson(valor: Any) -> Any:
    """Convierte recursivamente objetos NumPy a tipos nativos Python para JSON."""
    if isinstance(valor, np.generic):
        return valor.item()
    if isinstance(valor, np.ndarray):
        return valor.tolist()
    if isinstance(valor, dict):
        return {k: sanitizarParaJson(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple, set)):
        return [sanitizarParaJson(v) for v in valor]
    return valor


##########################
# Extractores de características (globales)
##########################

# Inicializar extractores globales al cargar la app usando los mismos
# parámetros que ProcesadorCaracteristicas
params_momentos = PARAMETROS_CARACTERISTICAS.get("momentos", {})
params_sift = PARAMETROS_CARACTERISTICAS.get("sift", {})
params_hog = PARAMETROS_CARACTERISTICAS.get("hog", {})

extractorMomentosGlobal = ExtractorMomentos(
    ordenMaximo=params_momentos.get("ordenMaximo", 3),
    momentosHu=params_momentos.get("momentosHu", 7),
)
extractorSiftGlobal = ExtractorSift(
    numeroCaracteristicas=params_sift.get("numeroCaracteristicas", 500),
    octavas=params_sift.get("octavas"),
    escalas=params_sift.get("escalas"),
    sigma=params_sift.get("sigma"),
    umbral=params_sift.get("umbral"),
)
extractorHogGlobal = ExtractorHog(
    orientaciones=params_hog.get("orientaciones", 9),
    pixelesPorCelda=tuple(params_hog.get("pixelesPorCelda", (16, 16))),
    celdasPorBloque=tuple(params_hog.get("celdasPorBloque", (2, 2))),
    normalizacionBloque=params_hog.get("normalizacionBloque", "L2-Hys"),
)
extractorEmbeddingsGlobal = ExtractorEmbeddingsRedProfunda()


##########################
# Vectorización de imágenes
##########################


def vectorizarImagen(
    preprocesador: PreprocesadorImagen,
    contenido: bytes,
    metodoExtraccion: Optional[str] = None,
):
    """Convierte una imagen en vector de características usando el método indicado.

    - Decodifica bytes a imagen BGR (formato OpenCV).
    - Redimensiona con PreprocesadorImagen.
    - Según `metodoExtraccion`, aplica momentos, SIFT, HOG o embeddings.
    - Por defecto, si no se configura nada, usa gris + flatten.
    """
    import cv2

    arr = np.frombuffer(contenido, np.uint8)
    imgBGR = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR (formato OpenCV)
    if imgBGR is None:
        raise ValueError("No se pudo decodificar la imagen")

    metodo = (metodoExtraccion or "").strip().lower()

    resultadosPreprocesamiento = None
    if metodo in ["momentos", "sift", "hog"]:
        resultadosPreprocesamiento = preprocesador.preprocesarImagen(
            imgBGR,
            espacioColor=CONFIGURACION_PREPROCESAMIENTO.get("espacioColor", "grayscale"),
            normalizar=CONFIGURACION_PREPROCESAMIENTO.get("normalizar", True),
            binarizar=True,
            correccionAutomatica=CONFIGURACION_PREPROCESAMIENTO.get("correccionAutomatica", True),
            fondoBlanco=CONFIGURACION_PREPROCESAMIENTO.get("fondoBlanco", False),
        )

    if metodo == "momentos":
        if resultadosPreprocesamiento is None:
            raise ValueError("No se obtuvo resultado de preprocesamiento para 'momentos'")
        imagen_binarizada = resultadosPreprocesamiento.get("binarizada")
        if imagen_binarizada is None:
            raise ValueError(
                "La imagen binarizada es None después de preprocesarImagen para 'momentos'"
            )
        return extractorMomentosGlobal.extraer24Momentos(imagen_binarizada)

    if metodo == "sift":
        imagenGray = resultadosPreprocesamiento.get("suavizada")
        if imagenGray is None:
            raise ValueError("La imagen gris es None luego de preprocesarImagen para 'sift'")
        return extractorSiftGlobal.extraerCaracteristicas(imagenGray)

    if metodo == "hog":
        imagenGray = resultadosPreprocesamiento.get("suavizada")
        if imagenGray is None:
            raise ValueError("La imagen gris es None luego de preprocesarImagen para 'hog'")
        return extractorHogGlobal.extraerCaracteristicas(imagenGray)

    if metodo == "embeddings":
        return extractorEmbeddingsGlobal.extraerEmbeddings(imgBGR)

    # Comportamiento clásico gris + flatten si no hay método
    import cv2

    imgGray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
    imgGray = cv2.resize(
        imgGray,
        (TAMANIO_OBJETIVO[1], TAMANIO_OBJETIVO[0]),
        interpolation=cv2.INTER_LANCZOS4,
    )
    vector = imgGray.flatten().astype(np.float32)
    return vector


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


##########################
# Endpoints FastAPI
##########################


@app.post("/clustering/iniciar")
async def iniciarClustering(parametros: ParametrosInicioClustering):
    """Inicializa una sesión de clustering online con CluORT.

    k: número de clusters.
    tamanosMaximos: lista con tamaños máximos por cluster.
    metodo: nombre del método de extracción (momentos, sift, hog, embeddings...).
    semilla: semilla opcional para reproducibilidad (si no se indica, usa semillaGlobal).
    """

    if parametros.semilla is not None:
        random.seed(parametros.semilla)
        np.random.seed(parametros.semilla)
    else:
        random.seed(semillaGlobal)
        np.random.seed(semillaGlobal)

    try:
        estado = crearSesionClustering(parametros.k, list(parametros.tamanosMaximos), parametros.metodo)
        return {"sesionId": estado.id, "k": estado.k, "tamanosMaximos": estado.tamanosMaximos, "metodo": estado.metodo}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/clustering/actualizar-tamanios")
async def actualizarTamaniosClustering(parametros: ParametrosActualizarTamanios):
    """Actualiza los tamaños máximos de una sesión de clustering existente."""

    sesion = obtenerSesion(parametros.sesionId)
    cluort = sesion["cluort"]

    resultado = cluort.actualizarTamaniosMaximos(parametros.tamanosNuevos)
    if resultado.get("estado") != "ok":
        raise HTTPException(status_code=400, detail=resultado.get("detalle", "Error al actualizar tamaños"))

    # Actualizar también el estado almacenado de la sesión
    sesion["estado"].tamanosMaximos = cluort.tamaniosMaximos

    return {
        "estado": "ok",
        "tamaniosMaximos": cluort.tamaniosMaximos,
        "tamaniosActuales": cluort.tamaniosActuales,
    }


@app.post("/clustering/agregarPunto")
async def agregarPuntoClustering(
    sesionId: str = Form(...),
    idFront: str = Form(...),
    etiqueta: Optional[str] = Form(None),
    imagen: UploadFile = File(...),
):
    """Agrega un nuevo punto (imagen) a una sesión de clustering existente.

    Devuelve el cluster asignado (si hay espacio) más centroides y métricas reales.
    """

    sesion = obtenerSesion(sesionId)
    cluort = sesion["cluort"]
    preprocesador = sesion["preprocesador"]
    metodoExtraccion = getattr(sesion["estado"], "metodo", None)

    contenido = await imagen.read()
    try:
        vector = vectorizarImagen(preprocesador, contenido, metodoExtraccion)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar imagen: {e}")

    resultado = cluort.asignarPunto(vector, etiquetaReal=etiqueta)
    if resultado.get("estado") != "ok":
        raise HTTPException(status_code=500, detail=resultado.get("detalle", "Error interno en CluORT"))

    etiqueta_asignada = resultado.get("etiquetaAsignada")
    aceptado = etiqueta_asignada is not None

    proyeccionPCA = None

    # --- PCA INCREMENTAL 2D (proyección para visualización, opcional) ---
    # Solo se calcula si HABILITAR_PCA está activo; en caso contrario,
    # el clustering funciona igual pero no se generan coordenadas 2D.
    if HABILITAR_PCA:
        # Se inicializa un modelo IncrementalPCA por sesión.
        pca_modelo: IncrementalPCA = sesion.get("pcaModelo")
        if pca_modelo is None:
            pca_modelo = IncrementalPCA(n_components=2)
            sesion["pcaModelo"] = pca_modelo

        # Buffer para las primeras muestras, para poder ajustar el modelo.
        # Solo usamos vectores de puntos ACEPTADOS en el clustering.
        buffer_pca: List[np.ndarray] = sesion.get("pcaBuffer", [])
        if aceptado:
            buffer_pca.append(vector.astype(np.float64))
            sesion["pcaBuffer"] = buffer_pca

        # Ajuste incremental: primero esperamos al menos 2 muestras aceptadas
        # para inicializar bien el PCA; luego lo vamos actualizando solo con
        # nuevos vectores aceptados.
        if not hasattr(pca_modelo, "components_") and len(buffer_pca) >= 2:
            X_init = np.vstack(buffer_pca)
            try:
                pca_modelo.partial_fit(X_init)
            except Exception:
                pass
        elif hasattr(pca_modelo, "components_") and aceptado:
            try:
                pca_modelo.partial_fit(vector.reshape(1, -1))
            except Exception:
                pass

        # Calcular proyección 2D solo si el PCA ya está entrenado
        if hasattr(pca_modelo, "components_"):
            pca_puntos: Dict[str, Dict[str, Any]] = sesion.get("pcaPuntos", {})

            # Solo proyectar y guardar puntos que fueron aceptados en el clustering
            if aceptado:
                try:
                    coord_punto = pca_modelo.transform(vector.reshape(1, -1))[0]
                    pca_puntos[idFront] = {
                        "idFront": idFront,
                        "x": float(coord_punto[0]),
                        "y": float(coord_punto[1]),
                        "cluster": etiqueta_asignada,
                    }
                except Exception:
                    # Si falla la proyección del punto, se omite
                    pass

            sesion["pcaPuntos"] = pca_puntos

            # Proyección de centroides actuales
            pca_centroides = []
            try:
                for idx, c in enumerate(cluort.centroidesAct):
                    if isinstance(c, np.ndarray):
                        try:
                            coord_centroide = pca_modelo.transform(c.reshape(1, -1))[0]
                            pca_centroides.append(
                                {
                                    "cluster": idx,
                                    "x": float(coord_centroide[0]),
                                    "y": float(coord_centroide[1]),
                                }
                            )
                        except Exception:
                            pca_centroides.append({"cluster": idx, "x": None, "y": None})
                    else:
                        pca_centroides.append({"cluster": idx, "x": None, "y": None})
            except Exception:
                pca_centroides = []

            sesion["pcaCentroides"] = pca_centroides

            proyeccionPCA = {
                "puntos": list(pca_puntos.values()),
                "centroides": pca_centroides,
            }

    # Centroides reales (vectores completos) para resumen
    centroides_reales = []
    try:
        for c in cluort.centroidesAct:
            if isinstance(c, np.ndarray):
                centroides_reales.append(c.tolist())
            else:
                centroides_reales.append(None)
    except Exception:
        centroides_reales = []

    respuesta = {
        "estado": "ok",
        "idFront": idFront,
        "cluster": etiqueta_asignada,
        "aceptado": aceptado,
        "tamaniosActuales": cluort.tamaniosActuales,
        "tamaniosMaximos": cluort.tamaniosMaximos,
        "centroides": centroides_reales,
        "metricasInternas": resultado.get("metricasInternas"),
        "metricasExternas": resultado.get("metricasExternas"),
        "distribucion": resultado.get("distribucion"),
    }

    if proyeccionPCA is not None:
        respuesta["proyeccionPCA"] = proyeccionPCA

    # Asegurar que la respuesta sea JSON serializable (sin tipos NumPy)
    return sanitizarParaJson(respuesta)

