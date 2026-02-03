"""Script de prueba para leer una imagen y mostrar su embedding.

Ejecutar desde la raíz del proyecto, por ejemplo:

    python backend/procesador_imagenes/src/extraccionCaracteristicas/pruebaEmbeddings.py
"""

from pathlib import Path

import cv2
import numpy as np

from extratorEmbeddingsRedProfunda import ExtractorEmbeddingsRedProfunda


def main() -> None:
    # Ruta de ejemplo: un jean procesado
    # Construir la ruta a partir de la ubicación de este archivo
    rutaBaseProcesador = Path(__file__).resolve().parents[2]
    rutaImagen = rutaBaseProcesador / "results" / "ecommerce" / "imagenes_procesadas" / "jeans" / "img1" / "original_1.jpg"

    print(f"Leyendo imagen desde: {rutaImagen}")
    imagen = cv2.imread(str(rutaImagen), cv2.IMREAD_COLOR)
    if imagen is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {rutaImagen}")

    extractor = ExtractorEmbeddingsRedProfunda()
    embedding = extractor.extraerEmbeddings(imagen)

    print("Embedding generado (forma):", embedding.shape)
    print("Embedding (primeros 10 valores):", embedding[:10])


if __name__ == "__main__":
    main()
