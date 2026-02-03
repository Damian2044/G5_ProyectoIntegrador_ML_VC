"""
Extractor de características HOG (Histogram of Oriented Gradients)
"""

import cv2
import numpy as np
from skimage.feature import hog
from typing import Tuple


class ExtractorHog:
    """
    Extrae características HOG usando scikit-image
    """
    
    def __init__(self, orientaciones: int = 9, 
                 pixelesPorCelda: Tuple[int, int] = (8,8),
                 celdasPorBloque: Tuple[int, int] = (2, 2),
                 normalizacionBloque: str = 'L2-Hys'):
        """
        Inicializa el extractor HOG
        
        Args:
            orientaciones: Número de bins de orientación
            pixelesPorCelda: Tamaño de cada celda en píxeles
            celdasPorBloque: Número de celdas por bloque
            normalizacionBloque: Normalización del bloque ('L1', 'L2', 'L1-sqrt', 'L2-Hys')
        """
        self.orientaciones = orientaciones
        self.pixelesPorCelda = pixelesPorCelda
        self.celdasPorBloque = celdasPorBloque
        self.normalizacionBloque = normalizacionBloque
    
    def extraerHog(self, imagen: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrae el vector HOG
        
        Args:
            imagen: Imagen en escala de grises

        Returns:
            vectorHog: Vector de características HOG
            imagenHog: Imagen visualizada de HOG
        """
        if imagen.dtype != np.uint8:
            if imagen.max() <= 1.0:
                imagen = (imagen * 255).astype(np.uint8)
            else:
                imagen = imagen.astype(np.uint8)

        # Calcular HOG
        vectorHog, imagenHog = hog(
            imagen,
            orientations=self.orientaciones,
            pixels_per_cell=self.pixelesPorCelda,
            cells_per_block=self.celdasPorBloque,
            block_norm=self.normalizacionBloque,
            visualize=True
        )
        return vectorHog, imagenHog
    
    def extraerCaracteristicas(self, imagen: np.ndarray) -> np.ndarray:
        """
        Extrae características HOG
        
        Args:
            imagen: Imagen en escala de grises
            
        Returns:
            Vector de características HOG
        """
        #Redimensionar si es necesario
        imagenRedimensionada = cv2.resize(imagen, (64, 128))
        vectorHog, _ = self.extraerHog(imagenRedimensionada)
        return vectorHog.astype(np.float32)
    
    def extraerCaracteristicasMejoradas(self, imagen: np.ndarray) -> np.ndarray:
        """
        Extrae características HOG con estadísticas adicionales
        
        Args:
            imagen: Imagen en escala de grises
            
        Returns:
            Vector de características mejorado
        """
        vectorHog, _ = self.extraerHog(imagen)

        # Agregar estadísticas del vector HOG
        caracteristicas = np.concatenate([
            vectorHog,
            [np.mean(vectorHog)],
            [np.std(vectorHog)],
            [np.min(vectorHog)],
            [np.max(vectorHog)],
            [np.sum(vectorHog)]
        ])
        return caracteristicas.astype(np.float32)
    
    def extraerHogMultiescala(self, imagen: np.ndarray, escalas: list = [0.5, 1.0, 2.0]) -> np.ndarray:
        """
        Extrae HOG a múltiples escalas
        
        Args:
            imagen: Imagen en escala de grises
            escalas: Lista de escalas a usar
            
        Returns:
            Vector de características multi-escala
        """
        todas = []

        for escala in escalas:
            if escala != 1.0:
                h, w = imagen.shape[:2]
                nuevaH, nuevaW = int(h * escala), int(w * escala)
                imagenEscalada = cv2.resize(imagen, (nuevaW, nuevaH))
            else:
                imagenEscalada = imagen

            # Asegurar que la imagen sea al menos del tamaño mínimo
            if imagenEscalada.shape[0] < self.pixelesPorCelda[0] or \
               imagenEscalada.shape[1] < self.pixelesPorCelda[1]:
                continue
            try:
                vector = self.extraerCaracteristicas(imagenEscalada)
                todas.append(vector)
            except:
                # Si hay error con una escala, continuar
                continue
            
        if len(todas) == 0:
            return np.zeros(1)
        
        # Concatenar todos los vectores
        return np.concatenate(todas).astype(np.float32)
