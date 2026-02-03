"""
Extractor de características SIFT (Scale-Invariant Feature Transform)
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


class ExtractorSift:
    """
    Extrae características SIFT de imágenes
    """
    
    def __init__(self, numeroCaracteristicas: int = 500, octavas: int = None, 
                 escalas: int = None, sigma: float = None, umbral: float = None):
        """
        Inicializa el extractor SIFT
        
        Args:
            numeroCaracteristicas: Número máximo de características a detectar
            octavas: Número de octavas (parámetro opcional)
            escalas: Número de escalas por octava (parámetro opcional)
            sigma: Sigma inicial (parámetro opcional)
            umbral: Umbral de contraste (parámetro opcional)
        """
        self.numeroCaracteristicas = numeroCaracteristicas
        self.octavas = octavas
        self.escalas = escalas
        self.sigma = sigma
        self.umbral = umbral
        try:
            self.sift = cv2.SIFT_create(nfeatures=numeroCaracteristicas)
        except AttributeError:
            # Fallback para versiones antiguas
            self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=numeroCaracteristicas)
    
    def detectarYCalcular(self, imagen: np.ndarray) -> Tuple[List, Optional[np.ndarray]]:
        """
        Detecta puntos clave y calcula descriptores SIFT
        
        Args:
            imagen: Imagen en escala de grises
            
        Returns:
            Tuple de (keypoints, descriptors)
        """
        if imagen.dtype != np.uint8:
            if imagen.max() <= 1.0:
                imagen = (imagen * 255).astype(np.uint8)
            else:
                imagen = imagen.astype(np.uint8)

        puntosClave, descriptores = self.sift.detectAndCompute(imagen, None)
        
        return puntosClave, descriptores
    
    def extraerEstadisticasDescriptores(self, descriptores: Optional[np.ndarray]) -> np.ndarray:
        """
        Extrae estadísticas agregadas de los descriptores SIFT
        
        Args:
            descriptores: Array de descriptores (N x 128)
            
        Returns:
            Vector de características agregadas
        """
        if descriptores is None or len(descriptores) == 0:
            # Retornar vector de ceros si no hay descriptores
            return np.zeros(48, dtype=np.float32)
        
        descriptores = descriptores.astype(np.float32)
        # Estadísticas: media, std, min, max por dimensión (128 * 4 = 512 features)
        # Pero usaremos versión comprimida: media, std, min, max de todo + histograma
        
        caracteristicas = []

        # Estadísticas globales
        caracteristicas.extend([
            np.mean(descriptores),
            np.std(descriptores),
            np.min(descriptores),
            np.max(descriptores),
            np.median(descriptores)
        ])

        # Estadísticas por canal/dimensión (reducidas)
        for i in range(0, 128, 16): # Tomar cada 16 dimensiones
            canal = descriptores[:, i:i+16]
            caracteristicas.extend([
                np.mean(canal),
                np.std(canal),
                np.min(canal),
                np.max(canal)
            ])
        
        # Número de descriptores
        caracteristicas.append(len(descriptores))

        # Histograma de magnitudes de descriptores
        magnitudes = np.linalg.norm(descriptores, axis=1)
        hist, _ = np.histogram(magnitudes, bins=10, range=(0, 256))
        caracteristicas.extend(hist)
        return np.array(caracteristicas, dtype=np.float32)
    

    
    def extraerEstadisticasPuntosClave(self, puntosClave: List) -> np.ndarray:
        """
        Extrae estadísticas de los puntos clave
        
        Args:
            puntosClave: Lista de keypoints detectados
            
        Returns:
            Vector de características de keypoints
        """
        caracteristicas = []
        # Número de puntos clave
        numeroPuntos = len(puntosClave)
        caracteristicas.append(numeroPuntos)

        if numeroPuntos == 0:
            caracteristicas.extend([0] * 8)
            return np.array(caracteristicas, dtype=np.float32)
        
        # Número de keypoints de puntos clave
        respuestas = np.array([kp.response for kp in puntosClave])
        tamanos = np.array([kp.size for kp in puntosClave])
        angulos = np.array([kp.angle for kp in puntosClave])

        # Estadísticas de respuesta (fuerza del keypoint)
        caracteristicas.extend([
            np.mean(respuestas),
            np.std(respuestas),
            np.max(respuestas),
            np.min(respuestas)
        ])

        # Estadísticas de tamaño
        caracteristicas.extend([
            np.mean(tamanos),
            np.std(tamanos),
            np.max(tamanos),
            np.min(tamanos)
        ])
        return np.array(caracteristicas, dtype=np.float32)
    
    def extraerPromedio128(self, imagen: np.ndarray) -> np.ndarray:
        """
        Extrae el vector promedio de 128 dimensiones de los descriptores
        
        Args:
            imagen: Imagen en escala de grises
            
        Returns:
            Vector de 128 dimensiones
        """
        _, descriptores = self.detectarYCalcular(imagen)
        
        if descriptores is None or len(descriptores) == 0:
            return np.zeros(128, dtype=np.float32)
        
        return np.mean(descriptores, axis=0).astype(np.float32)


    def extraerCaracteristicas(self, imagen: np.ndarray) -> np.ndarray:
        """
        Extrae características SIFT completas
        
        Args:
            imagen: Imagen en escala de grises
            
        Returns:
            Vector de características agregadas
        """
        puntosClave, descriptores = self.detectarYCalcular(imagen)

        # Estadísticas de descriptores
        estadDesc = self.extraerEstadisticasDescriptores(descriptores)
        
        # Estadísticas de puntos clave
        estadPuntos = self.extraerEstadisticasPuntosClave(puntosClave)

        # Promedio de 128 dimensiones
        promedio128 = self.extraerPromedio128(imagen)
        
        # Combinar ambas
        return np.concatenate([estadDesc, estadPuntos])
