"""
Extractor de Momentos (Hu, Zernike)
"""

import cv2
import numpy as np
import mahotas

class ExtractorMomentos:
    """
    Extrae características basadas en momentos (Momentos de Hu, Zernike, etc.)
    """
    
    def __init__(self, ordenMaximo: int = 3, momentosHu: int = 7):
        """
        Inicializa el extractor de momentos
        
        Args:
            ordenMaximo: Orden máximo para momentos de Zernike
            momentosHu: Número de momentos invariantes de Hu
        """
        self.ordenMaximo = ordenMaximo
        self.momentosHu = momentosHu
    
    def extraerMomentosHu(self, imagen: np.ndarray) -> np.ndarray:
        """
        Extrae los 7 momentos invariantes de Hu
        
        Args:
            imagen: Imagen en escala de grises (uint8)
            
        Returns:
            Array de 7 momentos de Hu
        """
        # Calcular momentos
        momentos = cv2.moments(imagen)

        # Calcular momentos invariantes de Hu
        hu = cv2.HuMoments(momentos)

        # Convertir a log scale para mejor manejo de rangos
        for i in range(7):
            hu[i] = -np.sign(hu[i]) * np.log10(np.abs(hu[i]) + 1e-10)

        return hu.flatten()
    
    def extraerMomentosCrudos(self, imagen: np.ndarray) -> dict:
        """
        Extrae momentos crudos (primeros órdenes)
        
        Args:
            imagen: Imagen en escala de grises
            
        Returns:
            Diccionario con momentos de orden 0 a 3
        """
        m = cv2.moments(imagen)

        return {
            'm00': m['m00'], # Área
            'm10': m['m10'], # Centro X
            'm01': m['m01'], # Centro Y
            'm20': m['m20'], # Momento segundo orden X
            'm02': m['m02'], # Momento segundo orden Y
            'm11': m['m11'], # Momento mixto
            'm30': m['m30'],
            'm03': m['m03'],
            'm21': m['m21'],
            'm12': m['m12']
        }
    
    def extraerMomentosCentrales(self, imagen: np.ndarray) -> dict:
        """
        Extrae momentos centrales (centralizados en el centroide)
        
        Args:
            imagen: Imagen en escala de grises
            
        Returns:
            Array de momentos centrales
        """
        m = cv2.moments(imagen)

        # Momentos centrales
        return {
                'mu20': m['mu20'],
                'mu02': m['mu02'],
                'mu11': m['mu11'],
                'mu30': m['mu30'],
                'mu03': m['mu03'],
                'mu21': m['mu21'],
                'mu12': m['mu12']
        }
    
    def extraerMomentosNormalizados(self, imagen: np.ndarray) -> dict:
        """
        Extrae momentos normalizados (invariantes a escala)
        
        Calcula momentos normalizados del segundo y tercer orden para invariancia a escala.
        Los momentos normalizados son útiles para comparar formas de diferentes tamaños.
        
        Args:
            imagen: Imagen en escala de grises
            
        Returns:
            Array de 7 momentos normalizados [nu20, nu02, nu11, nu30, nu03, nu21, nu12]
        """
        m = cv2.moments(imagen)
        
            # Si el área es cero, retornar ceros
        if m['m00'] <= 0:
            return {key: 0.0 for key in ['nu20','nu02','nu11','nu30','nu03','nu21','nu12']}
        
        return {
            'nu20': m['nu20'],
            'nu02': m['nu02'],
            'nu11': m['nu11'],
            'nu30': m['nu30'],
            'nu03': m['nu03'],
            'nu21': m['nu21'],
            'nu12': m['nu12']
        }
        
    def calcularMomentosZernike(self, imagen: np.ndarray, radio: int = 10, grado: int = 8) -> np.ndarray:
        """
        Calcula los momentos de Zernike de una imagen usando Mahotas.
        La imagen se binariza automáticamente si no lo está.
        """

        # Asegurar uint8
        if imagen.dtype != np.uint8:
            imagen = (imagen * 255).astype(np.uint8) if imagen.max() <= 1 else imagen.astype(np.uint8)

        # Binarización robusta (sirve para 0/1, 0/255, escala de grises)
        imagenBinaria = (imagen > 0).astype(np.uint8)

        # Calcular momentos Zernike
        zernike = mahotas.features.zernike_moments(
            imagenBinaria,
            radius=radio,
            degree=grado
        )

        return zernike


    
    def extraerTodosMomentos(self, imagen: np.ndarray) -> dict:
        """
        Extrae todos los tipos de momentos disponibles
        
        Retorna 10 crudos + 7 centrales + 7 normalizados + 7 Hu + n Zernike.
        
        Args:
            imagen: Imagen en escala de grises (uint8)
            
        Returns:
            Diccionario con todos los momentos extraídos (38+ dimensiones totales)
        """
        # Asegurar que la imagen sea uint8
        if imagen.dtype != np.uint8:
            imagen = (imagen * 255).astype(np.uint8) if imagen.max() <= 1 else imagen.astype(np.uint8)
        
        crudos = self.extraerMomentosCrudos(imagen)
        centrales = self.extraerMomentosCentrales(imagen)
        normalizados = self.extraerMomentosNormalizados(imagen)
        hu = self.extraerMomentosHu(imagen)
        zernike = self.calcularMomentosZernike(imagen, self.ordenMaximo)
        
        # Vector de características: 10 crudos + 7 centrales + 7 normalizados + 7 Hu + Zernike
        vectorCaracteristicas = np.concatenate([
            list(crudos.values()),  # 10 momentos crudos
            list(centrales.values()),              # 7 momentos centrales
            list(normalizados.values()),           # 7 momentos normalizados
            hu,                     # 7 momentos de Hu
            zernike                 # ordenMaximo+1 momentos de Zernike
        ])
        
        return {
            'momentosCrudos': crudos,
            'momentosCentrales': centrales,
            'momentosNormalizados': normalizados,
            'momentosHu': hu,
            'momentosZernike': zernike,
            'vectorCaracteristicas': vectorCaracteristicas
        }


