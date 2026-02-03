from typing import List, Dict, Union
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score



class CluORT:  # Clustering online con restricciones de tamaño

    def __init__(
        self,
        numClusters: int = 3,
        tamaniosMaximos: List[int] = [50],
        metricasAproximadas: bool = True,
        escalarDatos: bool = True,
        guardarPuntos: bool = False,
    ):
        """Inicializa el clustering online con restricciones de tamaño.

        Args:
            numClusters: Número de clusters.
            tamaniosMaximos: Lista con tamaños máximos por cluster.
            metricasAproximadas:
                - True  -> métricas internas aproximadas (centroides/radios).
                - False -> métricas internas exactas usando todos los puntos guardados.
            escalarDatos: Si True, aplica escalado online incremental.
            guardarPuntos:
                - Si metricasAproximadas es False, SIEMPRE se guardan puntos (ignora este flag).
                - Si metricasAproximadas es True, este flag decide si se guardan puntos
                  para poder calcular métricas exactas al final.
        """
        self.numClusters: int = numClusters  # Número de clusters
        self.metricasAproximadas: bool = metricasAproximadas  # Modo de cálculo de métricas

        # Decidir si se guardan puntos por cluster
        # - Si metricasAproximadas=False -> siempre True (necesario para métricas exactas).
        # - Si metricasAproximadas=True  -> depende del parámetro guardarPuntos.
        self.guardarPuntos: bool = (not metricasAproximadas) or guardarPuntos

        # Tamaños máximos permitidos por cluster
        if tamaniosMaximos is None:
            self.tamaniosMaximos: List[int] = [50] * numClusters
        else:
            self.tamaniosMaximos: List[int] = tamaniosMaximos

        self.tamaniosActuales: List[int] = [0] * numClusters  # Tamaños actuales de cada cluster
        self.etiquetasAsignadas: List[int] = []  # Etiquetas de cluster asignadas ej: [0,1,0,2,1,...]
        self.etiquetasReales: List[Union[int, str, None]] = []  # Etiquetas reales de los datos (pueden ser int, str o None)
        self.centroidesAct: List[np.ndarray] = []  # Centroides actuales de los clusters
        self.varianzas: List[float] = []  # Varianzas de cada cluster
        self.maxScoreHistorico: float = 0.0  # Establece el score máximo histórico
        
        # Almacenamiento de puntos por cluster (controlado por guardarPuntos)
        self.puntosCluster: List[List[np.ndarray]] = [[] for _ in range(numClusters)]  # Puntos por cluster
        
        # Para cálculo de métricas aproximadas (siempre se mantienen)
        self.radiosMaximos: List[float] = [0.0] * numClusters  # Radio máximo de cada cluster (distancia máxima al centroide)
        
        # Para escalado online incremental PERMANENTE
        self.habilitarEscalado: bool = escalarDatos  # Si True, escala los puntos por defecto (True por defecto)
        self.nPuntosVistos: int = 0  # Contador de puntos procesados
        self.mediaAcumulada: np.ndarray = None  # Media corriente de todas las características (inicializada en primer punto)
        self.varianzaAcumulada: np.ndarray = None  # Varianza corriente de todas las características (inicializada en primer punto)

    def actualizarTamaniosActuales(self, tamaniosNuevos: List[int]) -> dict:
        """
        Actualiza los tamaños máximos de los clusters
        Args:
            tamaniosNuevos: Lista con los nuevos tamaños máximos para cada cluster
        Returns:
            Diccionario con el estado de la operación"""

        try:
            if len(tamaniosNuevos) != self.numClusters:
                return {"estado": "error",
                     "detalle": "La longitud de tamaniosNuevos no coincide con el número de clusters.",
                     "datos": None}
            for i in range(self.numClusters):
                # Actualizar tamaños máximos
                if tamaniosNuevos[i] >= self.tamaniosActuales[i]:
                    self.tamaniosMaximos[i] = tamaniosNuevos[i]
                else:
                    return {"estado": "error",
                           "detalle": f"El nuevo tamaño para cluster {i} ({tamaniosNuevos[i]}) no puede ser menor que el actual ({self.tamaniosActuales[i]}).",
                           "datos": None}

            return {"estado": "ok", "detalle": "Tamaños máximos actualizados correctamente.", "datos": self.tamaniosMaximos}
        except Exception as e:
            return {"estado": "error", "detalle": str(e), "datos": None}
        

    #------------------------------------------------------------------------------
    # Funciones Auxiliares
    #------------------------------------------------------------------------------
    def _escalarPuntoOnline(self, punto: np.ndarray) -> np.ndarray:
        """
        Escala un punto usando media y varianza acumuladas (algoritmo de Welford)
        """
        if not self.habilitarEscalado:
            return punto
        
        # Primer punto: inicializar estadísticas y centrar en origen
        if self.mediaAcumulada is None:
            self.mediaAcumulada = punto.copy()
            self.varianzaAcumulada = np.zeros_like(punto)
            self.nPuntosVistos = 1
            return np.zeros_like(punto)  # Primer punto centrado en origen
        
        # Puntos siguientes: actualizar con Welford
        self.nPuntosVistos += 1
        nPuntos = self.nPuntosVistos
        mediaAnterior = self.mediaAcumulada.copy()
        self.mediaAcumulada = self.mediaAcumulada + (punto - self.mediaAcumulada) / nPuntos
        self.varianzaAcumulada = self.varianzaAcumulada + (punto - mediaAnterior) * (punto - self.mediaAcumulada)
        
        desviacionEstandar = np.sqrt(self.varianzaAcumulada / max(nPuntos - 1, 1))
        desviacionEstandar = np.where(desviacionEstandar < 1e-8, 1.0, desviacionEstandar)
        
        puntoEscalado = (punto - self.mediaAcumulada) / desviacionEstandar
        
        # Validar que no haya NaN o Inf
        if np.any(np.isnan(puntoEscalado)) or np.any(np.isinf(puntoEscalado)):
            return punto  # Si hay problemas, retornar punto sin escalar
        
        return puntoEscalado
    
    
    def _calcularScore(self, punto: np.ndarray) -> List:
        """
        Calcula el score de asignación para cada cluster considerando:
        - Distancia al centroide
        - Balance de tamaños
        - Capacidad disponible
        """
        distancias = pairwise_distances([punto], self.centroidesAct)[0]
        scoresFinales = []
        nPuntosAsignados = sum(self.tamaniosActuales)  # Cantidad de puntos asignados
        nPuntosMaximos = sum(self.tamaniosMaximos)  # Cantidad de puntos máximos esperados

        for i in range(len(self.centroidesAct)):
            # Si el cluster está lleno, no asignarle score
            if self.tamaniosActuales[i] >= self.tamaniosMaximos[i]:
                scoresFinales.append(None)
                continue
            
            # Calcular score para clusters con espacio disponible
            proporcionTeorica = self.tamaniosMaximos[i] / max(1, nPuntosMaximos)
            esperadoPuntos = max(1.0, proporcionTeorica * nPuntosAsignados)
            balance = (self.tamaniosActuales[i] + 1) / esperadoPuntos
            espacioLibre = self.tamaniosMaximos[i] - self.tamaniosActuales[i]
            capacidad = 1 / espacioLibre
            
            # Score final = distancia × balance × capacidad
            scoresFinales.append(distancias[i] * balance * capacidad)
        return scoresFinales
            
    def _crearCentroide(self, punto: np.ndarray) -> int:
        """Crea un nuevo centroide y devuelve su índice"""
        idx = len(self.centroidesAct)
        self.centroidesAct.append(punto.copy())
        self.tamaniosActuales[idx] = 1
        self.etiquetasAsignadas.append(idx)
        self.varianzas.append(0.0)
        self.radiosMaximos[idx] = 0.0
        
        # Almacenar puntos según configuración (para posibles métricas exactas)
        if self.guardarPuntos:
            self.puntosCluster[idx].append(punto.copy())
        
        return idx
    

    def _calcularFactorAprendizaje(self) -> float:
        """Calcula la tasa de aprendizaje adaptativa basada en entropía"""
        varianzasArray = np.array(self.varianzas, dtype=float)
        if varianzasArray.sum() == 0 or self.numClusters <= 1:
            return 1.0  # Si no hay puntos o solo 1 cluster, movemos rápido

        varianzasNorm = varianzasArray / varianzasArray.sum()
        entropia = -np.sum([vi * np.log2(vi) if vi > 0 else 0 for vi in varianzasNorm])
        logClusters = np.log2(self.numClusters)
        
        if logClusters == 0:
            return 1.0
        
        factorApren = (logClusters - entropia) / logClusters
        # Asegurar que el factor esté en rango válido [0, 1]
        return np.clip(factorApren, 0.0, 1.0)
    
    def _actualizarCluster(self, idx: int, punto: np.ndarray):
        """Actualiza un cluster existente con un nuevo punto"""
        diferencia = punto - self.centroidesAct[idx]
        distanciaAlCentroide = np.linalg.norm(diferencia)
        
        self.tamaniosActuales[idx] += 1
        nActual = self.tamaniosActuales[idx]
        
        # Actualizar varianza incremental
        self.varianzas[idx] = ((nActual - 1) * self.varianzas[idx] + np.sum(diferencia ** 2)) / nActual
        
        # Actualizar radio máximo (para Dunn aproximado)
        if distanciaAlCentroide > self.radiosMaximos[idx]:
            self.radiosMaximos[idx] = distanciaAlCentroide
        
        # Actualizar centroide con factor adaptativo
        factorAprendizaje = self._calcularFactorAprendizaje()
        nuevoCentroide = self.centroidesAct[idx] + factorAprendizaje * diferencia
        
        # Validar que el nuevo centroide no tenga NaN o Inf
        if not np.any(np.isnan(nuevoCentroide)) and not np.any(np.isinf(nuevoCentroide)):
            self.centroidesAct[idx] = nuevoCentroide
        
        self.etiquetasAsignadas.append(idx)
        
        # Almacenar puntos según configuración (para posibles métricas exactas)
        if self.guardarPuntos:
            self.puntosCluster[idx].append(punto.copy())
    
    def _calcularSiluetaAproximada(self) -> float:
        """
        Calcula silueta aproximada basada en centroides
        Para cada punto asignado, usa la distancia al centroide propio vs otros centroides
        """
        if len(self.centroidesAct) < 2 or len(self.etiquetasAsignadas) < 2:
            return 0.0
        
        siluetaTotal = 0.0
        contPuntos = 0
        
        # Para cada cluster, estimar silueta promedio
        for idxCluster in range(len(self.centroidesAct)):
            if self.tamaniosActuales[idxCluster] == 0:
                continue
            
            # a(i) ≈ desviación estándar del cluster (aproximación de distancia intra-cluster)
            aPromedio = np.sqrt(self.varianzas[idxCluster])
            
            # b(i) = mínima distancia a otros centroides
            distanciasACentroides = []
            for idxOtro in range(len(self.centroidesAct)):
                if idxOtro != idxCluster and self.tamaniosActuales[idxOtro] > 0:
                    dist = np.linalg.norm(self.centroidesAct[idxCluster] - self.centroidesAct[idxOtro])
                    distanciasACentroides.append(dist)
            
            if not distanciasACentroides:
                continue
            
            bPromedio = min(distanciasACentroides)
            
            # Silueta para este cluster
            if max(aPromedio, bPromedio) > 0:
                siluetaCluster = (bPromedio - aPromedio) / max(aPromedio, bPromedio)
                siluetaTotal += siluetaCluster * self.tamaniosActuales[idxCluster]
                contPuntos += self.tamaniosActuales[idxCluster]
        
        return siluetaTotal / contPuntos if contPuntos > 0 else 0.0
    
    def _calcularSiluetaExacta(self, puntos: np.ndarray) -> float:
        """Calcula silueta exacta usando sklearn con todos los puntos almacenados"""
        if len(np.unique(self.etiquetasAsignadas)) < 2 or len(self.etiquetasAsignadas) < 3:
            return 0.0
        
        try:
            return silhouette_score(puntos, self.etiquetasAsignadas)
        except:
            return 0.0
    
    def _calcularDunnAproximado(self) -> float:
        """
        Calcula índice de Dunn aproximado usando centroides y radios máximos
        Dunn = separación_mínima / diámetro_máximo
        """
        if len(self.centroidesAct) < 2:
            return 0.0
        
        # Calcular separación mínima entre centroides
        separacionMin = float('inf')
        for i in range(len(self.centroidesAct)):
            if self.tamaniosActuales[i] == 0:
                continue
            for j in range(i + 1, len(self.centroidesAct)):
                if self.tamaniosActuales[j] == 0:
                    continue
                dist = np.linalg.norm(self.centroidesAct[i] - self.centroidesAct[j])
                if dist < separacionMin:
                    separacionMin = dist
        
        # Calcular diámetro máximo aproximado (2 × radio máximo)
        diametroMax = 0.0
        for i in range(len(self.centroidesAct)):
            if self.tamaniosActuales[i] > 0:
                diametroAprox = 2 * self.radiosMaximos[i]
                if diametroAprox > diametroMax:
                    diametroMax = diametroAprox
        
        if diametroMax == 0.0:
            return 0.0
        
        return separacionMin / diametroMax
    
    def _calcularDunnExacto(self) -> float:
        """
        Calcula índice de Dunn exacto usando todos los puntos almacenados
        Dunn = min_separación / max_diámetro
        """
        if len(self.centroidesAct) < 2:
            return 0.0
        
        # Calcular separación mínima EXACTA entre puntos de clusters diferentes
        separacionMin = float('inf')
        for i in range(len(self.centroidesAct)):
            if len(self.puntosCluster[i]) == 0:
                continue
            for j in range(i + 1, len(self.centroidesAct)):
                if len(self.puntosCluster[j]) == 0:
                    continue
                # Comparar todos los puntos de cluster i con todos los de cluster j
                for punto_i in self.puntosCluster[i]:
                    for punto_j in self.puntosCluster[j]:
                        dist = np.linalg.norm(punto_i - punto_j)
                        if dist < separacionMin:
                            separacionMin = dist
        
        # Calcular diámetro máximo EXACTO (mayor distancia dentro de cada cluster)
        diametroMax = 0.0
        for i in range(len(self.centroidesAct)):
            if len(self.puntosCluster[i]) < 2:
                continue
            puntos_cluster = np.array(self.puntosCluster[i])
            # Calcular todas las distancias pairwise dentro del cluster
            for k in range(len(puntos_cluster)):
                for l in range(k + 1, len(puntos_cluster)):
                    dist = np.linalg.norm(puntos_cluster[k] - puntos_cluster[l])
                    if dist > diametroMax:
                        diametroMax = dist
        
        if diametroMax == 0.0:
            return 0.0
        
        return separacionMin / diametroMax
    
    def _calcularMetricasInternas(self) -> Dict[str, float]:
        """Calcula métricas internas (Silueta y Dunn) - aproximadas o exactas según metricasAproximadas"""
        metricas = {"silueta": 0.0, "dunn": 0.0}
        
        if len(self.etiquetasAsignadas) < 2:
            return metricas
        
        if self.metricasAproximadas:
            # Usar aproximaciones (rápido, sin depender de guardar todos los puntos)
            metricas["silueta"] = self._calcularSiluetaAproximada()
            metricas["dunn"] = self._calcularDunnAproximado()
        else:
            # Usar exactas (requiere almacenamiento de puntos)
            # Recolectar todos los puntos y sus etiquetas en el mismo orden
            todosPuntos = []
            etiquetasPorPunto = []
            
            for idxCluster in range(len(self.puntosCluster)):
                for punto in self.puntosCluster[idxCluster]:
                    todosPuntos.append(punto)
                    etiquetasPorPunto.append(idxCluster)
            
            if len(todosPuntos) >= 3 and len(np.unique(etiquetasPorPunto)) >= 2:
                todos_puntos_array = np.array(todosPuntos)
                metricas["silueta"] = silhouette_score(todos_puntos_array, etiquetasPorPunto)
            
            metricas["dunn"] = self._calcularDunnExacto()
        
        return metricas
    
    def _calcularMetricasExternas(self) -> Dict[str, float]:
        """
        Calcula métricas externas (ARI, AMI, NMI) excluyendo puntos sin etiqueta real
        """
        metricas = {"ari": 0.0, "ami": 0.0, "nmi": 0.0}
        
        # Filtrar solo puntos con etiquetas reales (no None)
        etiquetasRealesValidas = []
        etiquetasAsignadasValidas = []
        
        for i, etiqReal in enumerate(self.etiquetasReales):
            if etiqReal is not None and i < len(self.etiquetasAsignadas):
                etiquetasRealesValidas.append(etiqReal)
                etiquetasAsignadasValidas.append(self.etiquetasAsignadas[i])
        
        # Si no hay suficientes etiquetas válidas, retornar 0
        if len(etiquetasRealesValidas) < 2:
            return metricas
        
        try:
            metricas["ari"] = adjusted_rand_score(etiquetasRealesValidas, etiquetasAsignadasValidas)
            metricas["ami"] = adjusted_mutual_info_score(etiquetasRealesValidas, etiquetasAsignadasValidas)
            metricas["nmi"] = normalized_mutual_info_score(etiquetasRealesValidas, etiquetasAsignadasValidas)
        except:
            pass
        
        return metricas
    
    def _calcularDistribucion(self) -> Dict:
        """
        Calcula la distribución de puntos por cluster incluyendo conteo de etiquetas reales
        Retorna un diccionario con:
        - Por cada cluster: número total de puntos y diccionario de etiquetas reales
        """
        distribucion = {}
        
        for idxCluster in range(len(self.centroidesAct)):
            # Encontrar todos los puntos asignados a este cluster
            conteoEtiquetasReales = {}
            totalPuntos = 0
            
            for i, etiqAsignada in enumerate(self.etiquetasAsignadas):
                if etiqAsignada == idxCluster:
                    totalPuntos += 1
                    etiqReal = self.etiquetasReales[i] if i < len(self.etiquetasReales) else None
                    
                    # Mostrar None como "sinEtiqueta"
                    if etiqReal is None:
                        etiqReal = "sinEtiqueta"
                    
                    if etiqReal not in conteoEtiquetasReales:
                        conteoEtiquetasReales[etiqReal] = 0
                    conteoEtiquetasReales[etiqReal] += 1
            
            distribucion[f"cluster_{idxCluster}"] = {
                "totalPuntos": totalPuntos,
                "etiquetasReales": conteoEtiquetasReales
            }
        
        return distribucion

    def obtenerResumenFinal(self, usarAproximadas: bool | None = None) -> Dict:
        """Devuelve un resumen final del estado del clustering.

        Incluye métricas internas, externas y distribución. Decide automáticamente
        si usa métricas aproximadas o exactas según la configuración y si se
        han guardado puntos.

        Args:
            usarAproximadas: Opcional.
                - None (por defecto):
                    * Si self.guardarPuntos es True -> intenta usar métricas exactas.
                    * Si self.guardarPuntos es False -> usa solo aproximadas.
                - True  -> fuerza uso de métricas aproximadas.
                - False -> fuerza uso de métricas exactas (requiere puntos guardados).

        Returns:
            dict con:
                - metricasInternas
                - metricasExternas
                - distribucion
                - tamaniosActuales
                - tamaniosMaximos
        """

        # Decidir modo interno según configuración y parámetros
        if usarAproximadas is None:
            usarAproximadas = not self.guardarPuntos

        # Guardar y restaurar el flag original para no afectar el flujo online
        flag_original = self.metricasAproximadas
        self.metricasAproximadas = usarAproximadas
        try:
            metricasInternas = self._calcularMetricasInternas()
        finally:
            self.metricasAproximadas = flag_original

        metricasExternas = self._calcularMetricasExternas()
        distribucion = self._calcularDistribucion()

        return {
            "metricasInternas": metricasInternas,
            "metricasExternas": metricasExternas,
            "distribucion": distribucion,
            "tamaniosActuales": self.tamaniosActuales.copy(),
            "tamaniosMaximos": self.tamaniosMaximos.copy(),
        }

    def asignarPunto(self, nuevoP: np.array, etiquetaReal: Union[int, str, None] = None) -> dict:
        """
        Asigna un nuevo punto a un cluster considerando las restricciones de tamaño
        Args:
            nuevoP: Nuevo punto a asignar
            etiquetaReal: Etiqueta real del punto (puede ser int, str o None)
        Returns:
            Diccionario con estado, etiquetaAsignada, metricasInternas, metricasExternas y distribución
        """
        try:
            # Escalar el punto si está habilitado el escalado online
            nuevoP = self._escalarPuntoOnline(nuevoP)

            # ============================================================
            # 1. Manejar el inicio de los centroides
            # ============================================================
            if not self.centroidesAct:
                # Almacenar etiqueta real antes de crear centroide
                self.etiquetasReales.append(etiquetaReal)
                self._crearCentroide(nuevoP)
                self.maxScoreHistorico = 0.0
                
                return {
                    "estado": "ok",
                    "detalle": "Primer punto asignado como centroide.",
                    "etiquetaAsignada": 0,
                    "metricasInternas": {
                        "silueta": 0.0,
                        "dunn": 0.0
                    },
                    "metricasExternas": {
                        "ari": 0.0,
                        "ami": 0.0,
                        "nmi": 0.0
                    },
                    "distribucion": self._calcularDistribucion()
                }

            # ============================================================
            # 2. Decidir si pertenece a un cluster o crear uno nuevo
            # ============================================================
            scoresFinales = self._calcularScore(nuevoP)
            validIndices = [i for i, s in enumerate(scoresFinales) if s is not None]  # Índices de los clusters válidos no llenos
            
            if not validIndices:  # No hay espacio en ningún cluster
                return {
                    "estado": "ok",
                    "detalle": "No hay espacio en ningún cluster",
                    "etiquetaAsignada": None,
                    "metricasInternas": self._calcularMetricasInternas(),
                    "metricasExternas": self._calcularMetricasExternas(),
                    "distribucion": self._calcularDistribucion()
                }
            
            scoresValidos = [scoresFinales[i] for i in validIndices]
            scoreMin = min(scoresValidos)
            indAsignado = int(validIndices[scoresValidos.index(scoreMin)])
            
            # Crear nuevo cluster si corresponde
            if len(self.centroidesAct) < self.numClusters and scoreMin > self.maxScoreHistorico:
                # Almacenar etiqueta real antes de crear centroide
                self.etiquetasReales.append(etiquetaReal)
                indAsignado = self._crearCentroide(nuevoP)
                self.maxScoreHistorico = scoreMin
                
                return {
                    "estado": "ok",
                    "detalle": "Nuevo cluster creado.",
                    "etiquetaAsignada": indAsignado,
                    "metricasInternas": self._calcularMetricasInternas(),
                    "metricasExternas": self._calcularMetricasExternas(),
                    "distribucion": self._calcularDistribucion()
                }
            
            # ============================================================
            # 3. Actualizar cluster existente
            # ============================================================
            # Almacenar etiqueta real antes de actualizar cluster
            self.etiquetasReales.append(etiquetaReal)
            self._actualizarCluster(indAsignado, nuevoP)
            
            # Actualizar histórico si superó el anterior
            if scoreMin > self.maxScoreHistorico:
                self.maxScoreHistorico = scoreMin
            
            # Calcular todas las métricas
            metricasInternas = self._calcularMetricasInternas()
            metricasExternas = self._calcularMetricasExternas()
            distribucion = self._calcularDistribucion()
            
            return {
                "estado": "ok",
                "detalle": "Punto asignado correctamente.",
                "etiquetaAsignada": indAsignado,
                "metricasInternas": metricasInternas,
                "metricasExternas": metricasExternas,
                "distribucion": distribucion
            }

        except Exception as e:
            return {"estado": "error", "detalle": str(e), "datos": None}


