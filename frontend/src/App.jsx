import React, { useState, useEffect, useRef } from 'react';
import LateralIzquierdo from './componentes/LateralIzquierdo';
import BarraSeleccionFlotante from './componentes/BarraSeleccionFlotante';
import GaleriaImagenes from './componentes/GaleriaImagenes';
import PanelConfiguracion from './componentes/PanelConfiguracion';
import PanelEventosClustering from './componentes/PanelEventosClustering';
import PanelClustersImagenes from './componentes/PanelClustersImagenes';
import VisualizacionClustering from './componentes/VisualizacionClustering';
import {
  iniciarClusteringServicio,
  agregarPuntoClusteringServicio,
  actualizarTamaniosClusteringServicio,
} from './servicios/servicioClustering';

function App() {
  // --- ESTADOS GLOBALES ---
  const [batches, setBatches] = useState([]);
  const [processedImages, setProcessedImages] = useState([]);
  const [selectedImageIds, setSelectedImageIds] = useState([]);
  const [bulkLabelInput, setBulkLabelInput] = useState('');

  // --- ESTADOS CLUSTERING ---
  const [isClustering, setIsClustering] = useState(false);
  const [clusterPoints, setClusterPoints] = useState([]);
  const [initialMaxSizes, setInitialMaxSizes] = useState([]);
  const [selectedMethod, setSelectedMethod] = useState(null);
  const [incomingQueue, setIncomingQueue] = useState([]);
  const [idSesionClustering, setIdSesionClustering] = useState(null);
  const [eventosClustering, setEventosClustering] = useState([]);
  const [asignacionesClusters, setAsignacionesClusters] = useState({});
  const [idsEnClustering, setIdsEnClustering] = useState({});
  const [tiemposClustering, setTiemposClustering] = useState({});
  const [tamaniosActuales, setTamaniosActuales] = useState([]);
  const [proyeccionPCA, setProyeccionPCA] = useState(null);

  // REFERENCIAS
  const lastSelectedIndexRef = useRef(null); // Índice de la última imagen clicada
  const clusteringSectionRef = useRef(null);

  const [clusterParams, setClusterParams] = useState({
    k: 3,
    maxSizes: [10, 10, 10],
  });

  const semillaGlobal = 42;

  const crearGeneradorSemilla = (semilla) => {
    let m = 2147483647; // 2^31 - 1
    let a = 1103515245;
    let c = 12345;
    let estado = semilla % m;
    return () => {
      estado = (a * estado + c) % m;
      return estado / m;
    };
  };

  const barajarConSemilla = (array, semilla) => {
    const rnd = crearGeneradorSemilla(semilla);
    const arr = [...array];
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  };

  // --- FUNCIONES AUXILIARES ---
  const generateImageId = (batchId, index) => `b${batchId}-i${index}-${Date.now()}`;

  // --- MANEJADORES DE CARGA ---
  const handleFolderSelect = (event) => {
    const files = Array.from(event.target.files || []);
    if (files.length === 0) {
      event.target.value = null;
      return;
    }

    // Agrupar por subcarpeta dentro de la carpeta seleccionada
    // Ej: "ecommerce products/jeans/img1.jpg" -> lote "jeans"
    const gruposPorCarpeta = {};
    for (const file of files) {
      const relativePath = file.webkitRelativePath || file.name;
      const partes = relativePath.split('/');

      let nombreCarpeta;
      if (partes.length >= 3) {
        // root/subcarpeta/archivo...
        nombreCarpeta = partes[1];
      } else if (partes.length === 2) {
        // root/archivo
        nombreCarpeta = partes[0];
      } else {
        nombreCarpeta = 'Carpeta sin nombre';
      }

      nombreCarpeta = (nombreCarpeta || 'Carpeta sin nombre').trim();

      if (!gruposPorCarpeta[nombreCarpeta]) {
        gruposPorCarpeta[nombreCarpeta] = [];
      }
      gruposPorCarpeta[nombreCarpeta].push(file);
    }

    const baseId = Date.now();
    let offset = 0;
    const nuevosBatches = Object.entries(gruposPorCarpeta).map(([carpeta, archivos]) => ({
      id: baseId + offset++,
      name: carpeta,
      type: 'dataset',
      count: archivos.length,
      files: archivos,
      label: carpeta,
    }));

    if (nuevosBatches.length > 0) {
      setBatches((prev) => [...prev, ...nuevosBatches]);
    }

    event.target.value = null;
  };

  const handleMultipleSelect = (event) => {
    const files = Array.from(event.target.files);
    if (files.length > 0) {
      const newBatch = { id: Date.now(), name: "", type: 'multiple', count: files.length, files: files, label: "" };
      setBatches(prev => [...prev, newBatch]);
    }
    event.target.value = null;
  };

  const removeBatch = (id) => setBatches((prev) => prev.filter((batch) => batch.id !== id));
  const updateBatchLabel = (id, newLabel) =>
    setBatches((prev) => prev.map((b) => (b.id === id ? { ...b, label: newLabel } : b)));

  const handleProcessBatch = (batchId) => {
    const batch = batches.find(b => b.id === batchId);
    if (!batch) return;
    const finalLabel = batch.label.trim() === "" ? "Sin etiqueta" : batch.label;
    const newGalleryImages = batch.files.map((file, index) => ({
      id: generateImageId(batchId, index),
      file: file,
      previewUrl: URL.createObjectURL(file),
      label: finalLabel, 
      batchName: batch.name,
    }));
    setProcessedImages((prev) => [...prev, ...newGalleryImages]);
    // Una vez añadidas las imágenes a la galería, se elimina el lote de la lista
    setBatches((prev) => prev.filter((b) => b.id !== batchId));
  };

  // --- SELECCIÓN POR CLICK SIMPLE (SIN CTRL / SHIFT) ---
  const handleImageClick = (imgId) => {
    setSelectedImageIds((prev) => {
      if (prev.includes(imgId)) {
        // Si ya estaba seleccionada, se deselecciona
        return prev.filter((id) => id !== imgId);
      }
      // Si no estaba, se añade manteniendo posibles múltiples seleccionadas
      return [...prev, imgId];
    });
  };

  // Limpiar selección al hacer clic en el fondo vacío
  const handleBackgroundClick = (e) => {
    // Si el click fue directo al fondo (no en una imagen)
    if (e.target === e.currentTarget) {
        clearSelection();
    }
  };

  const clearSelection = () => { 
      setSelectedImageIds([]); 
      setBulkLabelInput(""); 
      lastSelectedIndexRef.current = null;
  };
  
  const applyBulkLabel = () => {
    if (bulkLabelInput.trim() === "") return;
    setProcessedImages(prev => prev.map(img => selectedImageIds.includes(img.id) ? { ...img, label: bulkLabelInput } : img));
    clearSelection();
  };

  const deleteSelectedImages = () => {
    setProcessedImages((prev) => prev.filter((img) => !selectedImageIds.includes(img.id)));
    clearSelection();
  };

  const handleMethodSelect = (method) => {
    if (isClustering) return;
    setSelectedMethod(prev => prev === method ? null : method);
  };

  // --- CONFIGURACIÓN ---
  const handleKChange = (e) => {
    if (isClustering) return; 
    const val = parseInt(e.target.value);
    if (!val || val < 1 || val > 10) return;
    const newMaxSizes = new Array(val).fill(10);
    setClusterParams(prev => ({ ...prev, k: val, maxSizes: newMaxSizes }));
  };

  const handleMaxSizeChange = (index, newVal) => {
    const val = parseInt(newVal) || 1;
    const newSizes = [...clusterParams.maxSizes];
    newSizes[index] = val;
    setClusterParams(prev => ({ ...prev, maxSizes: newSizes }));
  };

  // --- START / STOP CLUSTERING ---
  const toggleClustering = async () => {
    if (isClustering) {
      // DETENER
      setIsClustering(false);
      setClusterPoints([]);       
      setProcessedImages([]);     
      setSelectedImageIds([]);    
      setBulkLabelInput("");
      setIncomingQueue([]);
      setSelectedMethod(null);
      setIdSesionClustering(null);
      setEventosClustering([]);
      setAsignacionesClusters({});
      setIdsEnClustering({});
      setTiemposClustering({});
      setTamaniosActuales([]);
      setProyeccionPCA(null);
      lastSelectedIndexRef.current = null;
      window.location.reload();
    } else {
      // INICIAR
      if (processedImages.length < 1) {
        alert("⚠️ Carga al menos una imagen.");
        return;
      }
      if (!selectedMethod) {
        alert("⚠️ Selecciona un método de extracción.");
        return;
      }

      setInitialMaxSizes([...clusterParams.maxSizes]); 
      setClusterPoints([]); 
      setIsClustering(true);
      setProyeccionPCA(null);
      try {
        const respuestaInicio = await iniciarClusteringServicio({
          k: clusterParams.k,
          tamanosMaximos: clusterParams.maxSizes,
          metodo: selectedMethod,
          semilla: semillaGlobal,
        });

        const sesionId = respuestaInicio.sesionId;
        setIdSesionClustering(sesionId);

        const imagenesBarajadas = barajarConSemilla(processedImages, semillaGlobal);

        for (const img of imagenesBarajadas) {
          // Marcar imagen como ya enviada al clustering (sale de la galería)
          const ahora = Date.now();
          setIdsEnClustering((prev) => ({
            ...prev,
            [img.id]: true,
          }));
          setTiemposClustering((prev) => ({
            ...prev,
            [img.id]: ahora,
          }));
          try {
            const resPunto = await agregarPuntoClusteringServicio({
              sesionId,
              idFront: img.id,
              etiqueta: img.label,
              archivo: img.file,
            });

            // Actualizamos asignación de cluster por imagen
            if (resPunto.aceptado && resPunto.cluster !== null && resPunto.idFront) {
              setAsignacionesClusters((prev) => ({
                ...prev,
                [resPunto.idFront]: resPunto.cluster,
              }));
            }

            // Guardamos evento completo para métricas/resumen
            setEventosClustering((prev) => [...prev, resPunto]);
            if (Array.isArray(resPunto.tamaniosActuales)) {
              setTamaniosActuales(resPunto.tamaniosActuales);
            }
            if (resPunto.proyeccionPCA) {
              setProyeccionPCA(resPunto.proyeccionPCA);
            }
          } catch (e) {
            console.error('Error agregando punto al clustering', e);
          }
        }
      } catch (error) {
        alert("Error conectando con backend.");
        setIsClustering(false);
      }
    }
  };

  const confirmarAumentoTamanios = async () => {
    if (!isClustering || !idSesionClustering) return;

    // Validación frontend: no permitir valores menores que los tamaños actuales
    if (Array.isArray(tamaniosActuales) && tamaniosActuales.length > 0) {
      for (let i = 0; i < clusterParams.maxSizes.length; i++) {
        const actual = tamaniosActuales[i] ?? 0;
        const propuesto = clusterParams.maxSizes[i] ?? 0;
        if (propuesto < actual) {
          alert(
            `El tamaño máximo de C${i + 1} (${propuesto}) no puede ser menor que el tamaño actual (${actual}).`
          );
          return;
        }
      }
    }
    try {
      const respuesta = await actualizarTamaniosClusteringServicio({
        sesionId: idSesionClustering,
        tamanosNuevos: clusterParams.maxSizes,
      });

      if (respuesta?.estado === 'ok' && Array.isArray(respuesta.tamaniosMaximos)) {
        setClusterParams((prev) => ({
          ...prev,
          maxSizes: [...respuesta.tamaniosMaximos],
        }));
        setInitialMaxSizes([...respuesta.tamaniosMaximos]);
        if (Array.isArray(respuesta.tamaniosActuales)) {
          setTamaniosActuales(respuesta.tamaniosActuales);
        }
      }
    } catch (e) {
      console.error('Error actualizando tamaños máximos', e);
      let mensaje = 'No se pudieron actualizar los tamaños máximos. Revisa las restricciones.';
      if (e && typeof e === 'object') {
        if (e.detail) mensaje = e.detail;
        else if (e.message) mensaje = e.message;
      }
      alert(mensaje);
    }
  };

  const enviarNuevasImagenesAlClustering = async () => {
    if (!isClustering || !idSesionClustering) return;

    const nuevasImagenes = processedImages.filter((img) => !idsEnClustering[img.id]);
    if (nuevasImagenes.length === 0) {
      alert('No hay nuevas imágenes pendientes para enviar al clustering.');
      return;
    }

    // Puedes barajar las nuevas o enviarlas en orden actual
    const imagenesAEnviar = barajarConSemilla(nuevasImagenes, semillaGlobal);

    for (const img of imagenesAEnviar) {
      const ahora = Date.now();
      setIdsEnClustering((prev) => ({
        ...prev,
        [img.id]: true,
      }));
      setTiemposClustering((prev) => ({
        ...prev,
        [img.id]: ahora,
      }));

      try {
        const resPunto = await agregarPuntoClusteringServicio({
          sesionId: idSesionClustering,
          idFront: img.id,
          etiqueta: img.label,
          archivo: img.file,
        });

        if (resPunto.aceptado && resPunto.cluster !== null && resPunto.idFront) {
          setAsignacionesClusters((prev) => ({
            ...prev,
            [resPunto.idFront]: resPunto.cluster,
          }));
        }

        setEventosClustering((prev) => [...prev, resPunto]);
        if (Array.isArray(resPunto.tamaniosActuales)) {
          setTamaniosActuales(resPunto.tamaniosActuales);
        }
        if (resPunto.proyeccionPCA) {
          setProyeccionPCA(resPunto.proyeccionPCA);
        }
      } catch (e) {
        console.error('Error agregando nueva imagen al clustering', e);
      }
    }
  };

  // --- EFECTOS ---
  useEffect(() => {
    if (isClustering && clusteringSectionRef.current) {
        setTimeout(() => clusteringSectionRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
    }
  }, [isClustering]);

  useEffect(() => {
    // La lógica de animación de puntos se ha desactivado temporalmente
  }, [isClustering, incomingQueue, clusterParams]);

  return (
    <div className="flex min-h-screen w-screen bg-slate-900 text-white font-sans overflow-x-hidden relative select-none">
      {/* COLUMNA IZQUIERDA: CARGADOR + CONFIGURACIÓN */}
        <div className="flex flex-col shrink-0 bg-slate-900 border-r border-slate-800">
          <LateralIzquierdo
            lotes={batches}
            onProcesarLote={handleProcessBatch}
            onEliminarLote={removeBatch}
            onActualizarEtiqueta={updateBatchLabel}
            onSeleccionCarpeta={handleFolderSelect}
            onSeleccionMultiple={handleMultipleSelect}
          />
          <PanelConfiguracion
            estaClustering={isClustering}
            metodoSeleccionado={selectedMethod}
            onSeleccionMetodo={handleMethodSelect}
            parametrosCluster={clusterParams}
            tamaniosIniciales={initialMaxSizes}
            onCambiarK={handleKChange}
            onCambiarTamanoMaximo={handleMaxSizeChange}
            onToggleClustering={toggleClustering}
            onAumentarTamanios={confirmarAumentoTamanios}
            onEnviarNuevosDatos={enviarNuevasImagenesAlClustering}
          />
      </div>

      {/* ÁREA PRINCIPAL: GALERÍA + RESULTADOS */}
      <main className="flex-1 flex flex-col h-full bg-slate-950 relative min-w-0">
        <header className="h-20 shrink-0 border-b border-slate-800 bg-slate-900/80 backdrop-blur-md px-8 flex items-center justify-between z-10 relative">
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">
              Proyecto Integrador: Visión por Computador y Aprendizaje Automático
            </h1>
            <p className="text-xs text-slate-500">Carga, etiquetado y clustering de imágenes</p>
          </div>
        </header>

        {/* Barra flotante de selección */}
        <BarraSeleccionFlotante
          cantidadSeleccionados={selectedImageIds.length}
          textoEtiqueta={bulkLabelInput}
          onCambiarTextoEtiqueta={setBulkLabelInput}
          onAplicarEtiquetaMasiva={applyBulkLabel}
          onBorrarSeleccion={deleteSelectedImages}
          onLimpiarSeleccion={clearSelection}
        />

        {/* Contenedor principal */}
        <div className="flex-1 overflow-y-auto custom-scrollbar p-6 flex flex-col gap-8 pb-20 scroll-smooth">
          {/* Galería de imágenes */}
          <GaleriaImagenes
            imagenesProcesadas={processedImages.filter((img) => !idsEnClustering[img.id])}
            idsSeleccionados={selectedImageIds}
            onClickImagen={handleImageClick}
            onClickFondo={handleBackgroundClick}
            onLimpiarTodo={() => setProcessedImages([])}
          />

          {/* Proyección PCA 2D (visualización) */}
          <VisualizacionClustering proyeccionPCA={proyeccionPCA} />

          {/* Resumen de eventos y métricas del clustering */}
          <PanelEventosClustering eventos={eventosClustering} />

          {/* Visualización de imágenes agrupadas por cluster */}
          <PanelClustersImagenes
            imagenesProcesadas={processedImages}
            asignacionesClusters={asignacionesClusters}
            tiemposClustering={tiemposClustering}
            cantidadClusters={clusterParams.k}
          />
        </div>
      </main>

    </div>
  );
}

export default App; 