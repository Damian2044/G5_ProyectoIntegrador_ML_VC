import React, { useState, useEffect, useRef } from 'react';

// --- COMPONENTES AUXILIARES ---
const ClusterPoint = ({ x, y, color }) => (
  <div 
    className="absolute w-4 h-4 rounded-full shadow-lg border-2 border-white transition-all duration-700 ease-out"
    style={{ left: `${x}%`, top: `${y}%`, backgroundColor: color, transform: 'translate(-50%, -50%)' }}
  />
);

const CLUSTER_COLORS = ['#06b6d4', '#ec4899', '#eab308', '#8b5cf6', '#10b981', '#f97316', '#ef4444', '#6366f1'];

function App() {
  // --- ESTADOS ---
  const [batches, setBatches] = useState([]); 
  const [processedImages, setProcessedImages] = useState([]); 
  const [selectedImageIds, setSelectedImageIds] = useState([]); 
  const [bulkLabelInput, setBulkLabelInput] = useState(""); 
  
  // --- ESTADOS CLUSTERING ---
  const [isClustering, setIsClustering] = useState(false);
  const [clusterPoints, setClusterPoints] = useState([]);
  const [initialMaxSizes, setInitialMaxSizes] = useState([]); 
  const [selectedMethod, setSelectedMethod] = useState(null); 
  const [incomingQueue, setIncomingQueue] = useState([]);

  // --- REFERENCIA PARA LA LÓGICA DE SHIFT (WINDOWS STYLE) ---
  const lastSelectedIndexRef = useRef(null); // Guarda el ÍNDICE de la última foto clickeada

  const [clusterParams, setClusterParams] = useState({
    k: 3,
    maxSizes: [10, 10, 10], 
    speed: 300
  });

  // REFS
  const clusteringSectionRef = useRef(null);

  // --- FUNCIONES AUXILIARES ---
  const generateImageId = (batchId, index) => `b${batchId}-i${index}-${Date.now()}`;

  // --- MANEJADORES DE CARGA ---
  const handleFolderSelect = (event) => {
    const files = Array.from(event.target.files);
    if (files.length > 0) {
      const folderName = files[0].webkitRelativePath.split('/')[0] || "Carpeta sin nombre";
      const newBatch = { id: Date.now(), name: folderName, type: 'dataset', count: files.length, files: files, label: folderName };
      setBatches(prev => [...prev, newBatch]);
    }
    event.target.value = null;
  };

  const handleMultipleSelect = (event) => {
    const files = Array.from(event.target.files);
    if (files.length > 0) {
      const newBatch = { id: Date.now(), name: `Selección Manual #${batches.length + 1}`, type: 'multiple', count: files.length, files: files, label: "" };
      setBatches(prev => [...prev, newBatch]);
    }
    event.target.value = null;
  };

  const removeBatch = (id) => setBatches(prev => prev.filter(batch => batch.id !== id));
  const updateBatchLabel = (id, newLabel) => setBatches(prev => prev.map(b => (b.id === id ? { ...b, label: newLabel } : b)));

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
    setProcessedImages(prev => [...prev, ...newGalleryImages]);
  };

  // --- LÓGICA DE SELECCIÓN TIPO WINDOWS (SHIFT / CTRL) ---
  const handleImageClick = (imgId, index, e) => {
    // 1. SHIFT + CLICK (Selección de Rango)
    if (e.shiftKey && lastSelectedIndexRef.current !== null) {
      const start = Math.min(lastSelectedIndexRef.current, index);
      const end = Math.max(lastSelectedIndexRef.current, index);
      
      // Obtenemos los IDs de todas las imágenes en ese rango
      const rangeIds = processedImages.slice(start, end + 1).map(img => img.id);
      
      // En Windows, Shift reemplaza la selección con el rango.
      // Si quieres que Shift agregue a lo que ya tenías con Ctrl, usa la lógica de abajo.
      // Aquí hacemos la versión estándar: Rango define la nueva selección activa.
      setSelectedImageIds(rangeIds);
    } 
    // 2. CTRL + CLICK (Selección Múltiple Individual)
    else if (e.ctrlKey || e.metaKey) {
      setSelectedImageIds(prev => {
        if (prev.includes(imgId)) {
          return prev.filter(id => id !== imgId); // Deseleccionar si ya estaba
        } else {
          return [...prev, imgId]; // Agregar si no estaba
        }
      });
      // Actualizamos el "ancla" para un futuro Shift
      lastSelectedIndexRef.current = index;
    } 
    // 3. CLICK NORMAL (Selección Única - Borra lo demás)
    else {
      setSelectedImageIds([imgId]);
      lastSelectedIndexRef.current = index; // Actualizamos el ancla
    }
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
    setProcessedImages(prev => prev.filter(img => !selectedImageIds.includes(img.id)));
    clearSelection();
  };

  const handleFeatureExtraction = (method) => {
    if (selectedImageIds.length === 0) { alert(`Selecciona imágenes para ${method}`); return; }
    console.log(`Extrayendo ${method}...`);
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
    if (isClustering) { if (val < initialMaxSizes[index]) return; }
    const newSizes = [...clusterParams.maxSizes];
    newSizes[index] = val;
    setClusterParams(prev => ({ ...prev, maxSizes: newSizes }));
  };

  const handleSpeedChange = (e) => {
    const { value } = e.target;
    if (value < 100 && value !== "") return;
    setClusterParams(prev => ({ ...prev, speed: parseInt(value) || 100 }));
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
      lastSelectedIndexRef.current = null;
    } else {
      // INICIAR
      if (processedImages.length < 2) {
        alert("⚠️ Carga más imágenes.");
        return;
      }
      if (!selectedMethod) {
        alert("⚠️ Selecciona un método de extracción.");
        return;
      }

      setInitialMaxSizes([...clusterParams.maxSizes]); 
      setClusterPoints([]); 
      setIsClustering(true);
      
      const formData = new FormData();
      formData.append("k", clusterParams.k);
      formData.append("metodo", selectedMethod);
      processedImages.forEach((img) => formData.append("files", img.file));

      try {
        const response = await fetch("http://127.0.0.1:8000/api/cluster", { method: "POST", body: formData });
        if (!response.ok) throw new Error("Error en el servidor");
        const data = await response.json();
        
        if (data.puntos && data.puntos.length > 0) {
            const coloredPoints = data.puntos.map(p => ({
                ...p,
                uniqueId: Math.random(), 
                color: CLUSTER_COLORS[p.cIndex % CLUSTER_COLORS.length]
            }));
            setIncomingQueue(coloredPoints);
        } else {
            alert("El backend no devolvió puntos.");
            setIsClustering(false);
        }
      } catch (error) {
        alert("Error conectando con backend.");
        setIsClustering(false);
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
    let interval;
    if (isClustering && incomingQueue.length > 0) {
      interval = setInterval(() => {
        setIncomingQueue(currentQueue => {
          if (currentQueue.length === 0) return [];
          const [nextPoint, ...rest] = currentQueue;
          setClusterPoints(prevPoints => {
             const countInCluster = prevPoints.filter(p => p.cIndex === nextPoint.cIndex).length;
             if (countInCluster < clusterParams.maxSizes[nextPoint.cIndex]) {
                 return [...prevPoints, nextPoint];
             }
             return prevPoints;
          });
          return rest;
        });
      }, clusterParams.speed);
    }
    return () => clearInterval(interval);
  }, [isClustering, incomingQueue, clusterParams]);


  return (
    <div className="flex h-screen w-screen bg-slate-900 text-white font-sans overflow-hidden relative select-none">
      
      {/* 1. IZQUIERDA: SIDEBAR */}
      <aside className="w-80 bg-slate-900 border-r border-slate-800 flex flex-col shrink-0 z-20 shadow-xl">
        <div className="p-5 border-b border-slate-800">
           <h3 className="text-lg font-bold text-slate-200 flex items-center gap-2">
            <span className="w-1.5 h-6 bg-gradient-to-b from-blue-500 to-cyan-400 rounded-full"></span> 
            Cola de Lotes
          </h3>
          <span className="text-xs text-slate-500 ml-3.5">{batches.length} pendientes</span>
        </div>
        <div className="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-4">
          {batches.map((batch) => (
              <div key={batch.id} className="bg-slate-800 border border-slate-700 rounded-xl p-3 shadow-lg relative group transition-all hover:border-slate-500">
                <div className={`absolute left-0 top-3 bottom-3 w-1 rounded-r ${batch.type === 'dataset' ? 'bg-blue-500' : 'bg-emerald-500'}`}></div>
                <div className="pl-3">
                  <div className="flex justify-between items-start mb-2">
                    <h4 className="text-sm font-bold text-white truncate w-32" title={batch.name}>{batch.name}</h4>
                    <span className="bg-slate-900 text-slate-400 text-[10px] px-1.5 py-0.5 rounded border border-slate-700">{batch.count}</span>
                  </div>
                  <div className="relative mb-2">
                    <span className="text-[10px] text-slate-500 absolute -top-1.5 left-2 bg-slate-800 px-1">Etiqueta</span>
                    <input type="text" value={batch.label} onChange={(e) => updateBatchLabel(batch.id, e.target.value)} className="w-full bg-slate-900/50 border border-slate-700 rounded px-2 py-1.5 text-xs text-cyan-300 focus:border-cyan-500 outline-none"/>
                  </div>
                  <div className="flex gap-2">
                    <button onClick={() => handleProcessBatch(batch.id)} className="flex-1 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white text-xs font-bold py-1.5 rounded transition-all shadow-lg shadow-cyan-900/20">Procesar</button>
                    <button onClick={() => removeBatch(batch.id)} className="bg-slate-700 hover:bg-red-500/20 hover:text-red-400 text-slate-400 p-1.5 rounded transition-colors">✕</button>
                  </div>
                </div>
              </div>
          ))}
        </div>
        <div className="p-4 border-t border-slate-800 bg-slate-900">
           <div className="grid grid-cols-2 gap-2">
              <label className="bg-slate-800 hover:bg-slate-700 border border-slate-700 cursor-pointer rounded-lg p-3 text-center transition-all group">
                 <div className="text-xl mb-1">📂</div>
                 <div className="text-[10px] text-slate-400 font-bold uppercase">Carpeta</div>
                 <input type="file" webkitdirectory="true" directory="true" className="hidden" onChange={handleFolderSelect} />
              </label>
              <label className="bg-slate-800 hover:bg-slate-700 border border-slate-700 cursor-pointer rounded-lg p-3 text-center transition-all group">
                 <div className="text-xl mb-1">🖼️</div>
                 <div className="text-[10px] text-slate-400 font-bold uppercase">Fotos</div>
                 <input type="file" multiple accept="image/*" className="hidden" onChange={handleMultipleSelect} />
              </label>
           </div>
        </div>
      </aside>

      {/* 2. ÁREA CENTRAL */}
      <main className="flex-1 flex flex-col h-full bg-slate-950 relative overflow-hidden min-w-0">
        
        <header className="h-20 shrink-0 border-b border-slate-800 bg-slate-900/80 backdrop-blur-md px-8 flex items-center justify-between z-10 relative">
            <div>
              <h1 className="text-2xl font-bold text-white tracking-tight">AI Vision Studio</h1>
              <p className="text-xs text-slate-500">Preprocesamiento y Etiquetado</p>
            </div>
        </header>

        {/* Barra Flotante */}
        <div className={`fixed top-24 left-1/2 -translate-x-1/2 z-50 transition-all duration-300 transform ${selectedImageIds.length > 0 ? 'translate-y-0 opacity-100' : 'translate-y-[-20px] opacity-0 pointer-events-none'}`}>
            <div className="bg-slate-800/90 backdrop-blur-md border border-slate-600 rounded-xl p-4 flex items-center gap-4 shadow-2xl w-[90vw] max-w-2xl">
                <span className="bg-cyan-500/10 text-cyan-400 text-sm font-bold px-3 py-1.5 rounded-lg whitespace-nowrap">{selectedImageIds.length} items</span>
                <input type="text" placeholder="Nueva etiqueta..." value={bulkLabelInput} onChange={(e) => setBulkLabelInput(e.target.value)} className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:border-cyan-500 outline-none"/>
                <button onClick={applyBulkLabel} className="bg-cyan-600 hover:bg-cyan-500 text-white px-4 py-2 rounded-lg text-sm font-bold transition-colors">Aplicar</button>
                <div className="h-6 w-px bg-slate-600 mx-1"></div>
                <button onClick={deleteSelectedImages} className="bg-red-500/10 hover:bg-red-500 hover:text-white text-red-400 border border-red-500/30 px-3 py-2 rounded-lg text-sm font-bold transition-colors flex items-center gap-2">Borrar</button>
                <button onClick={clearSelection} className="text-slate-400 hover:text-white px-2 text-sm font-medium">✕</button>
            </div>
        </div>

        {/* CONTENEDOR PRINCIPAL */}
        <div className="flex-1 overflow-y-auto custom-scrollbar p-6 flex flex-col gap-8 pb-20 scroll-smooth">
          
          {/* A. GALERÍA (Con clic en fondo para limpiar) */}
          <div className="h-[500px] shrink-0 bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden flex flex-col shadow-2xl relative">
            <div className="h-10 bg-slate-800/50 border-b border-slate-800 flex items-center justify-between px-4 shrink-0 z-20 relative pointer-events-auto">
               <span className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                 <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4"><path fillRule="evenodd" d="M1 2.75A.75.75 0 011.75 2h16.5a.75.75 0 010 1.5H1.75A.75.75 0 011 2.75zm0 9A.75.75 0 011.75 11h16.5a.75.75 0 010 1.5H1.75A.75.75 0 011 11.75zm0 5A.75.75 0 011.75 16h16.5a.75.75 0 010 1.5H1.75A.75.75 0 011 16.75zM1.75 7a.75.75 0 000 1.5h16.5a.75.75 0 000-1.5H1.75z" clipRule="evenodd" /></svg>
                 Galería ({processedImages.length})
               </span>
               <button onClick={() => setProcessedImages([])} className="text-[10px] text-slate-500 hover:text-red-400 transition-colors">Limpiar Todo</button>
            </div>
            
            <div 
                className="flex-1 overflow-y-auto custom-scrollbar p-6 bg-slate-900/50 relative"
                onClick={handleBackgroundClick} // CLICK EN FONDO LIMPIA
            >
              {processedImages.length === 0 ? (
                 <div className="h-full flex flex-col items-center justify-center opacity-30 pointer-events-none">
                    <div className="w-24 h-24 bg-slate-800 rounded-full flex items-center justify-center mb-4"><span className="text-4xl">📷</span></div>
                    <p className="text-slate-400 font-medium">Esperando procesamiento...</p>
                 </div>
              ) : (
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
                  {processedImages.map((imgData, index) => {
                    const isSelected = selectedImageIds.includes(imgData.id);
                    return (
                      <div 
                        key={imgData.id} 
                        onClick={(e) => handleImageClick(imgData.id, index, e)} // AHORA PASAMOS EL INDEX
                        className={`relative group cursor-pointer transition-all duration-100 bg-slate-800 rounded-xl p-2 border image-card ${isSelected ? 'border-cyan-500 ring-2 ring-cyan-500/30 shadow-lg shadow-cyan-900/50 scale-[1.02]' : 'border-slate-700 hover:border-slate-500 hover:bg-slate-750'}`}
                      >
                        <div className="bg-white rounded-lg h-32 w-full flex items-center justify-center overflow-hidden mb-2 relative pointer-events-none">
                           <img src={imgData.previewUrl} alt="preview" className="max-h-full max-w-full object-contain p-2" loading="lazy" />
                           <div className={`absolute inset-0 bg-slate-900/10 transition-opacity ${isSelected ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'}`}>
                              <div className={`absolute top-2 left-2 w-5 h-5 rounded border shadow-sm flex items-center justify-center ${isSelected ? 'bg-cyan-500 border-cyan-500' : 'bg-white border-slate-300'}`}>{isSelected && <span className="text-white text-xs font-bold">✓</span>}</div>
                           </div>
                        </div>
                        <div className="px-1 pointer-events-none">
                          <p className={`text-xs font-bold truncate mb-0.5 ${imgData.label === "Sin etiqueta" ? "text-slate-500 italic" : "text-cyan-300"}`}>{imgData.label}</p>
                          <p className="text-[10px] text-slate-500 truncate">{imgData.batchName}</p>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>

          {/* B. CLUSTERING ONLINE */}
          {isClustering && (
            <div ref={clusteringSectionRef} className="h-[600px] shrink-0 bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl relative flex flex-col animate-in slide-in-from-bottom-10 fade-in duration-500 scroll-mt-6">
                <div className="h-10 bg-green-900/20 border-b border-green-500/30 flex items-center justify-between px-4 shrink-0">
                    <span className="text-xs font-bold text-green-400 uppercase tracking-wider flex items-center gap-2 animate-pulse">
                        ● Visualización en Tiempo Real
                    </span>
                    <span className="text-[10px] text-slate-400">Scatter Plot 2D (Datos Reales)</span>
                </div>
                
                <div className="flex-1 relative bg-slate-950 overflow-hidden p-4">
                  <div className="absolute inset-0 opacity-10" style={{ backgroundImage: 'radial-gradient(#4b5563 1px, transparent 1px)', backgroundSize: '20px 20px' }}></div>
                  {clusterPoints.map(p => (
                      <ClusterPoint key={p.uniqueId} x={p.x} y={p.y} color={p.color} />
                  ))}
                  <div className="absolute bottom-4 left-4 flex gap-4 bg-slate-900/80 backdrop-blur border border-slate-700 p-2 rounded-lg z-10">
                      {Array.from({length: clusterParams.k}).map((_, idx) => {
                          const count = clusterPoints.filter(p => p.cIndex === idx).length;
                          const max = clusterParams.maxSizes[idx];
                          return (
                            <div key={idx} className="flex items-center gap-2 text-xs">
                                <span className="w-3 h-3 rounded-full" style={{backgroundColor: CLUSTER_COLORS[idx % CLUSTER_COLORS.length]}}></span>
                                <span className="text-slate-300 font-bold">C{idx+1}:</span>
                                <span className={`${count >= max ? 'text-red-400 font-bold' : 'text-cyan-400'}`}>{count}/{max}</span>
                            </div>
                          );
                      })}
                  </div>
               </div>
            </div>
          )}
          
        </div>
      </main>

      {/* 3. DERECHA: CONFIGURACIÓN */}
      <aside className="w-80 bg-slate-900 border-l border-slate-800 flex flex-col shrink-0 z-20 shadow-xl">
        <div className="p-5 border-b border-slate-800">
           <h3 className="text-lg font-bold text-slate-200 flex items-center gap-2">
            <span className="w-1.5 h-6 bg-purple-500 rounded-full"></span> 
            Configuración
          </h3>
          <p className="text-xs text-slate-500 mt-1 ml-3.5">Extracción & Clustering</p>
        </div>

        <div className="flex-1 p-5 space-y-6 overflow-y-auto custom-scrollbar pb-10">
          
          <div>
            <h4 className="text-xs font-bold text-slate-400 uppercase mb-3">Extracción</h4>
            <div className="flex gap-2">
              <button onClick={() => handleMethodSelect('momentos')} disabled={isClustering} className={`flex-1 border rounded-lg py-2 text-xs font-bold transition-all ${selectedMethod === 'momentos' ? 'bg-purple-600 border-purple-400 text-white shadow-lg shadow-purple-900/50' : 'bg-slate-800 border-slate-700 text-slate-300 hover:border-purple-500 hover:text-white'} ${isClustering ? 'opacity-50 cursor-not-allowed' : ''}`}>Momentos</button>
              <button onClick={() => handleMethodSelect('sift')} disabled={isClustering} className={`flex-1 border rounded-lg py-2 text-xs font-bold transition-all ${selectedMethod === 'sift' ? 'bg-orange-600 border-orange-400 text-white shadow-lg shadow-orange-900/50' : 'bg-slate-800 border-slate-700 text-slate-300 hover:border-orange-500 hover:text-white'} ${isClustering ? 'opacity-50 cursor-not-allowed' : ''}`}>SIFT</button>
              <button onClick={() => handleMethodSelect('hog')} disabled={isClustering} className={`flex-1 border rounded-lg py-2 text-xs font-bold transition-all ${selectedMethod === 'hog' ? 'bg-pink-600 border-pink-400 text-white shadow-lg shadow-pink-900/50' : 'bg-slate-800 border-slate-700 text-slate-300 hover:border-pink-500 hover:text-white'} ${isClustering ? 'opacity-50 cursor-not-allowed' : ''}`}>HOG</button>
            </div>
            {!selectedMethod && !isClustering && (
               <p className="text-[10px] text-yellow-500 mt-2 text-center animate-pulse">Selecciona un método para iniciar</p>
            )}
          </div>

          <hr className="border-slate-800" />

          <div>
             <h4 className="text-xs font-bold text-slate-400 uppercase mb-3 flex justify-between items-center">
               Clustering Online
               {isClustering && <span className="text-[10px] bg-green-500 text-black px-2 py-0.5 rounded font-bold animate-pulse">RUNNING</span>}
             </h4>
             <div className={`border rounded-lg p-3 mb-4 transition-colors ${isClustering ? 'bg-green-500/10 border-green-500/30' : 'bg-yellow-500/5 border-yellow-500/20'}`}>
                <p className={`text-[10px] font-bold flex gap-2 items-start ${isClustering ? 'text-green-400' : 'text-yellow-500'}`}>
                  <span>{isClustering ? '🔒' : '⚠'}</span>
                  <span>{isClustering ? "Modo Seguro: Parámetros bloqueados." : "Restricciones activas: Max 10 clusters."}</span>
                </p>
             </div>
             <div className="space-y-4">
               <div>
                 <label className="text-[10px] text-slate-400 font-bold block mb-1">Número de Clusters (k)</label>
                 <input type="number" min="1" max="10" value={clusterParams.k} onChange={handleKChange} disabled={isClustering} className={`w-full bg-slate-900/50 border rounded-lg px-3 py-2 text-sm outline-none transition-colors ${isClustering ? 'border-slate-800 text-slate-600 cursor-not-allowed' : 'border-slate-700 focus:border-cyan-500 text-white'}`}/>
               </div>
               <div>
                 <label className="text-[10px] text-slate-400 font-bold block mb-2">Tamaños Máximos por Cluster</label>
                 <div className="space-y-2 max-h-40 overflow-y-auto pr-1 custom-scrollbar">
                   {clusterParams.maxSizes.map((size, index) => (
                     <div key={index} className="flex items-center gap-2">
                       <span className="text-[10px] w-6 font-mono text-slate-500">C{index + 1}</span>
                       <input type="number" min="1" value={size} onChange={(e) => handleMaxSizeChange(index, e.target.value)} className={`flex-1 bg-slate-900/50 border rounded px-2 py-1.5 text-xs outline-none transition-all ${isClustering ? 'border-green-500/30 focus:border-green-500' : 'border-slate-700 focus:border-cyan-500'}`}/>
                       {isClustering && <span className="text-[9px] text-slate-600 w-12 text-right">Min: {initialMaxSizes[index]}</span>}
                     </div>
                   ))}
                 </div>
               </div>
               <div>
                 <label className="text-[10px] text-slate-400 font-bold block mb-1">Velocidad Simulación (ms)</label>
                 <input type="number" min="100" step="100" value={clusterParams.speed} onChange={handleSpeedChange} className="w-full bg-slate-900/50 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:border-cyan-500 outline-none"/>
               </div>
             </div>
             <button 
                onClick={toggleClustering}
                disabled={!isClustering && !selectedMethod}
                className={`w-full mt-6 font-bold py-3 rounded-xl shadow-lg transition-all active:scale-95 ${isClustering ? 'bg-red-500/10 hover:bg-red-500 text-red-500 hover:text-white border border-red-500/50' : (!selectedMethod ? 'bg-slate-700 text-slate-500 cursor-not-allowed' : 'bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white shadow-cyan-500/20')}`}
             >
                {isClustering ? "Detener Clustering" : "Iniciar Clustering"}
             </button>
          </div>
        </div>
      </aside>

    </div>
  );
}

export default App; 