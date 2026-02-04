import React, { useState } from 'react'

function PanelClustersImagenes({ imagenesProcesadas, asignacionesClusters, tiemposClustering, cantidadClusters }) {
  const [clusterSeleccionado, setClusterSeleccionado] = useState(null)
  const [abierto, setAbierto] = useState(true)

  const obtenerImagenesCluster = (indiceCluster) => {
    const lista = imagenesProcesadas.filter((img) => asignacionesClusters[img.id] === indiceCluster)
    return lista
      .slice()
      .sort((a, b) => (tiemposClustering[b.id] || 0) - (tiemposClustering[a.id] || 0))
  }

  return (
    <div className="mt-6 bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
      <div
        className="h-14 bg-slate-800/60 border-b border-slate-800 flex items-center px-6 justify-between cursor-pointer"
        onClick={() => setAbierto((v) => !v)}
      >
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation()
              setAbierto((prev) => !prev)
            }}
            className="w-6 h-6 flex items-center justify-center rounded hover:bg-slate-700/80 transition-colors text-slate-300"
            title={abierto ? 'Cerrar clusters' : 'Abrir clusters'}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 20 20"
              fill="currentColor"
              className="w-4 h-4"
            >
              <path
                fillRule="evenodd"
                d="M1 2.75A.75.75 0 011.75 2h16.5a.75.75 0 010 1.5H1.75A.75.75 0 011 2.75zm0 9A.75.75 0 011.75 11h16.5a.75.75 0 010 1.5H1.75A.75.75 0 011 11.75zm0 5A.75.75 0 011.75 16h16.5a.75.75 0 010 1.5H1.75A.75.75 0 011 16.75zM1.75 7a.75.75 0 000 1.5h16.5a.75.75 0 000-1.5H1.75z"
                clipRule="evenodd"
              />
            </svg>
          </button>
          <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">
            Clusters de Imágenes
          </span>
        </div>
        <span className="text-sm text-slate-400">k = {cantidadClusters}</span>
      </div>
      {abierto && (
      <div className="p-5 space-y-5 bg-slate-950/70 text-sm">
        {cantidadClusters <= 0 ? (
          <p className="text-slate-600">Configura k &gt; 0 para ver clusters.</p>
        ) : (
          <>
            {/* Fila de clusters: Cluster 0 | Cluster 1 | ... con previsualización */}
            <div className="flex flex-wrap items-stretch gap-4">
              {Array.from({ length: cantidadClusters }).map((_, indiceCluster) => {
                const imagenesCluster = obtenerImagenesCluster(indiceCluster)
                const seleccion = clusterSeleccionado === indiceCluster
                const primera = imagenesCluster[0]

                return (
                  <button
                    key={indiceCluster}
                    type="button"
                    onClick={() => setClusterSeleccionado(indiceCluster)}
                    className={`flex flex-col items-center justify-between px-4 py-3 rounded-xl border min-w-[160px] cursor-pointer transition-all hover:border-cyan-500/60 hover:bg-slate-800/60 ${
                      seleccion ? 'border-cyan-500 bg-slate-800/70' : 'border-slate-700 bg-slate-900'
                    }`}
                  >
                    <span className="text-sm font-bold text-slate-200 mb-1">
                      Cluster {indiceCluster}
                    </span>
                    <span className="text-xs text-slate-300 mb-2">
                      {imagenesCluster.length} imagen
                      {imagenesCluster.length === 1 ? '' : 'es'}
                    </span>
                    <div className="w-28 h-20 rounded-lg bg-slate-800 overflow-hidden flex items-center justify-center">
                      {primera ? (
                        <img
                          src={primera.previewUrl}
                          alt={primera.file?.name || 'imagen'}
                          className="max-w-full max-h-full object-contain"
                        />
                      ) : (
                        <span className="text-xs text-slate-500">sin vista</span>
                      )}
                    </div>
                  </button>
                )
              })}
            </div>
            {/* Modal: todas las imágenes del cluster seleccionado */}
            {clusterSeleccionado !== null && (
              <div className="fixed inset-0 z-40 flex items-center justify-center bg-slate-950/80">
                <div className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl max-w-5xl w-full mx-4 max-h-[80vh] flex flex-col">
                  <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between">
                    <div>
                      <span className="text-sm font-bold text-slate-100 mr-2">
                        Cluster {clusterSeleccionado}
                      </span>
                      <span className="text-xs text-slate-400">
                        Imágenes asignadas a este cluster
                      </span>
                    </div>
                    <button
                      type="button"
                      className="text-xs text-slate-400 hover:text-cyan-400"
                      onClick={() => setClusterSeleccionado(null)}
                    >
                      Cerrar
                    </button>
                  </div>
                  <div className="flex-1 overflow-y-auto custom-scrollbar p-4">
                    <div className="grid grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                      {obtenerImagenesCluster(clusterSeleccionado).map((img) => (
                        <div
                          key={img.id}
                          className="bg-slate-900 border border-slate-800 rounded-lg overflow-hidden flex flex-col items-center p-2"
                        >
                          <div className="w-full h-28 overflow-hidden flex items-center justify-center bg-slate-800">
                            <img
                              src={img.previewUrl}
                              alt={img.file?.name || 'imagen'}
                              className="max-w-full max-h-full object-contain"
                            />
                          </div>
                          <p className="mt-1 text-[11px] text-slate-200 truncate w-full" title={img.file?.name}>
                            {img.file?.name}
                          </p>
                          <p className="text-[11px] text-cyan-400 truncate w-full" title={img.label}>
                            {img.label}
                          </p>
                        </div>
                      ))}
                      {obtenerImagenesCluster(clusterSeleccionado).length === 0 && (
                        <p className="text-sm text-slate-400 col-span-full">
                          Sin imágenes asignadas todavía a este cluster.
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
      )}
    </div>
  )
}

export default PanelClustersImagenes
