import React, { useState } from 'react'
import { obtenerPaletaClusters } from '../lib/paletaClusters'
import JSZip from 'jszip'
import { saveAs } from 'file-saver'

const MOSTRAR_DESCARGA_ZIP = !['0', 'false', 'no', 'off'].includes(
  String(import.meta.env.VITE_MOSTRAR_DESCARGA_ZIP ?? 'true').toLowerCase().trim()
)

function PanelClustersImagenes({ imagenesProcesadas, asignacionesClusters, tiemposClustering, cantidadClusters }) {
  const [clusterSeleccionado, setClusterSeleccionado] = useState(null)
  const [abierto, setAbierto] = useState(true)
  const [descargandoZip, setDescargandoZip] = useState(false)
  const paleta = obtenerPaletaClusters(cantidadClusters)

  const obtenerImagenesCluster = (indiceCluster) => {
    const lista = imagenesProcesadas.filter((img) => asignacionesClusters[img.id] === indiceCluster)
    return lista
      .slice()
      .sort((a, b) => (tiemposClustering[b.id] || 0) - (tiemposClustering[a.id] || 0))
  }

  const sanitizarTexto = (valor) => {
    if (!valor) return 'sin_etiqueta'
    return String(valor).replace(/[<>:"/\\|?*\x00-\x1F]/g, '_').trim() || 'sin_etiqueta'
  }

  const obtenerNombreArchivo = (img, indice) => {
    const original = img?.file?.name || `imagen_${indice + 1}.jpg`
    const etiqueta = sanitizarTexto(img?.label)
    return `${String(indice + 1).padStart(3, '0')}_${etiqueta}_${original}`
  }

  const crearZipDeClusters = async ({ soloCluster = null } = {}) => {
    const zip = new JSZip()
    let totalArchivos = 0

    const indices = soloCluster === null
      ? Array.from({ length: cantidadClusters }, (_, i) => i)
      : [soloCluster]

    for (const indiceCluster of indices) {
      const imagenes = obtenerImagenesCluster(indiceCluster)
      if (imagenes.length === 0) continue

      const carpeta = zip.folder(`cluster_${indiceCluster}`)
      for (let i = 0; i < imagenes.length; i++) {
        const img = imagenes[i]
        if (!img?.file) continue
        carpeta.file(obtenerNombreArchivo(img, i), img.file)
        totalArchivos += 1
      }
    }

    return { zip, totalArchivos }
  }

  const descargarResultadosClusters = async () => {
    if (descargandoZip) return

    try {
      setDescargandoZip(true)
      const { zip, totalArchivos } = await crearZipDeClusters()
      if (totalArchivos === 0) {
        alert('No hay imágenes asignadas a clusters para descargar.')
        return
      }

      const blob = await zip.generateAsync({ type: 'blob' })
      saveAs(blob, `resultados_clusters_k${cantidadClusters}.zip`)
    } catch (error) {
      console.error('Error generando ZIP de resultados', error)
      alert('No se pudo generar el ZIP de resultados.')
    } finally {
      setDescargandoZip(false)
    }
  }

  const descargarClusterSeleccionado = async () => {
    if (clusterSeleccionado === null || descargandoZip) return

    try {
      setDescargandoZip(true)
      const { zip, totalArchivos } = await crearZipDeClusters({ soloCluster: clusterSeleccionado })
      if (totalArchivos === 0) {
        alert('Este cluster no tiene imágenes para descargar.')
        return
      }

      const blob = await zip.generateAsync({ type: 'blob' })
      saveAs(blob, `cluster_${clusterSeleccionado}_resultados.zip`)
    } catch (error) {
      console.error('Error generando ZIP del cluster', error)
      alert('No se pudo generar el ZIP del cluster seleccionado.')
    } finally {
      setDescargandoZip(false)
    }
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
        <div className="flex items-center gap-3">
          <span className="text-sm text-slate-400">k = {cantidadClusters}</span>
          {MOSTRAR_DESCARGA_ZIP && (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation()
                descargarResultadosClusters()
              }}
              disabled={descargandoZip}
              className={`text-[11px] font-bold px-3 py-1.5 rounded-lg border transition-colors ${
                descargandoZip
                  ? 'border-slate-700 text-slate-500 cursor-not-allowed'
                  : 'border-cyan-500/60 text-cyan-300 hover:bg-cyan-500/10'
              }`}
            >
              {descargandoZip ? 'Generando ZIP...' : 'Descargar ZIP'}
            </button>
          )}
        </div>
      </div>
      {abierto && (
      <div className="p-5 space-y-5 bg-slate-950/70 text-sm">
        {cantidadClusters <= 0 ? (
          <p className="text-slate-600">Configura k &gt; 0 para ver clusters.</p>
        ) : (
          <>
            <div className="flex flex-wrap gap-2">
              {Array.from({ length: cantidadClusters }).map((_, indiceCluster) => (
                <div
                  key={`leyenda-cluster-${indiceCluster}`}
                  className="inline-flex items-center gap-1.5 bg-slate-900 border border-slate-700 rounded-full px-2.5 py-1"
                >
                  <span
                    className="w-2.5 h-2.5 rounded-full"
                    style={{ backgroundColor: paleta[indiceCluster % paleta.length] }}
                  />
                  <span className="text-[11px] text-slate-300 font-mono">C{indiceCluster}</span>
                </div>
              ))}
            </div>

            {/* Fila de clusters: Cluster 0 | Cluster 1 | ... con previsualización */}
            <div className="flex flex-wrap items-stretch gap-4">
              {Array.from({ length: cantidadClusters }).map((_, indiceCluster) => {
                const imagenesCluster = obtenerImagenesCluster(indiceCluster)
                const seleccion = clusterSeleccionado === indiceCluster
                const primera = imagenesCluster[0]
                const colorCluster = paleta[indiceCluster % paleta.length]

                return (
                  <button
                    key={indiceCluster}
                    type="button"
                    onClick={() => setClusterSeleccionado(indiceCluster)}
                    className={`flex flex-col items-center justify-between px-4 py-3 rounded-xl border min-w-[160px] cursor-pointer transition-all hover:border-cyan-500/60 hover:bg-slate-800/60 ${
                      seleccion ? 'border-cyan-500 bg-slate-800/70' : 'border-slate-700 bg-slate-900'
                    }`}
                    style={{ boxShadow: seleccion ? `0 0 0 1px ${colorCluster}55` : undefined }}
                  >
                    <span className="text-sm font-bold text-slate-200 mb-1 inline-flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: colorCluster }} />
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
                      <span className="text-sm font-bold text-slate-100 mr-2 inline-flex items-center gap-1.5">
                        <span
                          className="w-2.5 h-2.5 rounded-full"
                          style={{ backgroundColor: paleta[clusterSeleccionado % paleta.length] }}
                        />
                        Cluster {clusterSeleccionado}
                      </span>
                      <span className="text-xs text-slate-400">
                        Imágenes asignadas a este cluster
                      </span>
                    </div>
                    <div className="flex items-center gap-3">
                      {MOSTRAR_DESCARGA_ZIP && (
                        <button
                          type="button"
                          className={`text-xs font-semibold px-2.5 py-1 rounded border transition-colors ${
                            descargandoZip
                              ? 'border-slate-700 text-slate-500 cursor-not-allowed'
                              : 'border-cyan-500/60 text-cyan-300 hover:bg-cyan-500/10'
                          }`}
                          onClick={descargarClusterSeleccionado}
                          disabled={descargandoZip}
                        >
                          {descargandoZip ? 'Generando...' : 'Descargar ZIP'}
                        </button>
                      )}
                      <button
                        type="button"
                        className="text-xs text-slate-400 hover:text-cyan-400"
                        onClick={() => setClusterSeleccionado(null)}
                      >
                        Cerrar
                      </button>
                    </div>
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
