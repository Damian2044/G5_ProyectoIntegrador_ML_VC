import React from 'react'

function PanelEventosClustering({ eventos }) {
  const [abierto, setAbierto] = React.useState(true)
  const ultimo = eventos.length > 0 ? eventos[eventos.length - 1] : null
  const metricasInternas = ultimo?.metricasInternas || {}
  const metricasExternas = ultimo?.metricasExternas || {}
  const aceptados = eventos.filter((e) => e.aceptado).length
  const rechazados = eventos.length - aceptados
  const tamaniosActuales = ultimo?.tamaniosActuales || []
  const tamaniosMaximos = ultimo?.tamaniosMaximos || []
  const centroides = ultimo?.centroides || []
  const distribucion = ultimo?.distribucion || {}

  return (
    <div className="mt-6 bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl min-h-[56px]">
      <div className="h-14 bg-slate-800/70 border-b border-slate-800 flex items-center px-6 justify-between cursor-pointer" onClick={() => setAbierto((v) => !v)}>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation()
              setAbierto((prev) => !prev)
            }}
            className="w-6 h-6 flex items-center justify-center rounded hover:bg-slate-700/80 transition-colors text-slate-300"
            title={abierto ? 'Cerrar resumen' : 'Abrir resumen'}
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
            Resumen Clustering Online
          </span>
        </div>
        <span className="text-sm text-slate-300 font-medium">
          {eventos.length} puntos procesados
        </span>
      </div>
      {abierto && (
      <div className="p-6 bg-slate-950/80 text-base text-slate-200 space-y-4">
        {ultimo ? (
          <>
            <div className="grid grid-cols-3 gap-6">
              <div className="bg-slate-900 rounded-xl border border-slate-800 p-4">
                <p className="text-sm text-slate-400 mb-2">Métricas internas</p>
                <p className="mb-1">
                  Silueta{' '}
                  <span className="font-semibold text-cyan-400 text-xl">
                    {(metricasInternas.silueta ?? 0).toFixed?.(3)}
                  </span>
                </p>
                <p>
                  Dunn{' '}
                  <span className="font-semibold text-cyan-400 text-xl">
                    {(metricasInternas.dunn ?? 0).toFixed?.(3)}
                  </span>
                </p>
              </div>
              <div className="bg-slate-900 rounded-xl border border-slate-800 p-4">
                <p className="text-sm text-slate-400 mb-2">Métricas externas</p>
                <p className="mb-1">
                  ARI{' '}
                  <span className="font-semibold text-emerald-400 text-xl">
                    {(metricasExternas.ari ?? 0).toFixed?.(3)}
                  </span>
                </p>
                <p className="mb-1">
                  AMI{' '}
                  <span className="font-semibold text-emerald-400 text-xl">
                    {(metricasExternas.ami ?? 0).toFixed?.(3)}
                  </span>
                </p>
                <p>
                  NMI{' '}
                  <span className="font-semibold text-emerald-400 text-xl">
                    {(metricasExternas.nmi ?? 0).toFixed?.(3)}
                  </span>
                </p>
              </div>
              <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 flex flex-col justify-center">
                <p className="text-sm text-slate-400 mb-2">Estado de puntos</p>
                <p className="mb-1">
                  Aceptados{' '}
                  <span className="text-emerald-400 font-semibold text-2xl">{aceptados}</span>
                </p>
                <p>
                  Rechazados{' '}
                  <span className="text-red-400 font-semibold text-2xl">{rechazados}</span>
                </p>
              </div>
            </div>
            {tamaniosActuales.length > 0 && (
              <div className="mt-4 bg-slate-900 rounded-xl border border-slate-800 p-4">
                <p className="text-sm text-slate-400 mb-3">
                  Tamaños y centroides reales por cluster (Actual / Máximo)
                </p>
                <div className="max-h-52 overflow-y-auto pr-1 custom-scrollbar space-y-2">
                  {tamaniosActuales.map((tamAct, idx) => {
                    const tamMax = tamaniosMaximos[idx]
                    const centro = centroides[idx]
                    const esArray = Array.isArray(centro)
                    const vectorTexto = esArray
                      ? `[${centro.map((v) =>
                          typeof v === 'number' ? v.toFixed(3) : String(v)
                        ).join(', ')}]`
                      : '-'
                    const dim = esArray ? centro.length : 0
                    return (
                      <div
                        key={idx}
                        className="text-xs text-slate-200 bg-slate-950/60 border border-slate-800 rounded-md px-2 py-1.5"
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-mono text-slate-400 mr-2">C{idx + 1}</span>
                          <span className="text-cyan-300 font-semibold mr-2">
                            {tamAct}/{tamMax ?? '-'}
                          </span>
                          {esArray && (
                            <span className="text-[10px] text-slate-500">dim = {dim}</span>
                          )}
                        </div>
                        <div className="text-[11px] text-slate-300 bg-slate-900/80 rounded px-2 py-1 overflow-x-auto whitespace-pre">
                          {vectorTexto}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
            {distribucion && Object.keys(distribucion).length > 0 && (
              <div className="mt-4 bg-slate-900 rounded-xl border border-slate-800 p-4">
                <p className="text-sm text-slate-400 mb-3">
                  Distribución de etiquetas por cluster
                </p>
                <div className="max-h-64 overflow-y-auto pr-1 custom-scrollbar">
                  <table className="w-full text-xs text-left text-slate-200 border-collapse">
                    <thead>
                      <tr className="border-b border-slate-800 text-[11px] text-slate-400">
                        <th className="py-1 pr-2 font-medium">Cluster</th>
                        <th className="py-1 pr-2 font-medium text-right">Total</th>
                        <th className="py-1 pr-2 font-medium">Etiqueta</th>
                        <th className="py-1 pr-2 font-medium text-right">Cantidad</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(distribucion).map(([clusterKey, info]) => {
                        const etiquetasReales = info?.etiquetasReales || {}
                        const totalPuntos = info?.totalPuntos ?? 0
                        const filasEtiquetas = Object.entries(etiquetasReales)
                        if (filasEtiquetas.length === 0) {
                          return (
                            <tr key={clusterKey} className="border-b border-slate-900/60">
                              <td className="py-1.5 pr-2 font-mono text-slate-300">
                                {clusterKey.replace('cluster_', 'C')}
                              </td>
                              <td className="py-1.5 pr-2 text-right text-slate-200">{totalPuntos}</td>
                              <td className="py-1.5 pr-2 text-slate-500" colSpan={2}>
                                sin etiquetas
                              </td>
                            </tr>
                          )
                        }
                        return filasEtiquetas.map(([etiqueta, cantidad], idx) => (
                          <tr key={`${clusterKey}-${etiqueta}`} className="border-b border-slate-900/60">
                            {idx === 0 ? (
                              <>
                                <td
                                  className="py-1.5 pr-2 font-mono text-slate-300 align-top"
                                  rowSpan={filasEtiquetas.length}
                                >
                                  {clusterKey.replace('cluster_', 'C')}
                                </td>
                                <td
                                  className="py-1.5 pr-2 text-right text-slate-200 align-top"
                                  rowSpan={filasEtiquetas.length}
                                >
                                  {totalPuntos}
                                </td>
                              </>
                            ) : null}
                            <td className="py-1.5 pr-2 text-slate-200">
                              {etiqueta}
                            </td>
                            <td className="py-1.5 pr-2 text-right text-slate-200">
                              {cantidad}
                            </td>
                          </tr>
                        ))
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </>
        ) : (
          <p className="text-slate-400 text-base">
            Aún no hay puntos procesados. Inicia el clustering para ver el resumen.
          </p>
        )}
      </div>
      )}
    </div>
  )
}

export default PanelEventosClustering
