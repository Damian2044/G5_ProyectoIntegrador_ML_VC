import React from 'react';

function PanelConfiguracion({
  estaClustering,
  metodoSeleccionado,
  onSeleccionMetodo,
  parametrosCluster,
  tamaniosIniciales,
  onCambiarK,
  onCambiarTamanoMaximo,
  onAplicarTamanoTodos,
  onToggleClustering,
  onAumentarTamanios,
  onEnviarNuevosDatos,
}) {
  const [tamanoComun, setTamanoComun] = React.useState('100');

  return (
    <aside className="w-80 bg-slate-900 border-l border-slate-800 flex flex-col shrink-0 z-20 shadow-xl">
      <div className="p-5 border-b border-slate-800">
        <h3 className="text-lg font-bold text-slate-200 flex items-center gap-2">
          <span className="w-1.5 h-6 bg-purple-500 rounded-full" />
          Configuración
        </h3>
        <p className="text-xs text-slate-500 mt-1 ml-3.5">Extracción &amp; Clustering</p>
      </div>

      <div className="flex-1 p-5 space-y-6 overflow-y-auto custom-scrollbar pb-10">
        <div>
          <h4 className="text-xs font-bold text-slate-400 uppercase mb-3">Extracción</h4>
          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={() => onSeleccionMetodo('momentos')}
              disabled={estaClustering}
              className={`flex-1 border rounded-lg py-2 text-xs font-bold transition-all ${
                metodoSeleccionado === 'momentos'
                  ? 'bg-purple-600 border-purple-400 text-white shadow-lg shadow-purple-900/50'
                  : 'bg-slate-800 border-slate-700 text-slate-300 hover:border-purple-500 hover:text-white'
              } ${estaClustering ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              Momentos
            </button>
            <button
              onClick={() => onSeleccionMetodo('sift')}
              disabled={estaClustering}
              className={`flex-1 border rounded-lg py-2 text-xs font-bold transition-all ${
                metodoSeleccionado === 'sift'
                  ? 'bg-orange-600 border-orange-400 text-white shadow-lg shadow-orange-900/50'
                  : 'bg-slate-800 border-slate-700 text-slate-300 hover:border-orange-500 hover:text-white'
              } ${estaClustering ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              SIFT
            </button>
            <button
              onClick={() => onSeleccionMetodo('hog')}
              disabled={estaClustering}
              className={`flex-1 border rounded-lg py-2 text-xs font-bold transition-all ${
                metodoSeleccionado === 'hog'
                  ? 'bg-pink-600 border-pink-400 text-white shadow-lg shadow-pink-900/50'
                  : 'bg-slate-800 border-slate-700 text-slate-300 hover:border-pink-500 hover:text-white'
              } ${estaClustering ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              HOG
            </button>
            <button
              onClick={() => onSeleccionMetodo('embeddings')}
              disabled={estaClustering}
              className={`flex-1 border rounded-lg py-2 text-xs font-bold transition-all ${
                metodoSeleccionado === 'embeddings'
                  ? 'bg-cyan-600 border-cyan-400 text-white shadow-lg shadow-cyan-900/50'
                  : 'bg-slate-800 border-slate-700 text-slate-300 hover:border-cyan-500 hover:text-white'
              } ${estaClustering ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              Embeddings (ResNet18)
            </button>
          </div>
          {!metodoSeleccionado && !estaClustering && (
            <p className="text-[10px] text-yellow-500 mt-2 text-center animate-pulse">
              Selecciona un método para iniciar
            </p>
          )}
        </div>

        <hr className="border-slate-800" />

        <div>
          <h4 className="text-xs font-bold text-slate-400 uppercase mb-3 flex justify-between items-center">
            Clustering Online
            {estaClustering && (
              <span className="text-[10px] bg-green-500 text-black px-2 py-0.5 rounded font-bold animate-pulse">
                RUNNING
              </span>
            )}
          </h4>
          <div
            className={`border rounded-lg p-3 mb-4 transition-colors ${
              estaClustering
                ? 'bg-green-500/10 border-green-500/30'
                : 'bg-yellow-500/5 border-yellow-500/20'
            }`}
          >
            <p
              className={`text-[10px] font-bold flex gap-2 items-start ${
                estaClustering ? 'text-green-400' : 'text-yellow-500'
              }`}
            >
              <span>{estaClustering ? '🔒' : '⚠'}</span>
              <span>
                {estaClustering
                  ? 'Modo Seguro: Parámetros bloqueados.'
                  : 'Configura los parámetros antes de iniciar el clustering.'}
              </span>
            </p>
          </div>
          <div className="space-y-4">
            <div>
              <label className="text-[10px] text-slate-400 font-bold block mb-1">
                Número de Clusters (k)
              </label>
              <input
                type="number"
                min="1"
                value={parametrosCluster.k}
                onChange={onCambiarK}
                disabled={estaClustering}
                className={`w-full bg-slate-900/50 border rounded-lg px-3 py-2 text-sm outline-none transition-colors ${
                  estaClustering
                    ? 'border-slate-800 text-slate-600 cursor-not-allowed'
                    : 'border-slate-700 focus:border-cyan-500 text-white'
                }`}
              />
            </div>
            <div>
              <label className="text-[10px] text-slate-400 font-bold block mb-2">
                Tamaños Máximos por Cluster
              </label>
              {!estaClustering && (
                <div className="flex items-center gap-2 mb-2">
                  <input
                    type="number"
                    min="1"
                    value={tamanoComun}
                    onChange={(e) => setTamanoComun(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        onAplicarTamanoTodos(tamanoComun);
                      }
                    }}
                    className="flex-1 bg-slate-900/50 border border-slate-700 rounded px-2 py-1.5 text-xs outline-none transition-all focus:border-cyan-500 text-white"
                    placeholder="Ej: 100"
                  />
                  <button
                    type="button"
                    onClick={() => onAplicarTamanoTodos(tamanoComun)}
                    className="text-[11px] font-bold px-2.5 py-1.5 rounded border transition-colors border-cyan-500/60 text-cyan-300 hover:bg-cyan-500/10"
                  >
                    Aplicar a todos
                  </button>
                </div>
              )}
              <div className="space-y-2 max-h-40 overflow-y-auto pr-1 custom-scrollbar">
                {parametrosCluster.maxSizes.map((tamano, indice) => (
                  <div key={indice} className="flex items-center gap-2">
                    <span className="text-[10px] w-6 font-mono text-slate-500">
                      C{indice}
                    </span>
                    <input
                      type="number"
                      min={estaClustering ? (tamaniosIniciales?.[indice] ?? 1) : 1}
                      value={tamano}
                      onChange={(e) => onCambiarTamanoMaximo(indice, e.target.value)}
                      className={`flex-1 bg-slate-900/50 border rounded px-2 py-1.5 text-xs outline-none transition-all ${
                        estaClustering
                          ? 'border-green-500/30 focus:border-green-500'
                          : 'border-slate-700 focus:border-cyan-500'
                      }`}
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>
          <button
            onClick={onToggleClustering}
            disabled={!estaClustering && !metodoSeleccionado}
            className={`w-full mt-6 font-bold py-3 rounded-xl shadow-lg transition-all active:scale-95 ${
              estaClustering
                ? 'bg-red-500/10 hover:bg-red-500 text-red-500 hover:text-white border border-red-500/50'
                : !metodoSeleccionado
                ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
                : 'bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white shadow-cyan-500/20'
            }`}
          >
            {estaClustering ? 'Detener Clustering' : 'Iniciar Clustering'}
          </button>
          {estaClustering && (
            <div className="mt-3 space-y-2">
              <button
                type="button"
                onClick={onEnviarNuevosDatos}
                className="w-full text-[11px] font-bold py-2 rounded-lg border border-emerald-500/60 text-emerald-300 hover:bg-emerald-500/10 transition-colors"
              >
                Mandar nuevos datos al clustering
              </button>
              <button
                type="button"
                onClick={onAumentarTamanios}
                className="w-full text-[11px] font-bold py-2 rounded-lg border border-cyan-500/60 text-cyan-300 hover:bg-cyan-500/10 transition-colors"
              >
                Aumentar tamaño clustering
              </button>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}

export default PanelConfiguracion;
