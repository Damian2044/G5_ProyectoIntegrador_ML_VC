import React from 'react';

function LateralIzquierdo({
  lotes,
  onProcesarLote,
  onEliminarLote,
  onActualizarEtiqueta,
  onSeleccionCarpeta,
  onSeleccionMultiple,
}) {
  return (
    <aside className="w-80 bg-slate-900 border-r border-slate-800 flex flex-col shrink-0 z-20 shadow-xl">
      <div className="p-5 border-b border-slate-800">
        <h3 className="text-lg font-bold text-slate-200 flex items-center gap-2">
          <span className="w-1.5 h-6 bg-gradient-to-b from-blue-500 to-cyan-400 rounded-full" />
          Carga de imágenes
        </h3>
        <span className="text-xs text-slate-500 ml-3.5">{lotes.length} pendientes</span>
      </div>

      <div className="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-4">
        {lotes.map((lote) => (
          <div
            key={lote.id}
            className="bg-slate-800 border border-slate-700 rounded-xl p-3 shadow-lg relative group transition-all hover:border-slate-500"
          >
            <div
              className={`absolute left-0 top-3 bottom-3 w-1 rounded-r ${
                lote.type === 'dataset' ? 'bg-blue-500' : 'bg-emerald-500'
              }`}
            />
            <div className="pl-3">
              <div className="flex justify-between items-start mb-2">
                <h4
                  className="text-sm font-bold text-white truncate w-32"
                  title={lote.name}
                >
                  {lote.name}
                </h4>
                <span className="bg-slate-900 text-slate-400 text-[10px] px-1.5 py-0.5 rounded border border-slate-700">
                  {lote.count}
                </span>
              </div>
              <div className="relative mb-2">
                <span className="text-[10px] text-slate-500 absolute -top-1.5 left-2 bg-slate-800 px-1">
                  Etiqueta
                </span>
                <input
                  type="text"
                  value={lote.label}
                  placeholder="Opcional, pulsa Enter"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      onProcesarLote(lote.id);
                    }
                  }}
                  onChange={(e) => onActualizarEtiqueta(lote.id, e.target.value)}
                  className="w-full bg-slate-900/50 border border-slate-700 rounded px-2 py-1.5 text-xs text-cyan-300 focus:border-cyan-500 outline-none"
                />
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => onProcesarLote(lote.id)}
                  className="flex-1 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white text-xs font-bold py-1.5 rounded transition-all shadow-lg shadow-cyan-900/20"
                >
                  Añadir
                </button>
                <button
                  onClick={() => onEliminarLote(lote.id)}
                  className="bg-slate-700 hover:bg-red-500/20 hover:text-red-400 text-slate-400 p-1.5 rounded transition-colors"
                >
                  ✕
                </button>
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
            <input
              type="file"
              webkitdirectory="true"
              directory="true"
              className="hidden"
              onChange={onSeleccionCarpeta}
            />
          </label>
          <label className="bg-slate-800 hover:bg-slate-700 border border-slate-700 cursor-pointer rounded-lg p-3 text-center transition-all group">
            <div className="text-xl mb-1">🖼️</div>
            <div className="text-[10px] text-slate-400 font-bold uppercase">Fotos</div>
            <input
              type="file"
              multiple
              accept="image/*"
              className="hidden"
              onChange={onSeleccionMultiple}
            />
          </label>
        </div>
      </div>
    </aside>
  );
}

export default LateralIzquierdo;
