import React from 'react';

function BarraSeleccionFlotante({
  cantidadSeleccionados,
  textoEtiqueta,
  onCambiarTextoEtiqueta,
  onAplicarEtiquetaMasiva,
  onBorrarSeleccion,
  onLimpiarSeleccion,
}) {
  return (
    <div
      className={`fixed top-24 left-1/2 -translate-x-1/2 z-50 transition-all duration-300 transform ${
        cantidadSeleccionados > 0
          ? 'translate-y-0 opacity-100'
          : 'translate-y-[-20px] opacity-0 pointer-events-none'
      }`}
    >
      <div className="bg-slate-800/90 backdrop-blur-md border border-slate-600 rounded-xl p-4 flex items-center gap-4 shadow-2xl w-[90vw] max-w-2xl">
        <span className="bg-cyan-500/10 text-cyan-400 text-sm font-bold px-3 py-1.5 rounded-lg whitespace-nowrap">
          {cantidadSeleccionados} items
        </span>
        <input
          type="text"
          placeholder="Nueva etiqueta..."
          value={textoEtiqueta}
          onChange={(e) => onCambiarTextoEtiqueta(e.target.value)}
          className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:border-cyan-500 outline-none"
        />
        <button
          onClick={onAplicarEtiquetaMasiva}
          className="bg-cyan-600 hover:bg-cyan-500 text-white px-4 py-2 rounded-lg text-sm font-bold transition-colors"
        >
          Aplicar
        </button>
        <div className="h-6 w-px bg-slate-600 mx-1" />
        <button
          onClick={onBorrarSeleccion}
          className="bg-red-500/10 hover:bg-red-500 hover:text-white text-red-400 border border-red-500/30 px-3 py-2 rounded-lg text-sm font-bold transition-colors flex items-center gap-2"
        >
          Borrar
        </button>
        <button
          onClick={onLimpiarSeleccion}
          className="text-slate-400 hover:text-white px-2 text-sm font-medium"
        >
          ✕
        </button>
      </div>
    </div>
  );
}

export default BarraSeleccionFlotante;
