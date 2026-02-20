import React from 'react';
import { obtenerPaletaClusters } from '../lib/paletaClusters';

const PuntoCluster = ({ x, y, color, esCentroide }) => (
  <div
    className={`absolute shadow-lg transition-all duration-500 ease-out flex items-center justify-center border-2 border-white ${
      esCentroide ? 'w-6 h-6 rounded-none text-[10px] font-bold' : 'w-3 h-3 rounded-full'
    }`}
    style={{
      left: `${x}%`,
      top: `${y}%`,
      backgroundColor: color,
      color: esCentroide ? 'white' : 'transparent',
      transform: 'translate(-50%, -50%)',
    }}
  >
    {esCentroide ? '✕' : null}
  </div>
);

function normalizarCoordenadas(puntos, centroides) {
  const todos = [];
  puntos.forEach((p) => {
    if (p.x != null && p.y != null) {
      todos.push({ x: p.x, y: p.y });
    }
  });
  (centroides || []).forEach((c) => {
    if (c.x != null && c.y != null) {
      todos.push({ x: c.x, y: c.y });
    }
  });

  if (todos.length === 0) {
    return { puntosNorm: [], centroidesNorm: [] };
  }

  let minX = todos[0].x;
  let maxX = todos[0].x;
  let minY = todos[0].y;
  let maxY = todos[0].y;

  todos.forEach((t) => {
    if (t.x < minX) minX = t.x;
    if (t.x > maxX) maxX = t.x;
    if (t.y < minY) minY = t.y;
    if (t.y > maxY) maxY = t.y;
  });

  const rangoX = maxX - minX || 1;
  const rangoY = maxY - minY || 1;

  const escalar = (x, y) => {
    const nx = ((x - minX) / rangoX) * 80 + 10; // margen 10-90%
    const ny = ((y - minY) / rangoY) * 80 + 10;
    return { x: nx, y: ny };
  };

  const puntosNorm = puntos
    .filter((p) => p.x != null && p.y != null)
    .map((p) => {
      const { x, y } = escalar(p.x, p.y);
      return { ...p, x, y };
    });

  const centroidesNorm = (centroides || [])
    .filter((c) => c.x != null && c.y != null)
    .map((c) => {
      const { x, y } = escalar(c.x, c.y);
      return { ...c, x, y };
    });

  return { puntosNorm, centroidesNorm };
}

function VisualizacionClustering({ proyeccionPCA, cantidadClusters = 1 }) {
  const [abierto, setAbierto] = React.useState(true);

  if (!proyeccionPCA || !Array.isArray(proyeccionPCA.puntos) || proyeccionPCA.puntos.length === 0) {
    return null;
  }

  const { puntos, centroides } = proyeccionPCA;
  const { puntosNorm, centroidesNorm } = normalizarCoordenadas(puntos, centroides);
  const paleta = obtenerPaletaClusters(cantidadClusters);

  return (
    <section className="bg-slate-900/60 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
      <div
        className="h-14 bg-slate-800/60 border-b border-slate-800 flex items-center px-4 justify-between cursor-pointer"
        onClick={() => setAbierto((v) => !v)}
      >
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              setAbierto((prev) => !prev);
            }}
            className="w-6 h-6 flex items-center justify-center rounded hover:bg-slate-700/80 transition-colors text-slate-300"
            title={abierto ? 'Cerrar visualización PCA' : 'Abrir visualización PCA'}
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
          <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-cyan-500/10 text-cyan-400 text-xs font-bold">
            2D
          </span>
          <h2 className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-300">
            Proyección PCA 2D (ilustrativa)
          </h2>
        </div>
        <span className="text-[10px] text-slate-500">{puntosNorm.length} puntos</span>
      </div>

      {abierto && (
        <div className="p-4 flex flex-col gap-3">
          <div className="relative w-full h-64 bg-slate-950/80 rounded-xl overflow-hidden border border-slate-800">
            <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-800/60" />
            <div className="absolute top-1/2 left-0 right-0 h-px bg-slate-800/60" />

            {puntosNorm.map((p) => {
              const color = paleta[(p.cluster ?? 0) % paleta.length];
              return <PuntoCluster key={p.idFront} x={p.x} y={p.y} color={color} esCentroide={false} />;
            })}

            {centroidesNorm.map((c) => {
              const color = paleta[(c.cluster ?? 0) % paleta.length];
              return <PuntoCluster key={`centroide-${c.cluster}`} x={c.x} y={c.y} color={color} esCentroide />;
            })}
          </div>

          <div className="flex flex-wrap gap-2">
            {paleta.map((color, indice) => (
              <div
                key={`leyenda-${indice}`}
                className="inline-flex items-center gap-1.5 bg-slate-900/70 border border-slate-700 rounded-full px-2 py-1"
              >
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
                <span className="text-[10px] text-slate-300 font-mono">C{indice}</span>
              </div>
            ))}
          </div>

          <p className="text-[10px] text-slate-500">
            Esta vista es una proyección PCA incremental a 2 dimensiones de los vectores reales; solo se usa para
            visualizar, el clustering se hace siempre en el espacio completo.
          </p>
        </div>
      )}
    </section>
  );
}

export default VisualizacionClustering;
