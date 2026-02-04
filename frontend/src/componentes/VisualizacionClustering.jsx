import React from 'react';

const COLORES_CLUSTER = [
  '#06b6d4',
  '#ec4899',
  '#eab308',
  '#8b5cf6',
  '#10b981',
  '#f97316',
  '#ef4444',
  '#6366f1',
];

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

function VisualizacionClustering({ proyeccionPCA }) {
  if (!proyeccionPCA || !Array.isArray(proyeccionPCA.puntos) || proyeccionPCA.puntos.length === 0) {
    return null;
  }

  const { puntos, centroides } = proyeccionPCA;
  const { puntosNorm, centroidesNorm } = normalizarCoordenadas(puntos, centroides);

  return (
    <section className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4 shadow-xl flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-cyan-500/10 text-cyan-400 text-xs font-bold">
            2D
          </span>
          <h2 className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-300">
            Proyección PCA 2D (ilustrativa)
          </h2>
        </div>
        <span className="text-[10px] text-slate-500">{puntosNorm.length} puntos</span>
      </div>

      <div className="relative w-full h-64 bg-slate-950/80 rounded-xl overflow-hidden border border-slate-800">
        {/* Ejes simples */}
        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-800/60" />
        <div className="absolute top-1/2 left-0 right-0 h-px bg-slate-800/60" />

        {puntosNorm.map((p) => {
          const color = COLORES_CLUSTER[(p.cluster ?? 0) % COLORES_CLUSTER.length];
          return <PuntoCluster key={p.idFront} x={p.x} y={p.y} color={color} esCentroide={false} />;
        })}

        {centroidesNorm.map((c) => {
          const color = COLORES_CLUSTER[(c.cluster ?? 0) % COLORES_CLUSTER.length];
          return <PuntoCluster key={`centroide-${c.cluster}`} x={c.x} y={c.y} color={color} esCentroide />;
        })}
      </div>

      <p className="text-[10px] text-slate-500">
        Esta vista es una proyección PCA incremental a 2 dimensiones de los vectores reales; solo se usa para
        visualizar, el clustering se hace siempre en el espacio completo.
      </p>
    </section>
  );
}

export default VisualizacionClustering;
