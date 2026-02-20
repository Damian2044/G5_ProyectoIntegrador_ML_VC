export function generarColorCluster(indice, totalClusters = 1) {
  const total = Math.max(1, Number(totalClusters) || 1)
  const idx = Math.max(0, Number(indice) || 0)
  const hue = Math.round((idx * 360) / total) % 360
  return `hsl(${hue} 78% 55%)`
}

export function obtenerPaletaClusters(totalClusters = 1) {
  const total = Math.max(1, Number(totalClusters) || 1)
  return Array.from({ length: total }, (_, i) => generarColorCluster(i, total))
}