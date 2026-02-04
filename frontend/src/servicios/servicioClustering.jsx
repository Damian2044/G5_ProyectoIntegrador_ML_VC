import api from '../lib/axios'

const rutaClustering = '/clustering/'

export async function iniciarClusteringServicio({ k, tamanosMaximos, metodo, semilla }) {
  try {
    const cuerpo = {
      k,
      tamanosMaximos,
      metodo,
      semilla,
    }
    const { data } = await api.post(`${rutaClustering}iniciar`, cuerpo)
    return data
  } catch (error) {
    if (error.response && error.response.data) {
      throw error.response.data
    }
    throw error
  }
}

export async function agregarPuntoClusteringServicio({ sesionId, idFront, etiqueta, archivo }) {
  try {
    const formData = new FormData()
    formData.append('sesionId', sesionId)
    formData.append('idFront', idFront)
    if (etiqueta !== undefined && etiqueta !== null) {
      formData.append('etiqueta', etiqueta)
    }
    formData.append('imagen', archivo)

    const { data } = await api.post(`${rutaClustering}agregarPunto`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return data
  } catch (error) {
    if (error.response && error.response.data) {
      throw error.response.data
    }
    throw error
  }
}

export async function actualizarTamaniosClusteringServicio({ sesionId, tamanosNuevos }) {
  try {
    const cuerpo = { sesionId, tamanosNuevos }
    const { data } = await api.post(`${rutaClustering}actualizar-tamanios`, cuerpo)
    return data
  } catch (error) {
    if (error.response && error.response.data) {
      throw error.response.data
    }
    throw error
  }
}
