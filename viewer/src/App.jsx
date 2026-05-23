import { useEffect, useRef, useState } from 'react'
import maplibregl from 'maplibre-gl'
import { Protocol } from 'pmtiles'
import 'maplibre-gl/dist/maplibre-gl.css'
import { TRAITS } from './traits'

// Register the pmtiles:// protocol once, before any map source uses it.
maplibregl.addProtocol('pmtiles', new Protocol().tile)

// e.g. https://<host>/dem.pmtiles  (set in viewer/.env)
const DEM_URL = import.meta.env.VITE_DEM_PMTILES_URL

// Basemap source-layers to keep — everything else is stripped, leaving just labels.
// 'place' = country/state/city/town names, 'water_name' = ocean/sea/lake names.
const LABEL_SOURCE_LAYERS = new Set(['place', 'water_name'])

// Dark theme colours.
const OCEAN = '#34383e' // globe surface where the DEM is transparent (oceans)
const SPACE = '#24262b' // sky / atmosphere behind the globe
const MIN_ZOOM = 2 // hard floor on zoom-out

export default function App() {
  const containerRef = useRef(null)
  const mapRef = useRef(null)
  const [traitId, setTraitId] = useState('') // '' = none selected

  useEffect(() => {
    if (mapRef.current) return

    const map = new maplibregl.Map({
      container: containerRef.current,
      // Free CARTO basemap; stripped to place labels only on load (see below).
      style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
      center: [0, 20],
      zoom: MIN_ZOOM,
      minZoom: MIN_ZOOM, // don't allow zooming out past this
    })
    mapRef.current = map

    // Trackpad/wheel zoom can transiently overshoot minZoom; snap it back.
    map.on('zoom', () => {
      if (map.getZoom() < MIN_ZOOM) map.setZoom(MIN_ZOOM)
    })

    map.on('style.load', () => {
      map.setProjection({ type: 'globe' }) // MapLibre GL JS v5+

      // Strip the basemap to place labels only, keeping the background layer.
      let firstLabelId, bgId
      for (const layer of map.getStyle().layers) {
        if (layer.type === 'background') {
          bgId = layer.id
          continue
        }
        if (layer.type === 'symbol' && LABEL_SOURCE_LAYERS.has(layer['source-layer'])) {
          firstLabelId ??= layer.id
          continue
        }
        map.removeLayer(layer.id)
      }

      // Dark theme: grey oceans + no atmosphere glow behind the globe.
      if (bgId) map.setPaintProperty(bgId, 'background-color', OCEAN)
      try {
        map.setSky?.({
          'sky-color': SPACE,
          'horizon-color': SPACE,
          'fog-color': SPACE,
          'atmosphere-blend': 0, // disable the bright atmospheric limb
        })
      } catch {
        /* older MapLibre without sky support */
      }

      // DEM sits above the dark background but below the labels.
      if (DEM_URL) {
        map.addSource('dem', {
          type: 'raster',
          url: `pmtiles://${DEM_URL}`,
          tileSize: 256,
        })
        map.addLayer(
          { id: 'dem', type: 'raster', source: 'dem', paint: { 'raster-resampling': 'nearest' } },
          firstLabelId,
        )
      } else {
        console.warn('VITE_DEM_PMTILES_URL not set — showing labels only.')
      }

      // Country outlines (admin level 2) as a thin white line, above the DEM.
      map.addLayer(
        {
          id: 'country-borders',
          type: 'line',
          source: 'carto',
          'source-layer': 'boundary',
          filter: ['all', ['==', 'admin_level', 2], ['==', 'maritime', 0]],
          paint: { 'line-color': '#ffffff', 'line-width': 0.8 },
        },
        firstLabelId,
      )
    })

    return () => {
      map.remove()
      mapRef.current = null
    }
  }, [])

  // Predictions aren't ready yet. When they are, each trait will have its own
  // PMTiles served by the worker (e.g. .../traits/<id>.pmtiles); swap the raster
  // source/layer + colormap for the selected trait here.
  useEffect(() => {
    if (!traitId) return
    // TODO: load the selected trait's prediction layer.
    console.log('selected trait:', traitId)
  }, [traitId])

  return (
    <>
      <div id="map" ref={containerRef} />
      {/* Trait selector — hidden until predictions are ready.
      <select
        className="trait-select"
        value={traitId}
        onChange={(e) => setTraitId(e.target.value)}
      >
        <option value="">Select a trait…</option>
        {TRAITS.map((t) => (
          <option key={t.id} value={t.id} title={t.long}>
            {t.short} · {t.id}
          </option>
        ))}
      </select>
      */}
    </>
  )
}
