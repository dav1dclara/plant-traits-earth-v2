import { useEffect, useRef, useState } from 'react'
import maplibregl from 'maplibre-gl'
import { Protocol } from 'pmtiles'
import 'maplibre-gl/dist/maplibre-gl.css'
import { TRAITS } from './traits'

// Register the pmtiles:// protocol once, before any map source uses it.
maplibregl.addProtocol('pmtiles', new Protocol().tile)

// Origin of the R2-backed tile worker (path part of the env var is ignored).
const PMTILES_URL = import.meta.env.VITE_PMTILES_URL
const PMTILES_BASE = PMTILES_URL ? new URL(PMTILES_URL).origin : null

// Single trait this build displays. PMTiles + colour range come from R2.
const TRAIT_ID = 'X3112'
const TRAIT_PMTILES_URL = PMTILES_BASE && `${PMTILES_BASE}/predictions/${TRAIT_ID.slice(1)}.pmtiles`
const RANGES_URL = PMTILES_BASE && `${PMTILES_BASE}/predictions/ranges.json`

// Colormap stops for the legend gradient — keep in sync with the matplotlib
// colormaps the tiling script supports. Currently every trait is tiled with viridis.
const COLORMAP_STOPS = {
  viridis: ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'],
}

// Basemap source-layers to keep — everything else is stripped, leaving just labels.
const LABEL_SOURCE_LAYERS = new Set(['place', 'water_name'])

// Dark theme colours.
const OCEAN = '#34383e' // globe surface where the predictor raster is transparent (oceans)
const SPACE = '#24262b' // sky / atmosphere behind the globe
const INITIAL_ZOOM = 2 // whole-globe view on load
const LEGEND_BAR_WIDTH = 280 // fixed colorbar width in px

const fmt = (v) => (Math.abs(v) >= 100 || v === 0 ? v.toFixed(0) : v.toPrecision(3))

export default function App() {
  const containerRef = useRef(null)
  const mapRef = useRef(null)
  // { vmin, vmax, colormap } for TRAIT_ID, fetched from R2 ranges.json.
  const [range, setRange] = useState(null)

  useEffect(() => {
    if (mapRef.current) return

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
      center: [0, 20],
      zoom: INITIAL_ZOOM,
    })
    mapRef.current = map

    // Zoom around the map centre, not the cursor. On the globe, zooming towards
    // the cursor drifts the centre poleward (the point under an upper-canvas
    // cursor is at high latitude), which feels like the camera snapping north.
    map.scrollZoom.enable({ around: 'center' })
    map.touchZoomRotate.enable({ around: 'center' })

    map.on('style.load', () => {
      map.setProjection({ type: 'globe' })

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
          'atmosphere-blend': 0,
        })
      } catch {
        /* older MapLibre without sky support */
      }

      // Trait raster sits above the dark background but below the labels.
      if (TRAIT_PMTILES_URL) {
        map.addSource('trait', {
          type: 'raster',
          url: `pmtiles://${TRAIT_PMTILES_URL}`,
          tileSize: 256,
        })
        map.addLayer(
          {
            id: 'trait',
            type: 'raster',
            source: 'trait',
            paint: { 'raster-resampling': 'nearest' },
          },
          firstLabelId,
        )
      } else {
        console.warn('VITE_PMTILES_URL not set — showing labels only.')
      }

      // Country outlines (admin level 2) as a thin white line, above the trait raster.
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

  // One-shot fetch of the colour range for this trait.
  useEffect(() => {
    if (!RANGES_URL) return
    fetch(RANGES_URL)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`ranges.json ${r.status}`))))
      .then((data) => setRange(data[TRAIT_ID] ?? null))
      .catch((err) => console.warn('ranges.json unavailable:', err.message))
  }, [])

  const selected = TRAITS.find((t) => t.id === TRAIT_ID)
  const stops = COLORMAP_STOPS[range?.colormap ?? 'viridis'] ?? COLORMAP_STOPS.viridis

  return (
    <>
      <div id="map" ref={containerRef} />
      {selected && (
        <div className="legend">
          <div className="legend-title">{selected.display}</div>
          {range ? (
            <>
              <div
                className="legend-bar"
                style={{
                  width: LEGEND_BAR_WIDTH,
                  background: `linear-gradient(to right, ${stops.join(', ')})`,
                }}
              />
              <div className="legend-ticks" style={{ width: LEGEND_BAR_WIDTH }}>
                <span>{fmt(range.vmin)}</span>
                <span className="legend-unit">{selected.unit}</span>
                <span>{fmt(range.vmax)}</span>
              </div>
            </>
          ) : (
            <div
              className="legend-ticks"
              style={{ width: LEGEND_BAR_WIDTH, justifyContent: 'center' }}
            >
              <span className="legend-unit">{selected.unit}</span>
            </div>
          )}
        </div>
      )}
    </>
  )
}
