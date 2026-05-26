import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import maplibregl from 'maplibre-gl'
import { Protocol } from 'pmtiles'
import 'maplibre-gl/dist/maplibre-gl.css'
import { PREDICTED_TRAIT_IDS, TRAITS } from './traits'

// Register the pmtiles:// protocol once, before any map source uses it.
maplibregl.addProtocol('pmtiles', new Protocol().tile)

// Origin of the R2-backed tile worker. The .env value can be a full URL — we
// only use the origin and derive per-trait paths from it.
const PMTILES_URL = import.meta.env.VITE_PMTILES_URL
const PMTILES_BASE = PMTILES_URL ? new URL(PMTILES_URL).origin : null
const traitPMTilesUrl = (id) =>
  PMTILES_BASE && `${PMTILES_BASE}/predictions/${id.slice(1)}.pmtiles`
const RANGES_URL = PMTILES_BASE && `${PMTILES_BASE}/predictions/ranges.json`

// Trait selected on first paint.
const DEFAULT_TRAIT_ID = 'X3112'

// Dropdown options: traits that have a prediction source on disk, sorted alphabetically.
const SORTED_TRAITS = TRAITS.filter((t) => PREDICTED_TRAIT_IDS.has(t.id)).sort((a, b) =>
  a.display.localeCompare(b.display),
)

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

const fmt = (v) => (Math.abs(v) >= 100 || v === 0 ? v.toFixed(0) : v.toPrecision(3))

export default function App() {
  const containerRef = useRef(null)
  const mapRef = useRef(null)
  const selectRef = useRef(null)
  // Captured from the basemap on style.load; used as the beforeId when inserting
  // trait raster layers so they sit beneath the labels.
  const firstLabelIdRef = useRef(null)
  const [traitId, setTraitId] = useState(DEFAULT_TRAIT_ID)
  // Dropdown, bar, and ticks share this width — locked to the dropdown's widest
  // natural width so nothing reflows when the selection changes.
  const [legendWidth, setLegendWidth] = useState(undefined)
  // { X4: { vmin, vmax, colormap }, ... } from R2. null while loading, {} on failure.
  // Its keys are also the canonical "which traits are uploaded" signal.
  const [ranges, setRanges] = useState(null)
  // True after the basemap's style.load fires. Tracked as state (not just by
  // listening for the event inside the pre-create effect) so React schedules
  // pre-creation deterministically when *both* state and ranges are ready —
  // sidesteps the StrictMode-dev case where a `once('style.load', …)` listener
  // gets detached during a cleanup and the event then fires with no handler.
  const [styleLoaded, setStyleLoaded] = useState(false)
  const availableTraitIds = ranges && new Set(Object.keys(ranges))

  // --- Map init -------------------------------------------------------------
  useEffect(() => {
    if (mapRef.current) return

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
      center: [0, 20],
      zoom: INITIAL_ZOOM,
    })
    mapRef.current = map
    window.__map = map // devtools hook: __map.getStyle().layers, etc.
    console.log('[trait] map created')

    // Zoom around the map centre, not the cursor. On the globe, zooming towards
    // the cursor drifts the centre poleward (the point under an upper-canvas
    // cursor is at high latitude), which feels like the camera snapping north.
    map.scrollZoom.enable({ around: 'center' })
    map.touchZoomRotate.enable({ around: 'center' })

    // Surface every MapLibre error (including PMTiles 404s) instead of swallowing.
    map.on('error', (e) => console.warn('[trait] map error:', e?.error?.message ?? e))

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

      firstLabelIdRef.current = firstLabelId

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

      // Signal to the pre-create / visibility-toggle effects via React state.
      setStyleLoaded(true)
    })

    return () => {
      map.remove()
      mapRef.current = null
    }
  }, [])

  // --- Fetch ranges.json once on mount -------------------------------------
  useEffect(() => {
    console.log('[trait] ranges effect running, RANGES_URL=', RANGES_URL)
    if (!RANGES_URL) return
    fetch(RANGES_URL)
      .then((r) => {
        console.log('[trait] ranges response:', r.status, 'ok=', r.ok)
        return r.ok ? r.json() : Promise.reject(new Error(`ranges.json ${r.status}`))
      })
      .then((data) => {
        console.log('[trait] ranges parsed:', Object.keys(data).length, 'keys -> setRanges')
        setRanges(data)
      })
      .catch((err) => {
        console.warn('[trait] ranges.json unavailable:', err.message)
        setRanges({}) // loaded-but-empty so the dropdown stops "loading"
      })
  }, [])

  // --- Pre-create one hidden raster layer per available trait --------------
  // Runs once after ranges arrives + the basemap style is loaded. Each layer
  // gets a unique source/layer id (trait-X<n>) so MapLibre's per-source tile
  // cache cannot mix data across traits. Default trait starts visible.
  useEffect(() => {
    if (!ranges || !styleLoaded) return
    const map = mapRef.current
    if (!map || !PMTILES_BASE) return

    const ids = Object.keys(ranges)
    console.log(`[trait] ranges loaded — pre-creating ${ids.length} layers`)
    for (const id of ids) {
      const layerId = `trait-${id}`
      if (map.getSource(layerId)) continue // idempotent (StrictMode-safe)
      map.addSource(layerId, {
        type: 'raster',
        url: `pmtiles://${traitPMTilesUrl(id)}`,
        tileSize: 256,
      })
      map.addLayer(
        {
          id: layerId,
          type: 'raster',
          source: layerId,
          layout: { visibility: id === traitId ? 'visible' : 'none' },
          paint: { 'raster-resampling': 'nearest', 'raster-fade-duration': 200 },
        },
        firstLabelIdRef.current,
      )
    }
    console.log(`[trait] pre-created ${ids.length} hidden layers; visible = ${traitId}`)
  }, [ranges, styleLoaded])

  // --- Toggle visibility on selection change -------------------------------
  // One synchronous setLayoutProperty call per existing trait-* layer. No
  // add/remove, no header re-fetch, no blank flash. After the first visit
  // tiles for that trait stay in MapLibre's cache for instant re-show.
  useEffect(() => {
    if (!styleLoaded) return
    const map = mapRef.current
    if (!map) return
    const traitLayers = map.getStyle().layers.filter((l) => l.id.startsWith('trait-'))
    if (!traitLayers.length) return // pre-create hasn't run yet — its initial visibility will be correct
    console.log(`[trait] selection → ${traitId}`)
    for (const layer of traitLayers) {
      const visible = layer.id === `trait-${traitId}`
      map.setLayoutProperty(layer.id, 'visibility', visible ? 'visible' : 'none')
    }
  }, [traitId, styleLoaded])

  const selected = TRAITS.find((t) => t.id === traitId)
  const range = ranges?.[traitId]
  const stops = COLORMAP_STOPS[range?.colormap ?? 'viridis'] ?? COLORMAP_STOPS.viridis

  // --- Lock the dropdown/bar width to the widest option text ---------------
  useLayoutEffect(() => {
    const el = selectRef.current
    if (!el) return
    const prevValue = el.value
    const prevStyle = el.style.width
    el.style.width = 'auto'
    let max = 0
    for (const t of SORTED_TRAITS) {
      el.value = t.id
      if (el.offsetWidth > max) max = el.offsetWidth
    }
    el.value = prevValue
    el.style.width = prevStyle
    setLegendWidth(Math.ceil(max * 1.05))
  }, [])

  return (
    <>
      <div id="map" ref={containerRef} />
      {selected && (
        <div className="legend">
          <div className="legend-title">
            <select
              ref={selectRef}
              className="legend-select"
              style={{ width: legendWidth }}
              value={traitId}
              onChange={(e) => setTraitId(e.target.value)}
            >
              {SORTED_TRAITS.map((t) => (
                <option
                  key={t.id}
                  value={t.id}
                  title={t.long}
                  disabled={availableTraitIds ? !availableTraitIds.has(t.id) : false}
                >
                  {t.display}
                </option>
              ))}
            </select>
          </div>
          {range ? (
            <>
              <div
                className="legend-bar"
                style={{
                  width: legendWidth,
                  background: `linear-gradient(to right, ${stops.join(', ')})`,
                }}
              />
              <div className="legend-ticks" style={{ width: legendWidth }}>
                <span>{fmt(range.vmin)}</span>
                <span className="legend-unit">{selected.unit}</span>
                <span>{fmt(range.vmax)}</span>
              </div>
            </>
          ) : (
            <div
              className="legend-ticks"
              style={{ width: legendWidth, justifyContent: 'center' }}
            >
              <span className="legend-unit">{selected.unit}</span>
            </div>
          )}
        </div>
      )}
    </>
  )
}
