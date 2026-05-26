import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import maplibregl from 'maplibre-gl'
import { Protocol } from 'pmtiles'
import 'maplibre-gl/dist/maplibre-gl.css'
import { PREDICTED_TRAIT_IDS, TRAITS } from './traits'

// Dropdown options: only traits with a prediction source on disk, sorted alphabetically.
const SORTED_TRAITS = TRAITS.filter((t) => PREDICTED_TRAIT_IDS.has(t.id)).sort((a, b) =>
  a.display.localeCompare(b.display),
)

// Register the pmtiles:// protocol once, before any map source uses it.
maplibregl.addProtocol('pmtiles', new Protocol().tile)

// e.g. https://<host>/predictions/4.pmtiles  (set in viewer/.env). We only use the
// origin; per-trait URLs are derived below.
const PMTILES_URL = import.meta.env.VITE_PMTILES_URL
const PMTILES_BASE = PMTILES_URL ? new URL(PMTILES_URL).origin : null
const traitPMTilesUrl = (id) =>
  PMTILES_BASE && `${PMTILES_BASE}/predictions/${id.slice(1)}.pmtiles`

// Trait selected by default on load. Per-trait vmin/vmax/colormap come from
// predictions/ranges.json on R2 (written by viewer/scripts/predictor_to_pmtiles.py).
const DEFAULT_TRAIT_ID = 'X3106'

// Colormap stops for the legend gradient — keep in sync with the matplotlib colormaps
// the script supports. Currently every trait is tiled with viridis.
const COLORMAP_STOPS = {
  viridis: ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'],
}

// Basemap source-layers to keep — everything else is stripped, leaving just labels.
// 'place' = country/state/city/town names, 'water_name' = ocean/sea/lake names.
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
  // Captured from the basemap on style.load so the trait-swap effect can re-insert
  // the raster layer in the right z-order without re-querying the style every time.
  const firstLabelIdRef = useRef(null)
  // Currently selected trait — drives both the map raster URL and the legend.
  const [traitId, setTraitId] = useState(DEFAULT_TRAIT_ID)
  // Dropdown, bar, and ticks share this width (5% above the dropdown's natural size).
  const [legendWidth, setLegendWidth] = useState(undefined)
  // { X4: { vmin, vmax, colormap }, ... } — sidecar manifest from R2. Null while
  // loading, then either the loaded object or {} on failure. Its keys are the
  // canonical "which traits are uploaded" signal — see availableTraitIds below.
  const [ranges, setRanges] = useState(null)
  const availableTraitIds = ranges && new Set(Object.keys(ranges))

  useEffect(() => {
    if (mapRef.current) return

    const map = new maplibregl.Map({
      container: containerRef.current,
      // Free CARTO basemap; stripped to place labels only on load (see below).
      style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
      center: [0, 20],
      zoom: INITIAL_ZOOM,
    })
    mapRef.current = map
    window.__map = map // debug: inspect via devtools (__map.getStyle().layers etc.)

    // Zoom around the map centre, not the cursor. On the globe, zooming towards
    // the cursor drifts the centre poleward (the point under an upper-canvas
    // cursor is at high latitude), which feels like the camera snapping north.
    map.scrollZoom.enable({ around: 'center' })
    map.touchZoomRotate.enable({ around: 'center' })

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

      // Expose the label anchor to the trait-swap effect so it can re-insert the
      // raster layer beneath the labels each time the selection changes.
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
    })

    return () => {
      map.remove()
      mapRef.current = null
    }
  }, [])

  // Load the colour-ranges sidecar once on mount. Its keys also tell us which
  // traits have .pmtiles on R2 (the tiling script writes ranges.json atomically
  // after every successful tile), so we don't need a separate listing endpoint.
  useEffect(() => {
    if (!PMTILES_BASE) return
    fetch(`${PMTILES_BASE}/predictions/ranges.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`ranges.json ${r.status}`))))
      .then(setRanges)
      .catch((err) => {
        console.warn('ranges.json unavailable:', err.message)
        setRanges({}) // mark as loaded-but-empty so the dropdown stops "loading"
      })
  }, [])

  // Swap the trait raster on selection change. Force-removes any existing trait
  // layer/source and adds a brand-new one keyed by traitId — using a fresh
  // source id (not the shared 'trait' name) sidesteps MapLibre's internal
  // tile cache, which is keyed by source id and would otherwise serve stale
  // tiles from the previous selection.
  useEffect(() => {
    const map = mapRef.current
    if (!map || !PMTILES_BASE) return

    const id = `trait-${traitId}`
    const url = `pmtiles://${traitPMTilesUrl(traitId)}`
    console.log('[trait] swap requested ->', id, url)

    const setupTrait = () => {
      const before = map.getStyle().layers.map((l) => l.id)
      console.log('[trait] setupTrait running. layers before:', before)
      // Defensive cleanup: drop anything trait-related from prior runs (covers
      // both this new per-trait scheme and the legacy single 'trait' id).
      for (const layer of [...map.getStyle().layers]) {
        if (layer.id === 'trait' || layer.id.startsWith('trait-')) {
          console.log('[trait] removing layer', layer.id)
          map.removeLayer(layer.id)
        }
      }
      for (const srcId of Object.keys(map.getStyle().sources)) {
        if (srcId === 'trait' || srcId.startsWith('trait-')) {
          console.log('[trait] removing source', srcId)
          map.removeSource(srcId)
        }
      }

      map.addSource(id, {
        type: 'raster',
        url,
        tileSize: 256,
      })
      map.addLayer(
        {
          id,
          type: 'raster',
          source: id,
          paint: { 'raster-resampling': 'nearest', 'raster-fade-duration': 200 },
        },
        firstLabelIdRef.current, // undefined => append on top, still fine
      )
      console.log(
        '[trait] added',
        id,
        '   layers after:',
        map.getStyle().layers.map((l) => l.id),
      )
    }

    // Only gate on style.load. firstLabelIdRef is purely a z-order hint —
    // undefined just means "put it on top", which renders correctly.
    if (map.isStyleLoaded()) {
      console.log('[trait] style already loaded -> setupTrait now')
      setupTrait()
    } else {
      console.log('[trait] waiting on style.load')
      map.once('style.load', setupTrait)
    }

    return () => map.off('style.load', setupTrait)
  }, [traitId])

  const selected = TRAITS.find((t) => t.id === traitId)
  const range = ranges?.[traitId]
  const colormap = range?.colormap ?? 'viridis'
  const stops = COLORMAP_STOPS[colormap] ?? COLORMAP_STOPS.viridis

  useLayoutEffect(() => {
    const el = selectRef.current
    if (!el) return
    // Measure the widest possible selection by cycling through every trait id,
    // then lock the dropdown/bar/ticks to that width (+5%) so they never resize
    // when the user changes the selection.
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
            <div className="legend-ticks" style={{ width: legendWidth, justifyContent: 'center' }}>
              <span className="legend-unit">{selected.unit}</span>
            </div>
          )}
        </div>
      )}
    </>
  )
}
