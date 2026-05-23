# plant-traits-earth-v2 — viewer

A minimal MapLibre **globe** for inspecting plant-trait prediction rasters. Each
layer is a single **PMTiles** file served from a **private Cloudflare R2** bucket
(via a small Worker) and fetched with HTTP range requests — no tile server.

Right now it shows the GLO-30 **DEM** as a test layer while the predictions are
being finalised.

## What it renders

- Dark globe (MapLibre v5 globe projection): grey oceans, no atmosphere glow
- The DEM raster (viridis, baked at tiling time)
- Thin white **country outlines** (admin level 2)
- **Place labels** on top (countries / cities / oceans)
- Min zoom 2 (can't zoom out past the globe)
- A trait dropdown is scaffolded in `App.jsx` / `traits.js` but **commented out**
  until per-trait predictions exist.

## Stack

- React + Vite
- `maplibre-gl` v5 (globe projection)
- `pmtiles` (`pmtiles://` protocol)
- CARTO Dark Matter basemap, stripped to just label layers

## Develop

```bash
cd viewer
npm install
cp .env.example .env        # then set VITE_DEM_PMTILES_URL
npm run dev                 # http://localhost:5173
```

With `VITE_DEM_PMTILES_URL` empty you get the bare globe; set it to overlay the DEM.

## Data: GeoTIFF → PMTiles

`scripts/dem_to_pmtiles.py` does it all in pure Python (deps in the repo's
`requirements.txt`: rio-tiler + pmtiles). It reprojects the raster to Web Mercator,
colour-maps it to RGBA (NaN NoData → transparent), and writes one `.pmtiles`.

```bash
# default: masked GLO-30 DEM -> dem.pmtiles, zooms 0-8, viridis, 0-6000 m, WEBP
python scripts/dem_to_pmtiles.py -o dem.pmtiles

# options: any raster, custom zooms / colormap / value range / lossless PNG
python scripts/dem_to_pmtiles.py path/to/raster.tif -o out.pmtiles \
    --max-zoom 7 --colormap terrain --vmin 0 --vmax 4000 --format png
```

Notes:
- The GLO-30 DEM uses **NaN** as NoData, which value-based masking can't detect —
  the script derives the mask from `isfinite`, so ocean tiles are dropped and
  coastlines stay transparent.
- It builds overviews on a temp copy first, so low-zoom tiles are fast.

## Hosting: private Cloudflare R2 + Worker

The bucket stays **private**. A small Worker (`worker/`) bound to it serves objects
with HTTP Range + CORS, so the viewer reads byte ranges via `pmtiles://` without
exposing the bucket. See `worker/README.md` to deploy.

```bash
# upload tiles (object key dem/dem.pmtiles in bucket plant-traits-v2)
cd viewer/worker
npx wrangler r2 object put plant-traits-v2/dem/dem.pmtiles --file ../dem.pmtiles --remote
```

Then point the viewer at the Worker URL in `.env`:

```
VITE_DEM_PMTILES_URL=https://ptev2-tiles.davidclara.workers.dev/dem/dem.pmtiles
```

(For a quick public test you could instead enable the bucket's `r2.dev` URL or a
custom domain + a bucket CORS policy allowing the `range` header — but the Worker
keeps the bucket private and works for production.)

## Deploy: GitHub Pages

Pushing to `main` (changes under `viewer/**`) triggers
`.github/workflows/deploy.yml`, which builds `viewer/` and publishes it to:

**https://dav1dclara.github.io/plant-traits-earth-v2/**

One-time setup: repo **Settings → Pages → Source: GitHub Actions**.

Details:
- `vite.config.js` sets `base: '/plant-traits-earth-v2/'` for production builds
  (project Pages serve under `/<repo>/`); dev stays at `/`.
- `VITE_DEM_PMTILES_URL` is baked in at build from the workflow `env` (the public
  Worker URL). The Worker's `*` CORS already allows the `github.io` origin.
- If you rename the repo or add a custom domain, update `base` to match.
