# PTE v2 — prediction viewer

Minimal MapLibre + React globe for inspecting plant-trait prediction rasters.
Each layer is served as a single **PMTiles** file from **Cloudflare R2** (HTTP range
requests — no tile server needed).

## Stack

- React + Vite
- `maplibre-gl` v5 (globe projection)
- `pmtiles` (`pmtiles://` protocol)

## Develop

```bash
cd viewer
npm install
cp .env.example .env        # then set VITE_DEM_PMTILES_URL
npm run dev                 # http://localhost:5173
```

With `VITE_DEM_PMTILES_URL` empty you get the bare globe; set it to overlay the DEM.

## Data: GeoTIFF → PMTiles (DEM test)

`scripts/dem_to_pmtiles.py` does it all in pure Python (deps are in the repo's
`requirements.txt`: rio-tiler + pmtiles). It reprojects the raster to Web Mercator,
colour-maps it to RGBA (NaN NoData → transparent), and writes one `.pmtiles`.

```bash
# default: masked GLO-30 DEM -> dem.pmtiles, zooms 0-8, viridis, 0-6000 m, WEBP
python scripts/dem_to_pmtiles.py -o dem.pmtiles

# options: any raster, custom zooms / colormap / value range / lossless PNG
python scripts/dem_to_pmtiles.py path/to/raster.tif -o out.pmtiles \
    --max-zoom 7 --colormap terrain --vmin 0 --vmax 4000 --format png
```

Note: the GLO-30 DEM uses **NaN** as its NoData value, which value-based masking
can't detect — the script derives the mask from `isfinite` so ocean tiles are
dropped and coastlines stay transparent.

## Hosting: Cloudflare R2 (private bucket)

```bash
# upload the tiles to your bucket
npx wrangler r2 object put your-bucket/dem.pmtiles --file dem.pmtiles
```

The bucket stays **private**. A small Worker (`worker/`) bound to it serves objects
with HTTP Range + CORS, so the viewer fetches byte ranges via `pmtiles://` without
ever exposing the bucket. See `worker/README.md` to deploy, then set:

```
VITE_DEM_PMTILES_URL=https://ptev2-tiles.<your-subdomain>.workers.dev/dem.pmtiles
```

(Alternatively, for a quick public test you can enable the bucket's `r2.dev` URL or
attach a custom domain + a bucket CORS policy allowing the `range` header — but the
Worker keeps the bucket private and works for production too.)
