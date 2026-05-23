# tiles worker

Cloudflare Worker that serves objects from a **private** R2 bucket with HTTP Range
+ CORS, so the viewer can fetch PMTiles byte ranges. The bucket stays private —
no public-access toggle, no r2.dev URL.

## Deploy

```bash
cd viewer/worker
# 1. set your bucket name
#    edit wrangler.toml -> bucket_name = "your-bucket"
npx wrangler login          # one-time browser auth
npx wrangler deploy         # prints https://ptev2-tiles.<your-subdomain>.workers.dev
```

First deploy may ask you to register a `*.workers.dev` subdomain — accept it.

## Verify (Range request works)

```bash
curl -sI -H 'Range: bytes=0-99' \
  https://ptev2-tiles.<your-subdomain>.workers.dev/dem.pmtiles
# expect: HTTP/2 206, content-range: bytes 0-99/<size>, access-control-allow-origin: *
```

## Point the viewer at it

In `viewer/.env`:

```
VITE_DEM_PMTILES_URL=https://ptev2-tiles.<your-subdomain>.workers.dev/dem.pmtiles
```

The path after the host is the R2 object key, so `/dem.pmtiles` serves the
`dem.pmtiles` object (use the prefix you uploaded under, if any).

## Notes

- `access-control-allow-origin` is `*` in `src/index.js`. The Worker URL is
  reachable by anyone who has it; CORS only controls which browser origins may
  read it. To restrict, replace `*` with your origin, or add a token/referer check.
- Edge-cached for 1h (`cache-control`). Bump it once tiles are final.
