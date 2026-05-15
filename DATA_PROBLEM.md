# Data Format: Problem Analysis & Recommendations

## Current Setup

Chips are stored as zarr v3 stores, packed into `.zarr.zip` archives using zarr's `ZipStore`. Each split (`train`, `val`, `test`) is one zip file on LUSTRE (`/cluster/work/`).

Array layout per split:
- `predictors/{name}`: shape `(N, C, H, W)`, e.g. `modis` → `(29653, 72, 128, 128)`
- `targets/{name}`: shape `(N, T, H, W)`
- Chunk shape: `(1, C, H, W)` — one chip per chunk
- Codec: `ZstdCodec(level=0)` — minimal compression

---

## Why Training Is Slow

### 1. zarr v3 async overhead

zarr v3 uses an internal asyncio event loop for every array read. Even a simple `arr[i]` triggers an async pipeline:

```
arr[i]
  → sync() → asyncio event loop
    → _get_selection()
      → codec_pipeline.read()
        → concurrent_map(asyncio.gather(...))
          → ZipStore.get(chunk_key)
```

This overhead is significant per call. For a batch of 64 chips with 5 predictor arrays, that is **320 separate async event loop invocations** — one per chunk read. Measured on the cluster: ~36 seconds per batch, even from local NVMe (`$TMPDIR`).

### 2. One chunk per chip

With chunk shape `(1, C, H, W)`, reading a single chip always requires exactly one chunk read per array. There is no way to amortize the async overhead across samples — each sample in a batch triggers its own async pipeline independently via `__getitem__`.

### 3. LUSTRE is not the root cause

Moving data to local NVMe scratch (`$TMPDIR`) did not improve performance meaningfully. This confirms the bottleneck is zarr v3's async overhead, not storage I/O speed.

### 4. Multi-worker ZipStore bug (fixed)

An additional bug was present: `PlantTraitDataset.__init__` opened the zarr store to read metadata, caching it in `self._store`. When PyTorch forked DataLoader workers, each worker inherited the parent's open `ZipStore` file handle. Concurrent reads from the same handle caused `BadZipFile: Bad CRC-32` errors. **This is now fixed** — `__init__` opens a temporary store for metadata only (`del tmp`), so `self._store` is `None` at fork time and each worker opens its own handle.

---

## Recommended Fix: Convert to HDF5

Replace `.zarr.zip` stores with HDF5 (`.h5`) files, one per split.

**Why HDF5 solves the problem:**
- `h5py` is purely synchronous — zero async overhead per read
- Random access by integer index is O(1) and highly optimised
- Multiple DataLoader worker processes can safely open the same `.h5` file read-only (each gets its own file descriptor)
- Single large file on LUSTRE — no inode pressure, no unzipping
- No need to copy to `$TMPDIR` — large sequential chunk reads are what LUSTRE is optimised for

**Expected layout:**
```
data/1km/chips/patch128_stride64/
  train.h5
  val.h5
  test.h5
```

Each `.h5` file contains:
```
/predictors/canopy_height   (N, 2, 128, 128)  float32
/predictors/modis           (N, 72, 128, 128) float32
/predictors/soilgrids       (N, 61, 128, 128) float32
/predictors/vodca           (N, 9, 128, 128)  float32
/predictors/worldclim       (N, 6, 128, 128)  float32
/targets/splot              (N, T, 128, 128)  float32
/targets/gbif               (N, T, 128, 128)  float32
```

**Conversion:** a one-time script reads from the existing `.zarr.zip` files and writes `.h5` — no need to re-chip. The dataloader only needs minor changes: swap `zarr.open_group` for `h5py.File`.

---

## What Was Tried and Why It Did Not Work

| Approach | Outcome |
|---|---|
| Read `.zarr.zip` from LUSTRE (`/cluster/work/`) | Slow — zip random seek + zarr async overhead |
| Copy `.zarr.zip` to `$TMPDIR` (local NVMe) | No improvement — bottleneck is zarr v3 async, not I/O speed |
| Unzip to `/cluster/scratch/` | 830k files, took hours — LUSTRE is not designed for many small files |
| Lazy `_store` open per worker | Fixed `BadZipFile` CRC errors but did not fix speed |
| Increasing batch size (4 → 64) | Fewer batches but same per-sample cost — no meaningful improvement |
| `num_workers=0` | Avoids CRC errors but serialises all I/O — 2h/epoch |
