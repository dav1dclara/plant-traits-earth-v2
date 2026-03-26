import zarr

ZARR_DIR = "/scratch3/plant-traits-v2/data/22km/chips/patch25_stride20/"
SPLITS = ["train", "val", "test"]

for split in SPLITS:
    path = f"{ZARR_DIR}/{split}.zarr"
    try:
        z = zarr.open_group(path, mode="r")
    except Exception:
        print(f"[{split}] not found at {path}\n")
        continue

    print(f"{'=' * 50}")
    print(f"  {split}.zarr")
    print(f"{'=' * 50}")
    print(f"  split:         {z.attrs.get('split')}")
    print(f"  crs_epsg:      EPSG:{z.attrs.get('crs_epsg')}")
    print(f"  res_km:        {z.attrs.get('res_km')}")
    print(f"  patch_size:    {z.attrs.get('patch_size')}")
    print(f"  stride:        {z.attrs.get('stride')}")
    print(f"  creation_date: {z.attrs.get('creation_date')}")
    print(f"  transform:     {z.attrs.get('transform')}")

    print("\n  Arrays:")
    for group_name, group in z.groups():
        band_names = group.attrs.get("band_names")
        if band_names:
            print(f"    {group_name}/ band_names: {band_names}")
        for arr_name, arr in group.arrays():
            files = arr.attrs.get("files", [])
            print(
                f"    {group_name}/{arr_name}: shape={arr.shape}, dtype={arr.dtype}, chunks={arr.chunks}"
            )
            if files:
                print(f"      files ({len(files)} total):")
                for f in files[:5]:
                    print(f"        {f}")
                if len(files) > 5:
                    print(f"        ...")

    bounds = z["bounds"]
    print(f"\n  bounds: shape={bounds.shape}, dtype={bounds.dtype}")
    print()
