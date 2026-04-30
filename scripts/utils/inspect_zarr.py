"""Inspect the metadata and array structure of split zarr stores in a chips directory."""

from pathlib import Path

import zarr
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree

ZARR_DIR = "/scratch3/plant-traits-v2/data/1km/chips/patch256_stride256/"
SPLITS = ["train", "val", "test"]

console = Console()

console.print(f"[bold]{ZARR_DIR}[/bold]\n")

for split in SPLITS:
    path = Path(ZARR_DIR) / f"{split}.zarr"
    try:
        z = zarr.open_group(str(path), mode="r")
    except Exception:
        console.print(f"[red][{split}] not found at {path}[/red]\n")
        continue

    tree = Tree(f"[bold cyan]{split}.zarr[/bold cyan]")

    for k, v in z.attrs.items():
        tree.add(f"[yellow]{k}[/yellow]: {v}")

    for group_name, group in z.groups():
        group_branch = tree.add(f"[bold green]{group_name}/[/bold green]")
        for k, v in group.attrs.items():
            if isinstance(v, list) and len(v) > 5:
                group_branch.add(f"[yellow]{k}[/yellow]: {v[:5]} … ({len(v)} total)")
            else:
                group_branch.add(f"[yellow]{k}[/yellow]: {v}")
        for arr_name, arr in group.arrays():
            arr_branch = group_branch.add(
                f"[bold]{arr_name}[/bold]  "
                f"shape={arr.shape}  dtype={arr.dtype}  chunks={arr.chunks}"
            )
            for k, v in arr.attrs.items():
                if k == "files" and isinstance(v, list):
                    files_branch = arr_branch.add(
                        f"[yellow]files[/yellow] ({len(v)} total)"
                    )
                    for f in v:
                        files_branch.add(f"[dim]{f}[/dim]")
                elif isinstance(v, list) and len(v) > 5:
                    arr_branch.add(f"[yellow]{k}[/yellow]: {v[:5]} … ({len(v)} total)")
                else:
                    arr_branch.add(f"[yellow]{k}[/yellow]: {v}")

    if "bounds" in z:
        bounds = z["bounds"]
        tree.add(f"[bold]bounds[/bold]  shape={bounds.shape}  dtype={bounds.dtype}")

    console.print(Panel(tree, border_style="dim"))
    console.print()
