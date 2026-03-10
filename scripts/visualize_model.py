#!/usr/bin/env python3
"""
Visualisiert die Modell-Architektur und speichert sie als Bild.
"""

import torch
from torchview import draw_graph
from ptev2.models.baseline import PadarianInspiredPatchCNN
from ptev2.models.traitPatchCNN import ResPatchCenterCNN


def visualize_and_save(
    model: torch.nn.Module,
    input_shape: tuple,
    save_path: str = "model_architecture",
    format: str = "png",
    depth: int = 3,
):
    """
    Visualisiert ein PyTorch-Modell und speichert es als Bild.

    Args:
        model: PyTorch-Modell
        input_shape: Shape des Dummy-Inputs (B, C, H, W)
        save_path: Pfad ohne Dateiendung
        format: Bildformat ('png', 'pdf', 'svg', etc.)
        depth: Visualisierungstiefe (höher = detaillierter)
    """
    model.eval()

    print(f"Erstelle Visualisierung von {model.__class__.__name__}...")
    print(f"Input Shape: {input_shape}")

    # Erstelle Visualisierung
    model_graph = draw_graph(
        model,
        input_size=input_shape,
        device="cpu",
        depth=depth,
        expand_nested=True,
        save_graph=True,
        filename=save_path,
        directory=".",
    )

    # Speichere als gewünschtes Format
    output_file = f"{save_path}.{format}"
    model_graph.visual_graph.render(
        filename=save_path,
        format=format,
        cleanup=True,
    )

    print(f"✓ Visualisierung gespeichert: {output_file}")
    print(f"  - Format: {format}")
    print(f"  - Depth: {depth}")

    return model_graph


if __name__ == "__main__":
    # Beispiel: PadarianInspiredPatchCNN visualisieren

    print("=" * 60)
    print("Visualisiere: PadarianInspiredPatchCNN (single head)")
    print("=" * 60)

    model_single = PadarianInspiredPatchCNN(
        in_channels=10,  # z.B. 10 Sentinel-2 Bänder
        n_traits=31,
        hidden_dim=64,
        dropout_p=0.3,
        head_type="single",
    )

    # Single head model
    visualize_and_save(
        model_single,
        input_shape=(1, 10, 32, 32),
        save_path="padarian_single_head",
        format="png",
        depth=3,
    )

    print("\n" + "=" * 60)
    print("Visualisiere: PadarianInspiredPatchCNN (multi head)")
    print("=" * 60)

    # Multi head model
    model_multi = PadarianInspiredPatchCNN(
        in_channels=10,
        n_traits=31,
        hidden_dim=64,
        dropout_p=0.3,
        head_type="multi",
    )

    visualize_and_save(
        model_multi,
        input_shape=(1, 10, 32, 32),
        save_path="padarian_multi_head",
        format="png",
        depth=3,
    )

    print("\n" + "=" * 60)
    print("Visualisiere: ResPatchCenterCNN")
    print("=" * 60)

    # ResPatchCenterCNN model
    model_res = ResPatchCenterCNN(
        in_channels=54,  # 50 predictors + 4 trig coord channels
        n_traits=31,
        base_channels=32,
        norm="gn",
        dropout_p=0.1,
        use_residual=True,
    )

    visualize_and_save(
        model_res,
        input_shape=(1, 54, 15, 15),
        save_path="res_patch_center_cnn",
        format="png",
        depth=3,
    )

    print("\n✅ Alle Visualisierungen erstellt!")
    print("   - padarian_single_head.png")
    print("   - padarian_multi_head.png")
    print("   - res_patch_center_cnn.png")
