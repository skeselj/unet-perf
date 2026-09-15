"""
Module to support running inference on data with models.

TODO: rewrite this.
"""

# import itertools

# import torch

# from data import CarvanaData
# from model import UNet


# def pipeline(
#     data_dir: str = "data/carvana",
#     batch_size: int = 8,
#     batch_count: int = 4,
# ):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"device: {device}")

#     # Create a randomly initialized model.
#     torch.manual_seed(0)
#     model = UNet(base_height=128, base_width=128, output_channel_count=2, level_count=3)
#     model = model.to(device).eval()

#     # Take the first batch_count batches of the Carvana dataset.
#     data = CarvanaData(data_dir=data_dir, batch_size=batch_size)
#     batches = itertools.islice(data.get_dataset(), batch_count)
#     for batch_idx, (frames, masks) in enumerate(batches):
#         # uint8 B x H x W x 3 -> float B x 3 x H x W in [0, 1].
#         inputs = torch.from_numpy(frames).to(device)
#         inputs = inputs.permute(0, 3, 1, 2).float() / 255.0

#         # For each batch, run inference with the model.
#         outputs = model.predict(inputs)

#         # Print a sanity check, like the sum of the values of the inputs and
#         # outputs.
#         print(
#             f"batch {batch_idx:4d}  {tuple(inputs.shape)} -> "
#             f"{tuple(outputs.shape)}  "
#             f"mask {masks.shape}  "
#             f"input sum {inputs.sum().item():.2f}  "
#             f"output sum {outputs.sum().item():.4f}"
#         )


# if __name__ == "__main__":
#     # Run pipeline with default params.
#     pipeline()
