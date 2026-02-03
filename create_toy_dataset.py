"""
Create a toy NetCDF dataset for testing STConvS2S with configurable channels.

This script generates a small dataset with known dimensions to test the
--input-channels and --output-channels CLI arguments.

Dataset structure:
- 100 samples
- 10 timesteps per sample
- 10x10 spatial grid (lat x lon)
- 4 channels (e.g., temperature, precipitation, humidity, pressure)
"""

import numpy as np
import xarray as xr
from pathlib import Path

def create_toy_dataset(
    output_file="data/toy-dataset-seq5-ystep5.nc",
    n_samples=100,
    n_timesteps=5,
    n_lat=10,
    n_lon=10,
    # n_channels=4,
    n_channels=5,
    seed=42
):
    """
    Create a toy NetCDF dataset compatible with STConvS2S.
    
    Args:
        output_file: Path to save the NetCDF file
        n_samples: Number of samples
        n_timesteps: Number of timesteps per sample
        n_lat: Number of latitude points
        n_lon: Number of longitude points
        n_channels: Number of channels (variables)
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    # Create coordinates
    lat = np.linspace(-30, -20, n_lat)  # Example: Brazilian region
    lon = np.linspace(-50, -40, n_lon)
    sample = np.arange(n_samples)
    timestep = np.arange(n_timesteps)
    channel = np.arange(n_channels)
    
    # Generate random data with some structure
    # X data: input features
    x_data = np.random.randn(n_samples, n_timesteps, n_lat, n_lon, n_channels)
    
    # Add some spatial structure (smoother patterns)
    for s in range(n_samples):
        for t in range(n_timesteps):
            for c in range(n_channels):
                # Add a simple gradient pattern
                x_data[s, t, :, :, c] += np.linspace(0, 5, n_lat)[:, None]
                x_data[s, t, :, :, c] += np.linspace(0, 5, n_lon)[None, :]
    
    # Y data: target (similar structure but slightly different)
    y_data = np.random.randn(n_samples, n_timesteps, n_lat, n_lon, n_channels)
    
    # Make y somewhat correlated with x (simple relationship)
    y_data = 0.7 * x_data + 0.3 * y_data
    
    # Create xarray Dataset
    ds = xr.Dataset(
        {
            "x": (["sample", "timestep", "lat", "lon", "channel"], x_data),
            "y": (["sample", "timestep", "lat", "lon", "channel"], y_data),
        },
        coords={
            "sample": sample,
            "timestep": timestep,
            "lat": lat,
            "lon": lon,
            "channel": channel,
        },
        attrs={
            "description": "Toy dataset for testing STConvS2S channel configuration",
            "n_samples": n_samples,
            "n_timesteps": n_timesteps,
            "n_lat": n_lat,
            "n_lon": n_lon,
            "n_channels": n_channels,
        }
    )
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to NetCDF
    print(f"Creating toy dataset...")
    print(f"  Samples: {n_samples}")
    print(f"  Timesteps: {n_timesteps}")
    print(f"  Spatial grid: {n_lat}x{n_lon}")
    print(f"  Channels: {n_channels}")
    print(f"  X shape: {x_data.shape}")
    print(f"  Y shape: {y_data.shape}")
    
    ds.to_netcdf(output_file)
    print(f"\n✓ Dataset saved to: {output_file}")
    
    # Verify by loading
    print("\nVerifying dataset...")
    loaded_ds = xr.load_dataset(output_file)
    print(f"  X dimensions: {dict(loaded_ds.x.sizes)}")
    print(f"  Y dimensions: {dict(loaded_ds.y.sizes)}")
    print(f"\n✓ Dataset verified successfully!")
    
    return ds


if __name__ == "__main__":
    # Create the toy dataset
    ds = create_toy_dataset()
    
    print("\n" + "="*60)
    print("Dataset created successfully!")
    print("="*60)
    print("\nTest commands:")
    print("\n1. Test with default channels (5 input timesteps, all output channels):")
    print("   python main.py -m stconvs2s-r -e 5 -b 10 --small-dataset --verbose")
    
    print("\n2. Test with custom input channels (3 timesteps only):")
    print("   python main.py -m stconvs2s-r -e 5 -b 10 --small-dataset --input-channels 3 --verbose")
    
    print("\n3. Test with custom output channels (only first channel):")
    print("   python main.py -m stconvs2s-r -e 5 -b 10 --small-dataset --output-channels 1 --verbose")
    
    print("\n4. Test with both custom input and output channels:")
    print("   python main.py -m stconvs2s-r -e 5 -b 10 --small-dataset --input-channels 3 --output-channels 2 --verbose")
