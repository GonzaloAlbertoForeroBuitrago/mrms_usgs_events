"""
usgs-mrms-events

Unified USGS stage → event detection → MRMS RadarOnly pixel-only Zarr pipeline (resume-safe).
"""

from .config import PipelineConfig
from .pipeline import download_many_sites, download_single_site

__all__ = ["PipelineConfig", "download_single_site", "download_many_sites"]