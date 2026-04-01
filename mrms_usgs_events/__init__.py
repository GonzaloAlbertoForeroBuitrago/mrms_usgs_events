"""
usgs-mrms-events

Unified USGS stage → event detection → MRMS RadarOnly pixel-only Zarr pipeline (resume-safe).
"""

from .config import PipelineConfig
from .pipeline import download_many_sites, download_single_site
from .paths import normalize_site_id

__all__ = ["PipelineConfig", "download_single_site", "download_many_sites", "normalize_site_id"]