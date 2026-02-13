from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ASSET_",
        extra="ignore",
    )

    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Optional[Path] = None
    assets_dir: Optional[Path] = None
    projects_dir: Optional[Path] = None
    uploads_dir: Optional[Path] = None
    batch_output_dir: Optional[Path] = None
    startup_jobs_dir: Optional[Path] = None
    db_path: Optional[Path] = None

    @model_validator(mode="after")
    def derive_paths(self) -> "AppSettings":
        self.base_dir = self.base_dir.expanduser().resolve()
        self.data_dir = (self.data_dir or (self.base_dir / "data")).expanduser().resolve()
        self.assets_dir = (self.assets_dir or (self.data_dir / "assets")).expanduser().resolve()
        self.projects_dir = (self.projects_dir or (self.data_dir / "projects")).expanduser().resolve()
        self.uploads_dir = (self.uploads_dir or (self.data_dir / "uploads")).expanduser().resolve()
        self.batch_output_dir = (self.batch_output_dir or (self.data_dir / "batch_outputs")).expanduser().resolve()
        self.startup_jobs_dir = (self.startup_jobs_dir or (self.data_dir / "startup_jobs")).expanduser().resolve()
        self.db_path = (self.db_path or (self.data_dir / "app.db")).expanduser().resolve()
        return self


@lru_cache(maxsize=1)
def get_app_settings() -> AppSettings:
    return AppSettings()
