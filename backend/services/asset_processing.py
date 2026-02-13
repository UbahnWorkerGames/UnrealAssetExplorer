from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

IMAGE_EXTS = {".webp"}
THUMB_SIZE = 256
DETAIL_SIZE = 512
FULL_SIZE = 1024
THUMB_QUALITY = 80
DETAIL_QUALITY = 88
FULL_QUALITY = 92


def _safe_extract(zip_path: Path, target_dir: Path) -> None:
    target_root = target_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute():
                raise ValueError("Zip contains invalid paths")
            resolved_target = (target_root / member_path).resolve()
            try:
                resolved_target.relative_to(target_root)
            except ValueError as exc:
                raise ValueError("Zip contains invalid paths") from exc
        zf.extractall(target_dir)


def _find_meta(meta_path: Path) -> Dict:
    if not meta_path.exists():
        raise FileNotFoundError("meta.json not found in zip")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _dedupe_frames(frames: List[Path]) -> List[Path]:
    seen = set()
    unique = []
    for frame in frames:
        data = frame.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(frame)
    return unique


def _frame_list(meta: Dict, root: Path) -> List[Path]:
    frames: List[str] = []
    if isinstance(meta.get("frames"), list) and meta["frames"]:
        frames = [item.get("file") for item in meta["frames"] if isinstance(item, dict)]
    if not frames:
        frames = meta.get("preview_files") or []
    return [root / str(frame) for frame in frames if frame]


def _resize_webp(src: Path, dest: Path, size: int, quality: int) -> None:
    with Image.open(src) as img:
        base = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        copy = img.convert("RGBA")
        copy.thumbnail((size, size), Image.LANCZOS)
        offset = ((size - copy.width) // 2, (size - copy.height) // 2)
        base.paste(copy, offset, copy)
        base.save(dest, "WEBP", quality=quality, method=6)


def _create_animation(frames: List[Path], dest: Path, size: int, quality: int) -> None:
    imgs = []
    for frame in frames:
        with Image.open(frame) as img:
            base = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            copy = img.convert("RGBA")
            copy.thumbnail((size, size), Image.LANCZOS)
            offset = ((size - copy.width) // 2, (size - copy.height) // 2)
            base.paste(copy, offset, copy)
            imgs.append(base)
    if not imgs:
        return
    imgs[0].save(
        dest,
        format="WEBP",
        save_all=True,
        append_images=imgs[1:],
        duration=200,
        loop=0,
        quality=quality,
        method=6,
    )


def process_asset_zip(
    zip_path: Path,
    asset_dir: Path,
    base_name: str,
    asset_class: Optional[str],
) -> Tuple[Dict, List[str], str, str, str, str, str]:
    temp_dir = asset_dir / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    _safe_extract(zip_path, temp_dir)
    meta = _find_meta(temp_dir / "meta.json")
    frames = _frame_list(meta, temp_dir)
    frames = [frame for frame in frames if frame.exists() and frame.suffix.lower() in IMAGE_EXTS]

    dedupe = asset_class and asset_class.lower().startswith("material")
    if dedupe:
        frames = _dedupe_frames(frames)

    thumb_image = ""
    detail_image = ""
    full_image = ""
    anim_thumb = ""
    anim_detail = ""

    if frames:
        thumb_path = asset_dir / f"{base_name}_t.webp"
        detail_path = asset_dir / f"{base_name}_d.webp"
        full_path = asset_dir / f"{base_name}_f.webp"
        _resize_webp(frames[0], thumb_path, THUMB_SIZE, THUMB_QUALITY)
        _resize_webp(frames[0], detail_path, DETAIL_SIZE, DETAIL_QUALITY)
        _resize_webp(frames[0], full_path, FULL_SIZE, FULL_QUALITY)
        thumb_image = str(thumb_path.relative_to(asset_dir))
        detail_image = str(detail_path.relative_to(asset_dir))
        full_image = str(full_path.relative_to(asset_dir))

    if len(frames) > 1:
        frames = frames[1:]
    if len(frames) > 1:
        anim_thumb_path = asset_dir / f"{base_name}_ta.webp"
        anim_detail_path = asset_dir / f"{base_name}_da.webp"
        _create_animation(frames, anim_thumb_path, THUMB_SIZE, THUMB_QUALITY)
        _create_animation(frames, anim_detail_path, DETAIL_SIZE, DETAIL_QUALITY)
        if anim_thumb_path.exists():
            anim_thumb = str(anim_thumb_path.relative_to(asset_dir))
        if anim_detail_path.exists():
            anim_detail = str(anim_detail_path.relative_to(asset_dir))

    shutil.rmtree(temp_dir, ignore_errors=True)

    preview_files = [frame.name for frame in frames]
    return meta, preview_files, thumb_image, detail_image, full_image, anim_thumb, anim_detail
