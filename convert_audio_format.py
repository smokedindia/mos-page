#!/usr/bin/env python3
import argparse
from pathlib import Path

import soundfile as sf


SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".dbl", ".dlb"}


def convert_one_file(src_path: Path, src_root: Path, dst_root: Path, subtype: str) -> bool:
    rel_path = src_path.relative_to(src_root)
    dst_path = (dst_root / rel_path).with_suffix(".wav")
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        audio, sample_rate = sf.read(str(src_path), always_2d=False)
        sf.write(str(dst_path), audio, sample_rate, format="WAV", subtype=subtype)
        print(f"OK   {src_path} -> {dst_path}")
        return True
    except Exception as exc:
        print(f"FAIL {src_path} ({exc})")
        return False


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Load audio with soundfile and resave as properly formatted WAV files "
            "under a different root directory."
        )
    )
    parser.add_argument("--src-root", required=True, help="Source root directory")
    parser.add_argument("--dst-root", required=True, help="Destination root directory")
    parser.add_argument(
        "--subtype",
        default="PCM_16",
        help="Output WAV subtype (default: PCM_16)",
    )
    parser.add_argument(
        "--ext",
        nargs="*",
        default=None,
        help="Optional extension allowlist (e.g. --ext .wav .dbl .dlb)",
    )
    args = parser.parse_args()

    src_root = Path(args.src_root).expanduser().resolve()
    dst_root = Path(args.dst_root).expanduser().resolve()

    if not src_root.is_dir():
        raise SystemExit(f"Source root does not exist: {src_root}")

    extensions = (
        {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.ext}
        if args.ext
        else SUPPORTED_EXTENSIONS
    )

    src_files = [
        path
        for path in src_root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]

    if not src_files:
        print("No matching audio files found.")
        return

    success_count = 0
    fail_count = 0
    for src_path in src_files:
        if convert_one_file(src_path, src_root, dst_root, args.subtype):
            success_count += 1
        else:
            fail_count += 1

    print(
        f"Done. total={len(src_files)} success={success_count} fail={fail_count} "
        f"output_root={dst_root}"
    )


if __name__ == "__main__":
    main()
