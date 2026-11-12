"""Fetch the official annotation archives and only the selected COCO images."""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile


VQA = "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/"
COCO = "https://s3.amazonaws.com/images.cocodataset.org"
ARCHIVES = ["v2_Annotations_Train_mscoco.zip", "v2_Annotations_Val_mscoco.zip",
            "v2_Questions_Train_mscoco.zip", "v2_Questions_Val_mscoco.zip",
            "v2_Complementary_Pairs_Train_mscoco.zip", "v2_Complementary_Pairs_Val_mscoco.zip"]
NEEDED = {"captions_train2014.json", "captions_val2014.json", "instances_train2014.json", "instances_val2014.json",
          "v2_OpenEnded_mscoco_train2014_questions.json", "v2_OpenEnded_mscoco_val2014_questions.json",
          "v2_mscoco_train2014_annotations.json", "v2_mscoco_val2014_annotations.json",
          "v2_mscoco_train2014_complementary_pairs.json", "v2_mscoco_val2014_complementary_pairs.json"}
ARCHIVE_CONTENTS = {
    "v2_Annotations_Train_mscoco.zip": {"v2_mscoco_train2014_annotations.json"},
    "v2_Annotations_Val_mscoco.zip": {"v2_mscoco_val2014_annotations.json"},
    "v2_Questions_Train_mscoco.zip": {"v2_OpenEnded_mscoco_train2014_questions.json"},
    "v2_Questions_Val_mscoco.zip": {"v2_OpenEnded_mscoco_val2014_questions.json"},
    "v2_Complementary_Pairs_Train_mscoco.zip": {"v2_mscoco_train2014_complementary_pairs.json"},
    "v2_Complementary_Pairs_Val_mscoco.zip": {"v2_mscoco_val2014_complementary_pairs.json"},
    "annotations_trainval2014.zip": {"captions_train2014.json", "captions_val2014.json", "instances_train2014.json", "instances_val2014.json"},
}


def fetch(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={"User-Agent": "PA3-Task2/1.0"}), timeout=60) as source, temporary.open("wb") as out:
                while chunk := source.read(1024 * 1024):
                    out.write(chunk)
            os.replace(temporary, destination)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def annotations(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    sources = [(VQA + name, name) for name in ARCHIVES]
    sources.append((f"{COCO}/annotations/annotations_trainval2014.zip", "annotations_trainval2014.zip"))
    for url, name in sources:
        if all((destination / item).is_file() for item in ARCHIVE_CONTENTS[name]):
            print(f"Already extracted {name}", flush=True)
            continue
        archive = destination / name
        print(f"Downloading {name}", flush=True)
        fetch(url, archive)
        with ZipFile(archive) as handle:
            for member in handle.infolist():
                basename = Path(member.filename).name
                if basename in NEEDED and not (destination / basename).exists():
                    with handle.open(member) as source, (destination / basename).open("wb") as out:
                        while chunk := source.read(1024 * 1024):
                            out.write(chunk)
        archive.unlink()
    missing = sorted(name for name in NEEDED if not (destination / name).is_file())
    if missing:
        raise RuntimeError(f"Missing expected annotation files: {missing}")


def images(manifest: Path, workers: int) -> None:
    rows = json.loads(manifest.read_text(encoding="utf-8"))["images"]
    def one(row):
        split, filename = row["split"], row["filename"]
        if split not in ("train2014", "val2014") or Path(filename).name != filename:
            raise ValueError("Invalid COCO image source")
        fetch(f"{COCO}/{split}/{filename}", Path(row["path"]))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, _ in enumerate(pool.map(one, rows), 1):
            if count % 100 == 0 or count == len(rows):
                print(f"Downloaded {count}/{len(rows)} images", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("annotations", "images"))
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "annotations" and args.directory:
        annotations(args.directory)
    elif args.stage == "images" and args.manifest:
        images(args.manifest, args.workers)
    else:
        parser.error("annotations needs --directory; images needs --manifest")


if __name__ == "__main__":
    main()
