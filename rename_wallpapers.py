#!/usr/bin/env python3
"""Suggest descriptive wallpaper filenames using a local Ollama vision model."""

import argparse
import base64
import json
import os
import re
import shutil
import sys
import subprocess
import unicodedata
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def describe_image(path: Path, endpoint: str, model: str) -> str:
    payload = {
        "model": model,
        "prompt": (
            "Describe this wallpaper in a concise filename of 2 to 6 words. "
            "Return only the words, with no quotes or explanation."
        ),
        "images": [base64.b64encode(path.read_bytes()).decode("ascii")],
        "stream": False,
    }
    request = Request(
        f"{endpoint.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=300) as response:
        result = json.load(response)
    description = result.get("response", "").strip()
    if not description:
        raise ValueError(f"Ollama returned an empty description for {path.name}")
    return description


def suggest_names(path: Path, endpoint: str, model: str) -> list[str]:
    prompt = (
        "Suggest exactly 3 distinct concise filenames for this wallpaper, "
        "2 to 6 words each. Return only the names, one per line, with no "
        "numbering, quotes, or explanation."
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [base64.b64encode(path.read_bytes()).decode("ascii")],
        "stream": False,
    }
    request = Request(
        f"{endpoint.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=300) as response:
        result = json.load(response)
    text = result.get("response", "").strip()
    if not text:
        raise ValueError(f"Ollama returned no suggestions for {path.name}")

    suggestions = []
    for line in text.splitlines():
        name = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line).strip().strip('"\'`')
        slug = make_slug(name)
        if slug != "wallpaper" and slug not in suggestions:
            suggestions.append(slug)
    if not suggestions:
        raise ValueError(f"Ollama returned no usable suggestions for {path.name}")
    return suggestions[:3]


def make_slug(description: str) -> str:
    text = unicodedata.normalize("NFKD", description)
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return slug[:70].rstrip("-") or "wallpaper"


def needs_name(path: Path) -> bool:
    return path.stem.isdigit()


def show_image(path: Path) -> None:
    if os.environ.get("KITTY_WINDOW_ID") and shutil.which("kitty"):
        subprocess.run(
            ["kitty", "+kitten", "icat", "--align", "left", str(path)],
            check=False,
        )
    elif shutil.which("chafa"):
        subprocess.run(["chafa", str(path)], check=False)
    elif shutil.which("xdg-open"):
        subprocess.Popen(
            ["xdg-open", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        print("(Opened in your image viewer; close it when you're ready.)")
    else:
        print("(No Kitty icat, chafa, or xdg-open found; skipping image preview.)")


def choose_destination(source: Path, slug: str, used_names: set[str]) -> Path:
    destination = source.with_name(f"{slug}{source.suffix}")
    if destination == source:
        return source
    number = 2
    while destination.name in used_names:
        destination = source.with_name(f"{slug}-{number}{source.suffix}")
        number += 1
    return destination


def interactive_rename(files: list[Path], endpoint: str, model: str) -> int:
    used_names = {path.name for path in files[0].parent.iterdir()}
    for index, path in enumerate(files, start=1):
        print(f"\n[{index}/{len(files)}] Current name: {path.name}")
        show_image(path)
        while True:
            answer = input("Rename this image? [y/N/q] ").strip().lower()
            if answer in {"", "n"}:
                break
            if answer == "q":
                print("Stopped.")
                return 0
            if answer == "y":
                break
            print("Enter y to see name suggestions, n to keep this name, or q to quit.")
        if answer != "y":
            continue

        try:
            suggestions = suggest_names(path, endpoint, model)
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
            print(f"Could not generate suggestions: {error}", file=sys.stderr)
            return 1

        print("Suggested names:")
        for number, slug in enumerate(suggestions, start=1):
            destination = choose_destination(path, slug, used_names)
            print(f"  {number}. {destination.name}")
        print("  c. Enter a custom name")
        while True:
            choice = input("Choose a number, c, or press Enter to keep the current name: ").strip().lower()
            if not choice:
                break
            if choice == "c":
                custom = input("New name (without extension): ").strip()
                slug = make_slug(custom)
                if slug == "wallpaper":
                    print("Please enter a name containing letters or numbers.")
                    continue
                break
            if choice.isdigit() and 1 <= int(choice) <= len(suggestions):
                slug = suggestions[int(choice) - 1]
                break
            print("Choose one of the listed numbers, c, or press Enter to keep the name.")
        if not choice:
            continue

        destination = choose_destination(path, slug, used_names)
        if destination == path:
            print("The selected name is already in use for this image.")
            continue
        path.rename(destination)
        used_names.discard(path.name)
        used_names.add(destination.name)
        print(f"Renamed to {destination.name}")

    print("Finished.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("."))
    parser.add_argument("--model", default="llava", help="Ollama vision model (default: llava)")
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--apply",
        action="store_true",
        help="perform the renames; without this flag, only show a preview",
    )
    action.add_argument(
        "--interactive",
        action="store_true",
        help="preview each image and choose whether and how to rename it",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="only process images whose filenames contain no name (for example, 0111.jpg)",
    )
    args = parser.parse_args()

    directory = args.directory.resolve()
    if not directory.is_dir():
        parser.error(f"not a directory: {directory}")

    endpoint = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    files = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if args.auto:
        files = [path for path in files if needs_name(path)]
    if not files:
        if args.auto:
            print(f"No supported images with numeric-only filenames found in {directory}")
        else:
            print(f"No supported images found in {directory}")
        return 0

    if args.interactive:
        return interactive_rename(files, endpoint, args.model)

    # Reserve existing names so generated filenames never overwrite another image.
    used_names = {path.name for path in directory.iterdir()}
    plan = []
    try:
        for path in files:
            slug = make_slug(describe_image(path, endpoint, args.model))
            destination = path.with_name(f"{slug}{path.suffix.lower()}")
            if destination != path:
                stem = slug
                number = 2
                while destination.name in used_names:
                    destination = path.with_name(f"{stem}-{number}{path.suffix.lower()}")
                    number += 1
                used_names.add(destination.name)
            plan.append((path, destination))
    except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
        print(f"Could not generate filenames: {error}", file=sys.stderr)
        return 1

    for source, destination in plan:
        if source == destination:
            print(f"unchanged  {source.name}")
        else:
            print(f"{source.name} -> {destination.name}")

    if not args.apply:
        print("\nPreview only; rerun with --apply to rename.")
        return 0

    for source, destination in plan:
        if source != destination:
            source.rename(destination)
    print("\nRenamed files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
