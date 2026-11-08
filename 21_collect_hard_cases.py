"""Collect small, openly licensed visual-review candidates for difficult cases.

The downloaded Commons images are unverified candidates, not labeled training data.
The yawn review CSV points to this project's existing external evaluation images.
"""

import argparse
import csv
import html
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "hard_cases"
API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "VisionProject-hard-case-review/1.0 (research dataset sampler)"
IMAGE_LIMIT_BYTES = 5 * 1024 * 1024
LICENSE_PREFIXES = ("CC BY ", "CC BY-SA ", "CC0", "Public domain")
EYE_MAKEUP_SEARCHES = ("eye makeup woman portrait", "eyeliner woman eye", "mascara eye close up")
COMMONS_COLUMNS = (
    "image",
    "source_title",
    "source_url",
    "license",
    "creator",
    "source_description",
    "condition_verified",
    "review_notes",
)
YAWN_COLUMNS = (
    "image",
    "actual",
    "predicted",
    "positive_probability",
    "status",
    "case_type",
    "neck_wrinkle_visible",
    "review_notes",
)


def clean_html(value):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", " ", value or ""))).strip()


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def search_commons(query, limit):
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": "6",
        "gsrlimit": str(min(limit * 4, 50)),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": "768",
        "format": "json",
        "formatversion": "2",
    }
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    payload = get_json(url)
    return payload.get("query", {}).get("pages", [])


def allowed_license(metadata):
    license_name = clean_html(metadata.get("LicenseShortName", {}).get("value", ""))
    return license_name.startswith(LICENSE_PREFIXES), license_name


def image_extension(url, title):
    suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".png", ".webp"):
        suffix = Path(title).suffix.lower()
    return suffix if suffix in (".jpg", ".jpeg", ".png", ".webp") else ".jpg"


def download_image(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        content_type = response.headers.get_content_type()
        if not content_type.startswith("image/"):
            raise ValueError(f"Expected an image from {url}, received {content_type}")
        data = response.read(IMAGE_LIMIT_BYTES + 1)
    if len(data) > IMAGE_LIMIT_BYTES:
        raise ValueError(f"Image exceeds {IMAGE_LIMIT_BYTES} bytes: {url}")
    destination.write_bytes(data)


def collect_commons(category, queries, limit):
    destination = OUTPUT / category / "images"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    seen = set()

    for query in queries:
        for page in search_commons(query, limit):
            image_info = (page.get("imageinfo") or [{}])[0]
            metadata = image_info.get("extmetadata", {})
            permitted, license_name = allowed_license(metadata)
            image_url = image_info.get("thumburl")
            source_url = image_info.get("descriptionurl", "")
            title = page.get("title", "image").removeprefix("File:")
            description = clean_html(metadata.get("ImageDescription", {}).get("value", ""))
            searchable = f"{title} {description}".lower()
            eye_terms = ("eye makeup", "eye shadow", "eyeshadow", "mascara", "eyeliner")
            excluded_terms = ("palette", "brush", "kit", "ancient", "figurine", "sculpture", "statue")
            if (
                not permitted
                or not image_url
                or source_url in seen
                or not any(term in searchable for term in eye_terms)
                or any(term in searchable for term in excluded_terms)
            ):
                continue
            seen.add(source_url)
            filename = f"commons_{page.get('pageid', len(records))}{image_extension(image_url, title)}"
            image_path = destination / filename
            if not image_path.is_file():
                download_image(image_url, image_path)
            records.append(
                {
                    "image": str(image_path.relative_to(ROOT)),
                    "source_title": title.removeprefix("File:"),
                    "source_url": source_url,
                    "license": license_name,
                    "creator": clean_html(metadata.get("Artist", {}).get("value", "")),
                    "source_description": description,
                    "condition_verified": "",
                    "review_notes": "",
                }
            )
            if len(records) >= limit:
                break
        if len(records) >= limit:
            break

    write_csv(OUTPUT / category / "candidates.csv", COMMONS_COLUMNS, records)
    return len(records)


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def make_yawn_review_manifest():
    source_csv = ROOT / "reports" / "yawn_roboflow" / "external_predictions.csv"
    records = []
    if source_csv.is_file():
        with source_csv.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                if (
                    row.get("task") != "yawn"
                    or row.get("model") != "team_resnet18"
                    or row.get("actual") != "no_yawn"
                ):
                    continue
                predicted = row.get("predicted", "")
                records.append(
                    {
                        "image": row.get("image", ""),
                        "actual": row.get("actual", ""),
                        "predicted": predicted,
                        "positive_probability": row.get("positive_probability", ""),
                        "status": row.get("status", ""),
                        "case_type": "false_alarm" if predicted == "yawn" else "correct_negative",
                        "neck_wrinkle_visible": "",
                        "review_notes": "",
                    }
                )
    else:
        source_dir = ROOT / "data" / "external_driver" / "yawn" / "no_yawn"
        for image in sorted(source_dir.rglob("*")):
            if image.is_file() and image.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                records.append(
                    {
                        "image": str(image),
                        "actual": "no_yawn",
                        "predicted": "",
                        "positive_probability": "",
                        "status": "",
                        "case_type": "unscored_negative",
                        "neck_wrinkle_visible": "",
                        "review_notes": "",
                    }
                )

    review_dir = OUTPUT / "neck_wrinkle"
    review_dir.mkdir(parents=True, exist_ok=True)
    write_csv(review_dir / "yawn_false_alarm_review.csv", YAWN_COLUMNS, records)
    return len(records)


def main():
    parser = argparse.ArgumentParser(description="Build small hard-case review sets.")
    parser.add_argument("--limit", type=int, default=8, help="Maximum Commons candidates per category.")
    args = parser.parse_args()
    if not 1 <= args.limit <= 20:
        parser.error("--limit must be between 1 and 20")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    eye_count = collect_commons("eyes_makeup", EYE_MAKEUP_SEARCHES, args.limit)
    print(f"eyes_makeup: downloaded {eye_count} openly licensed visual candidates")

    yawn_count = make_yawn_review_manifest()
    print(f"neck wrinkle review: indexed {yawn_count} existing negative samples")
    print(f"Output: {OUTPUT}")
    print("Commons candidates and neck-wrinkle visibility need manual verification.")
    print("Do not use these candidates for training or evaluation before labeling and split review.")


if __name__ == "__main__":
    main()
