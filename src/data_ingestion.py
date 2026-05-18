"""
src/data_ingestion.py
=====================
Zero-auth dataset downloads. No API key needed for anything.

Sources
-------
1. CMP Facade DB   -> exterior_facade  (Czech Univ, ~55 MB)
2. MIT Indoor      -> office_interior  (MIT CSAIL, ~2.4 GB)
3. MIT Indoor      -> warehouse        (MIT CSAIL, same archive)
4. Wikimedia Commons + Open Images -> hvac_pipeline (free, no key)

SSL bypass enabled for Windows certificate errors.
"""

import csv, json, os, random, shutil, ssl, sys
import tarfile, time, urllib.parse, urllib.request, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from logger import get_logger
from exception import AppException

log = get_logger(__name__)

# SSL fix for Windows
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode    = ssl.CERT_NONE
ssl._create_default_https_context = ssl._create_unverified_context

FACADE_URL    = "http://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.zip"
MIT_URL       = "http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar"
WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"

# Open Images V7 (Google Cloud Storage - free, no auth)
OI_CLASSES_URL = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv"
OI_LABELS_URL  = "https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels.csv"
OI_IMAGES_URL  = "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-boxable.csv"
OI_HVAC_CLASSES = ["Air conditioner", "Gas pipeline", "Radiator", "Boiler"]

HVAC_WIKI_CATEGORIES = [
    "Air_conditioning_equipment", "HVAC",
    "Natural_gas_pipelines", "Oil_pipelines",
    "Industrial_pipe_fittings", "Pipes_(plumbing)",
    "Pipeline_transport", "Heat_exchangers",
    "Boilers_(steam_generation)", "Refrigeration",
    "Duct_(HVAC)", "Compressors",
]

DATA_ROOT     = Path("data/raw")
CLASS_MAP     = Path("data/class_names.json")
SPLITS        = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED          = 42
MAX_PER_CLASS = 600

MIT_MAP = {
    "office":       "office_interior",
    "meeting_room": "office_interior",
    "warehouse":    "warehouse",
    "garage":       "warehouse",
}
EXTRACT_ONLY = set(MIT_MAP.keys())


class DataIngestion:
    def __init__(self, local_dir=None, skip_mit=False):
        self.local_dir = local_dir
        self.skip_mit  = skip_mit

    def run(self) -> dict:
        try:
            log.info("=" * 55)
            log.info("  PHASE 1  -  Data Ingestion  (Zero API keys)")
            log.info("=" * 55)
            if self.local_dir:
                return self._split_local(Path(self.local_dir))
            merged = self._download_all()
            splits = self._split_merged(merged)
            self._save_class_map(splits["train"])
            log.info("Data Ingestion complete.")
            return splits
        except Exception as e:
            raise AppException(e, sys) from e

    # ------------------------------------------------------------------ #
    def _download_all(self) -> dict:
        merged = {"exterior_facade": [], "office_interior": [],
                  "warehouse": [], "hvac_pipeline": []}

        # 1. CMP Facade (exterior_facade)
        facade_dir = DATA_ROOT / "cmp_facade"
        if facade_dir.exists() and any(facade_dir.rglob("*.jpg")):
            log.info("CMP Facade already on disk.")
        else:
            z = DATA_ROOT / "cmp_facade.zip"
            DATA_ROOT.mkdir(parents=True, exist_ok=True)
            if not z.exists():
                self._http_download(FACADE_URL, z, "CMP Facade DB (~55 MB)")
            log.info("Extracting CMP Facade ...")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(facade_dir)
            z.unlink(missing_ok=True)

        imgs = (list(facade_dir.rglob("*.jpg"))
                + list(facade_dir.rglob("*.JPG"))
                + list(facade_dir.rglob("*.png")))
        merged["exterior_facade"].extend(imgs)
        log.info(f"exterior_facade: {len(imgs)} images (CMP Facade DB)")

        # 2. MIT Indoor (office + warehouse)
        if not self.skip_mit:
            mit_dir = DATA_ROOT / "mit_indoor"
            if mit_dir.exists() and any(mit_dir.rglob("*.jpg")):
                log.info("MIT Indoor already on disk.")
            else:
                tar = DATA_ROOT / "indoorCVPR_09.tar"
                if not tar.exists():
                    self._http_download(MIT_URL, tar,
                                        "MIT Indoor Scenes (~2.4 GB)")
                log.info("Extracting MIT Indoor (selected categories) ...")
                mit_dir.mkdir(parents=True, exist_ok=True)
                self._selective_tar(tar, mit_dir)
                tar.unlink(missing_ok=True)

            for root, _, files in os.walk(str(mit_dir)):
                folder = Path(root).name.lower()
                if folder in MIT_MAP:
                    label = MIT_MAP[folder]
                    imgs  = [Path(root)/f for f in files
                             if f.lower().endswith((".jpg",".jpeg",".png"))]
                    if imgs:
                        merged[label].extend(imgs)
                        log.info(f"  MIT '{folder}' -> '{label}': {len(imgs)}")
        else:
            log.warning("--skip-mit: office and warehouse skipped.")

        # 3. HVAC - Wikimedia + Open Images (zero auth)
        hvac_imgs = self._download_hvac()
        merged["hvac_pipeline"].extend(hvac_imgs)

        for cls, imgs in merged.items():
            log.info(f"  Total {cls}: {len(imgs)}")
        return merged

    # ------------------------------------------------------------------ #
    def _download_hvac(self) -> list:
        hvac_dir = DATA_ROOT / "hvac_images"
        existing = (list(hvac_dir.glob("*.jpg"))
                    + list(hvac_dir.glob("*.png"))
                    + list(hvac_dir.glob("*.jpeg"))) if hvac_dir.exists() else []

        if len(existing) >= 80:
            log.info(f"HVAC images already on disk: {len(existing)}")
            return existing

        hvac_dir.mkdir(parents=True, exist_ok=True)
        all_imgs = []

        # Source A: Wikimedia Commons (12 HVAC categories, no auth)
        log.info("Downloading HVAC images from Wikimedia Commons ...")
        wiki_imgs = self._wikimedia_hvac(hvac_dir)
        all_imgs.extend(wiki_imgs)
        log.info(f"  Wikimedia: {len(wiki_imgs)} HVAC images")

        # Source B: Open Images V7 (Google Cloud, no auth)
        if len(all_imgs) < MAX_PER_CLASS:
            log.info("Downloading HVAC images from Open Images V7 ...")
            oi_imgs = self._open_images_hvac(hvac_dir)
            all_imgs.extend(oi_imgs)
            log.info(f"  Open Images: {len(oi_imgs)} HVAC images")

        log.info(f"hvac_pipeline total: {len(all_imgs)} images (no API key)")
        return all_imgs

    # ------------------------------------------------------------------ #
    def _wikimedia_hvac(self, dest_dir: Path) -> list:
        all_titles = []
        for cat in HVAC_WIKI_CATEGORIES:
            titles = self._wiki_category_files(cat, limit=60)
            all_titles.extend(titles)
            time.sleep(0.2)

        all_titles = list(dict.fromkeys(all_titles))
        log.info(f"  Wikimedia: {len(all_titles)} unique files found")

        urls = []
        for i in range(0, len(all_titles), 50):
            urls.extend(self._wiki_resolve_urls(all_titles[i:i+50]))
            time.sleep(0.15)

        downloaded = []
        for idx, (fname, url) in enumerate(urls):
            if len(downloaded) >= MAX_PER_CLASS // 2:
                break
            ext = Path(fname).suffix.lower()
            if ext not in (".jpg", ".jpeg", ".png"):
                continue
            dest = dest_dir / f"wiki_{idx:05d}{ext}"
            if dest.exists():
                downloaded.append(dest)
                continue
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "PropertyClassifier/1.0 (academic)"})
                with urllib.request.urlopen(
                        req, timeout=12, context=SSL_CTX) as r:
                    data = r.read()
                if len(data) < 5000:
                    continue
                dest.write_bytes(data)
                downloaded.append(dest)
                if len(downloaded) % 30 == 0:
                    log.info(f"  Wikimedia: {len(downloaded)} downloaded ...")
                time.sleep(0.08)
            except Exception:
                continue
        return downloaded

    def _wiki_category_files(self, category: str, limit: int = 60) -> list:
        params = urllib.parse.urlencode({
            "action": "query", "list": "categorymembers",
            "cmtitle": f"Category:{category}", "cmlimit": str(limit),
            "cmtype": "file", "format": "json",
        })
        try:
            req = urllib.request.Request(
                f"{WIKIMEDIA_API}?{params}",
                headers={"User-Agent": "PropertyClassifier/1.0"})
            with urllib.request.urlopen(
                    req, timeout=12, context=SSL_CTX) as r:
                data = json.loads(r.read())
            return [m["title"] for m in
                    data.get("query", {}).get("categorymembers", [])]
        except Exception as e:
            log.warning(f"  Wiki '{category}': {e}")
            return []

    def _wiki_resolve_urls(self, titles: list) -> list:
        params = urllib.parse.urlencode({
            "action": "query", "titles": "|".join(titles),
            "prop": "imageinfo", "iiprop": "url", "format": "json",
        })
        try:
            req = urllib.request.Request(
                f"{WIKIMEDIA_API}?{params}",
                headers={"User-Agent": "PropertyClassifier/1.0"})
            with urllib.request.urlopen(
                    req, timeout=12, context=SSL_CTX) as r:
                data = json.loads(r.read())
            result = []
            for page in data.get("query", {}).get("pages", {}).values():
                infos = page.get("imageinfo", [])
                if infos:
                    result.append((page.get("title", ""), infos[0]["url"]))
            return result
        except Exception as e:
            log.warning(f"  Wiki resolve: {e}")
            return []

    # ------------------------------------------------------------------ #
    def _open_images_hvac(self, dest_dir: Path) -> list:
        """Download HVAC images from Open Images V7 (Google Cloud, no auth)."""
        try:
            log.info("  Fetching Open Images class list ...")
            req = urllib.request.Request(OI_CLASSES_URL, headers={
                "User-Agent": "PropertyClassifier/1.0"})
            with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as r:
                text = r.read().decode("utf-8")

            # Find label IDs for HVAC classes
            hvac_ids = set()
            for row in csv.reader(text.splitlines()):
                if len(row) >= 2 and any(
                        c.lower() in row[1].lower() for c in OI_HVAC_CLASSES):
                    hvac_ids.add(row[0])
                    log.info(f"  Open Images class found: {row[1]} ({row[0]})")

            if not hvac_ids:
                log.warning("  No HVAC classes found in Open Images.")
                return []

            log.info("  Fetching validation labels ...")
            req = urllib.request.Request(OI_LABELS_URL, headers={
                "User-Agent": "PropertyClassifier/1.0"})
            with urllib.request.urlopen(req, timeout=60, context=SSL_CTX) as r:
                labels_text = r.read().decode("utf-8")

            # Get image IDs with HVAC labels and confidence=1
            hvac_image_ids = set()
            for row in csv.reader(labels_text.splitlines()):
                if len(row) >= 4 and row[2] in hvac_ids and row[3] == "1":
                    hvac_image_ids.add(row[0])

            log.info(f"  Found {len(hvac_image_ids)} HVAC images in Open Images")

            if not hvac_image_ids:
                return []

            log.info("  Fetching image URLs ...")
            req = urllib.request.Request(OI_IMAGES_URL, headers={
                "User-Agent": "PropertyClassifier/1.0"})
            with urllib.request.urlopen(req, timeout=60, context=SSL_CTX) as r:
                images_text = r.read().decode("utf-8")

            # Map image IDs to download URLs
            url_map = {}
            for row in csv.reader(images_text.splitlines()):
                if len(row) >= 3 and row[0] in hvac_image_ids:
                    url_map[row[0]] = row[2]  # OriginalURL column

            log.info(f"  Resolved {len(url_map)} image URLs")

            # Download images
            downloaded = []
            for idx, (img_id, url) in enumerate(list(url_map.items())[:200]):
                if len(downloaded) >= MAX_PER_CLASS // 2:
                    break
                dest = dest_dir / f"oi_{idx:05d}.jpg"
                if dest.exists():
                    downloaded.append(dest)
                    continue
                try:
                    req = urllib.request.Request(url, headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
                    with urllib.request.urlopen(
                            req, timeout=10, context=SSL_CTX) as r:
                        data = r.read()
                    if len(data) < 5000:
                        continue
                    dest.write_bytes(data)
                    downloaded.append(dest)
                    if len(downloaded) % 20 == 0:
                        log.info(f"  Open Images: {len(downloaded)} downloaded ...")
                    time.sleep(0.05)
                except Exception:
                    continue

            return downloaded

        except Exception as e:
            log.warning(f"Open Images download failed: {e}")
            return []

    # ------------------------------------------------------------------ #
    def _http_download(self, url: str, dest: Path, label: str):
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"Downloading: {label}")
        log.info(f"  URL: {url}")
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, context=SSL_CTX) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                done  = 0
                with open(dest, "wb") as f:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if total > 0 and done % (1024*1024*50) < 65536:
                            log.info(f"  {done/total*100:.1f}%  "
                                     f"({done/1048576:.0f}/{total/1048576:.0f} MB)")
            log.info(f"  Complete -> {dest}")
        except Exception as e:
            if dest.exists():
                dest.unlink()
            raise AppException(f"Download failed: {url}\n{e}", sys) from e

    # ------------------------------------------------------------------ #
    def _selective_tar(self, tar_path: Path, dest: Path):
        with tarfile.open(tar_path, "r:*") as tar:
            members = tar.getmembers()
            keep = [m for m in members
                    if len(Path(m.name).parts) >= 3
                    and Path(m.name).parts[-2].lower() in EXTRACT_ONLY]
            log.info(f"  Extracting {len(keep)}/{len(members)} files ...")
            for i, m in enumerate(keep):
                tar.extract(m, dest)
                if i % 300 == 0 and i > 0:
                    log.info(f"  Extracted {i}/{len(keep)} ...")

    # ------------------------------------------------------------------ #
    def _split_merged(self, merged: dict) -> dict:
        random.seed(SEED)
        split_dirs = {s: DATA_ROOT / s for s in SPLITS}
        for d in split_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        for cls_name, images in merged.items():
            if not images:
                log.warning(f"No images for '{cls_name}' - skipping.")
                continue
            random.shuffle(images)
            images = images[:MAX_PER_CLASS]
            n  = len(images)
            n1 = int(n * SPLITS["train"])
            n2 = int(n * SPLITS["val"])
            for split, files in {
                "train": images[:n1],
                "val":   images[n1:n1+n2],
                "test":  images[n1+n2:],
            }.items():
                dest = split_dirs[split] / cls_name
                dest.mkdir(exist_ok=True)
                for i, src in enumerate(files):
                    ext = src.suffix.lower() or ".jpg"
                    dst = dest / f"{cls_name}_{i:05d}{ext}"
                    if not dst.exists():
                        shutil.copy2(src, dst)
            log.info(f"  {cls_name}: {n} -> "
                     f"train={n1} val={n2} test={n-n1-n2}")

        counts = {s: sum(1 for _ in (DATA_ROOT/s).rglob("*.jpg"))
                  for s in SPLITS}
        log.info(f"Split sizes: {counts}")
        return split_dirs

    def _split_local(self, source: Path) -> dict:
        random.seed(SEED)
        split_dirs = {s: DATA_ROOT / s for s in SPLITS}
        for d in split_dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        for cls_dir in (p for p in source.iterdir() if p.is_dir()):
            imgs = (list(cls_dir.glob("*.jpg"))
                    + list(cls_dir.glob("*.png"))
                    + list(cls_dir.glob("*.jpeg")))
            if not imgs:
                continue
            random.shuffle(imgs)
            n1, n2 = int(len(imgs)*0.7), int(len(imgs)*0.15)
            for split, files in {
                "train": imgs[:n1],
                "val":   imgs[n1:n1+n2],
                "test":  imgs[n1+n2:],
            }.items():
                dest = split_dirs[split] / cls_dir.name
                dest.mkdir(exist_ok=True)
                for f in files:
                    t = dest / f.name
                    if not t.exists():
                        shutil.copy2(f, t)
        self._save_class_map(split_dirs["train"])
        return split_dirs

    def _save_class_map(self, train_dir: Path):
        classes = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
        mapping = {i: c for i, c in enumerate(classes)}
        CLASS_MAP.parent.mkdir(exist_ok=True)
        CLASS_MAP.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        log.info(f"Class map -> {CLASS_MAP}")
        log.info(f"Classes   : {list(mapping.values())}")
