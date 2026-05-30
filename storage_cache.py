import os
from pathlib import Path


def storage_configured():
    return bool(os.environ.get("OBJECT_STORAGE_BUCKET"))


def ensure_era5_files(folder, year_start, year_end):
    """Download missing ERA5 files from S3-compatible object storage when configured."""
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)

    missing = []
    for year in range(int(year_start), int(year_end) + 1):
        missing.extend([
            folder_path / f"Hs_{year}.nc",
            folder_path / f"Tp_{year}.nc",
        ])
    missing = [p for p in missing if not p.exists()]

    if not missing or not storage_configured():
        return missing

    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required for object storage downloads. Run: pip install boto3") from exc

    endpoint_url = os.environ.get("OBJECT_STORAGE_ENDPOINT") or None
    bucket = os.environ["OBJECT_STORAGE_BUCKET"]
    prefix = os.environ.get("OBJECT_STORAGE_PREFIX", "era5").strip("/")

    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.environ.get("OBJECT_STORAGE_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("OBJECT_STORAGE_SECRET_KEY"),
        region_name=os.environ.get("OBJECT_STORAGE_REGION", "auto"),
    )

    for local_path in missing:
        key = f"{prefix}/{local_path.name}" if prefix else local_path.name
        client.download_file(bucket, key, str(local_path))

    return []
