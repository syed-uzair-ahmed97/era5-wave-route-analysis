# Deployment Notes

## Local Login

The app now uses SQLite sessions and a local user database.

Default development login:

- Email: `admin@example.com`
- Password: `admin123`

Before hosting online, set:

```bash
SECRET_KEY=...
ADMIN_EMAIL=...
ADMIN_PASSWORD=...
```

The first startup creates `app.db`, seeds the admin user, and loads the initial port database.

## Pages

Protected app routes:

- `/dashboard`
- `/route-builder`
- `/analysis`
- `/results-page`

They currently serve the same dashboard shell while the sidebar contains the route builder, analysis controls, and results view.

## Port Database

Ports are stored in SQLite in the `ports` table. The frontend queries:

```text
GET /api/ports/search?q=cape
```

For production, import UN/LOCODE or World Port Index records into this table.

## ERA5 Storage

Do not expose your ERA5/CDS API key to users. Store prepared NetCDF files in object storage.

Expected object layout:

```text
era5/Hs_2016.nc
era5/Tp_2016.nc
era5/Hs_2017.nc
era5/Tp_2017.nc
```

The backend checks `data/` first. If a file is missing and object storage is configured, it downloads the missing file before analysis.

## Runtime Output Strategy

Normal analysis runs are optimized for hosting:

- JSON is written for the dashboard.
- PNG/SVG figures are rendered for quick download.
- Excel export has been removed from the deployed app path to avoid slow workbook generation.

For your latest one-year test run, removing Excel reduced the analysis runtime from about 51.5 seconds to about 15.4 seconds.

S3-compatible environment variables:

```bash
OBJECT_STORAGE_ENDPOINT=...
OBJECT_STORAGE_BUCKET=...
OBJECT_STORAGE_PREFIX=
OBJECT_STORAGE_ACCESS_KEY=...
OBJECT_STORAGE_SECRET_KEY=...
OBJECT_STORAGE_REGION=auto
```

## Oracle Cloud Always Free Sketch

```bash
sudo apt update
sudo apt install python3-pip python3-venv nginx -y

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export SECRET_KEY="replace-me"
export ADMIN_PASSWORD="replace-me"
gunicorn -w 2 -b 127.0.0.1:5050 server:app
```

Put Nginx in front of Gunicorn for public HTTPS hosting.

## Platform Start Command

The included `Procfile` is ready for platforms that support it:

```text
web: gunicorn server:app --bind 0.0.0.0:$PORT --workers 1 --timeout 300
```

## Hugging Face Spaces

Use a Docker Space. The included `Dockerfile` runs the Flask app on port `7860`, which is the standard Hugging Face Spaces web port.

Set these Space secrets:

```text
SECRET_KEY=replace-with-a-long-random-string
ADMIN_EMAIL=your-email@example.com
ADMIN_PASSWORD=your-password
SESSION_COOKIE_SECURE=true
OBJECT_STORAGE_ENDPOINT=https://your-account-id.r2.cloudflarestorage.com
OBJECT_STORAGE_BUCKET=era5-wave-data
OBJECT_STORAGE_PREFIX=
OBJECT_STORAGE_ACCESS_KEY=your-r2-access-key-id
OBJECT_STORAGE_SECRET_KEY=your-r2-secret-access-key
OBJECT_STORAGE_REGION=auto
```

Leave `OBJECT_STORAGE_PREFIX` blank because the NetCDF files were uploaded directly into the R2 bucket root.
