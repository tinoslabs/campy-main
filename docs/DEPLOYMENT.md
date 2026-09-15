# Deploying Campy AI

## 1. Requirements

| | Minimum | Recommended |
|---|---|---|
| CPU | 2 cores | 8+ cores (≈2–3 cameras per core at 6 fps, all analytics on) |
| RAM | 2 GB | 8 GB |
| Disk | 20 GB | 200 GB+ (evidence snapshots dominate) |
| Python | 3.10 | 3.11 or 3.12 |
| Database | SQLite (evaluation only) | PostgreSQL 14+ |
| Cache/queue | none | Redis 6+ |

Camera analytics is CPU-bound and scales horizontally: add processes, shard by
site with `--site`, and put each group on its own machine if needed.

Size the CPU from a measurement, not from this table — `python manage.py
benchmark` runs the real pipeline on your hardware and prints cameras per
core. Two settings move that number more than anything else: the analysis
resolution (`CAMPY_WORKER_FRAME_WIDTH` / `_HEIGHT`, cost is roughly quadratic
in it) and which analytics each camera has switched on.

## 2. Install

```bash
git clone <repo> /opt/campyai && cd /opt/campyai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-prod.txt

cp .env.example .env    # then edit it — see below
export DJANGO_SETTINGS_MODULE=campyai.settings.prod

python manage.py migrate
python manage.py collectstatic --noinput
python manage.py seed_demo --skip-demo-org   # packages + CMS, no demo data
python manage.py createsuperuser
```

Generate a real secret key:

```bash
python -c "import secrets; print(secrets.token_urlsafe(64))"
```

## 3. Run

```bash
gunicorn campyai.wsgi:application --bind 0.0.0.0:8000 --workers 4 --timeout 120
celery -A campyai worker -l info
celery -A campyai beat   -l info
python manage.py run_camera --all --workers 8
```

Or use `deploy/docker-compose.yml`, which wires all five together.

Put TLS in front of it. Django is configured to trust `X-Forwarded-Proto`, so
a standard reverse proxy needs no extra work.

## 4. Payment gateways

### Razorpay

1. Dashboard → **Settings → API Keys** → generate. Put the key id and secret in
   `.env`.
2. Dashboard → **Settings → Webhooks** → add
   `https://your-host/webhooks/razorpay/`.
3. Subscribe to `payment.captured`, `payment.failed`, `order.paid`,
   `subscription.charged`, `subscription.cancelled`.
4. Copy the webhook secret into `RAZORPAY_WEBHOOK_SECRET`.

The signature on the browser's callback is verified server-side before
anything is activated, and every webhook is verified again independently.

### Stripe

1. Dashboard → **Developers → API keys**. Put the secret and publishable keys
   in `.env`.
2. **Developers → Webhooks** → add `https://your-host/webhooks/stripe/`.
3. Subscribe to `checkout.session.completed`, `payment_intent.succeeded`,
   `payment_intent.payment_failed`, `invoice.paid`,
   `customer.subscription.deleted`.
4. Copy the signing secret into `STRIPE_WEBHOOK_SECRET`.

Returning from Stripe Checkout does not activate anything on its own: the
session is confirmed against Stripe's API first.

### Testing without keys

Leave both unset. Checkout then offers "Request an invoice", which raises a
real invoice for offline settlement — useful for pilots and enterprise deals.

## 5. Base models

Give new workspaces a working detector before they have labelled anything:

```bash
python manage.py train_base_models --slot campynet_fire --epochs 30 --deploy
python manage.py train_base_models --slot campynet_face --epochs 30 --deploy
```

These train on generated data. Retrain them on real footage before relying on
them in production — the command warns about this.

## 6. Camera connection reference

| Vendor | URL |
|---|---|
| Hikvision | `rtsp://user:pass@HOST:554/Streaming/Channels/101` |
| Dahua | `rtsp://user:pass@HOST:554/cam/realmonitor?channel=1&subtype=0` |
| Axis | `rtsp://user:pass@HOST/axis-media/media.amp` |
| Generic ONVIF | `rtsp://user:pass@HOST:554/live/ch0` |
| MJPEG | `http://HOST/video.mjpg` |

Use the **sub-stream** where one exists. Analysis runs at 480×270 anyway, so a
1080p main stream just wastes bandwidth and decode time.

Test from the server before adding the camera:

```bash
ffprobe -rtsp_transport tcp "rtsp://user:pass@HOST:554/path"
```

## 7. Scaling

- **Web** — stateless; run several behind a load balancer.
- **Camera workers** — shard by site:
  `python manage.py run_camera --site <uid> --workers 8`.
- **Tuning cost** — `target_fps` per camera is the main dial. 4 fps is fine for
  corridors; use 8-10 only where fast motion matters.
- **Storage** — evidence snapshots dominate. Retention is enforced nightly by
  `apps.events.tasks.purge_expired_evidence`; keep the window as short as your
  policy allows.

## 8. Backups

```bash
pg_dump -Fc campyai > campyai-$(date +%F).dump
tar czf campyai-models-$(date +%F).tar.gz /var/lib/campyai/models
```

Trained model weights live on disk and are **not** in the database. Back up
both, or a restore leaves every deployed model missing.

## 9. Operational checks

- `GET /healthz` — database and model-directory reachability; returns 503 when
  degraded, so a load balancer can act on it.
- **Audit log** (`/app/audit/`) — every sensitive action, exportable.
- **Camera health** — `sweep_camera_health` marks a camera offline after ten
  minutes without a frame.
- **Failed payments** — visible on the platform console; `expire_due_subscriptions`
  moves past-due workspaces through the grace period nightly.
