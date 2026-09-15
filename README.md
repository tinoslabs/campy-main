<div align="center">
  <img src="static/brand/campy-ai-logo.svg" alt="Campy AI" height="56">
  <h3>Turn ordinary CCTV into workplace intelligence</h3>
  <p>
    Gesture tracking · Face recognition · Geofencing · Crowd management<br>
    Theft detection · Dangerous objects · Fire &amp; smoke detection
  </p>
</div>

---

Campy AI is a multi-tenant SaaS platform that runs AI analytics on the CCTV
cameras an organisation already owns. It reads every frame as it happens and
turns it into plain-language findings a non-technical operator can act on.

**The AI is entirely ours.** Every layer, gradient, optimiser, training loop,
weights format and inference runtime is implemented from scratch on NumPy in
`apps/aiengine/`. No PyTorch, no TensorFlow, and no third-party vision API —
no frame ever leaves the deployment.

## Table of contents

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [How the AI works](#how-the-ai-works)
- [Architecture](#architecture)
- [Roles and permissions](#roles-and-permissions)
- [Packages and payments](#packages-and-payments)
- [Training your own models](#training-your-own-models)
- [REST API](#rest-api)
- [Privacy and compliance](#privacy-and-compliance)
- [Deployment](#deployment)
- [Handbooks](#handbooks)
- [Testing](#testing)
- [Performance](#performance)

## What it does

| Analytic | What it watches for |
|---|---|
| **Employee gesture tracking** | Walking, standing, sitting, bending, running, prolonged inactivity, loitering and falls. Speeds are measured in *body-heights per second*, so thresholds hold at any camera distance and any frame rate. |
| **Face recognition** | Identifies enrolled employees so alerts say *who*, not just *what*. Consent-gated; enrolment stores a 128-D unit vector, never a photograph. |
| **Geofencing** | Restricted, hazard, counting, shelf and exclusion zones drawn over the camera view. Containment tests a person's **feet**, not their centroid — somebody standing just outside a line is not counted as inside it. |
| **Crowd management** | Headcount, local density and **flow coherence** — the difference between a busy corridor and a crush risk. Falls back to density-map regression when heads occlude each other. |
| **Theft detection** | Shelf dwell, concealment gestures, carry-change and stock disappearance combined into a decaying risk score. Findings are always phrased as review prompts, never accusations. |
| **Dangerous & unattended objects** | Watch-list classification *plus* geometric abandoned-object detection, which catches items no class was ever trained for. |
| **Fire & smoke** | YCrCb/HSV chromatic gating, flicker variance and area growth, adjudicated by a learned classifier. Sees fire the moment it is in frame — including high ceilings and outdoor areas where a smoke sensor is useless. |

## Quick start

```bash
git clone <this-repo> campyai && cd campyai

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python manage.py setup        # migrations + demo workspace, safe to re-run
python manage.py runserver
```

`setup` prints the accounts to sign in with. Prefer the steps separately, or
want an empty system?

```bash
python manage.py migrate            # tables only
python manage.py setup --no-demo    # tables, no demo content
python manage.py createsuperuser    # your own admin — asks for an email
python manage.py seed_demo          # demo content on its own
cp .env.example .env                # optional for local development
```

Requires **Python 3.10 or newer** — the suite is run on 3.10, 3.11 and 3.12.

Open <http://localhost:8000>. Sign in at `/accounts/login/`:

| Role | Email | Password |
|---|---|---|
| Super administrator | `admin@campy.ai` | `CampyDemo!2024` |
| Workspace owner | `owner@acme-demo.com` | `CampyDemo!2024` |
| Security manager | `security@acme-demo.com` | `CampyDemo!2024` |
| Control-room operator | `operator@acme-demo.com` | `CampyDemo!2024` |
| AI engineer | `ai@acme-demo.com` | `CampyDemo!2024` |
| Viewer | `viewer@acme-demo.com` | `CampyDemo!2024` |

> Sign in as each of these to see role-based access in action — the navigation,
> the buttons and the API surface all change.

The demo workspace ships with **simulated cameras**, so every detector can be
watched working before any hardware is connected. Open a camera and press
**Run analysis** to push frames through the real pipeline and generate real
events.

### Connecting a real camera

Easiest is **ONVIF**: give the camera's IP address, port and credentials and it
reports its own stream URL, so you do not need to know the vendor's RTSP path.

Otherwise add a camera with protocol **RTSP**. Put the address in the URL field
and the login in the username and password fields — if you paste a URL that
already carries `user:pass@`, Campy AI separates them for you, because sending
the login twice is a 401 that looks exactly like a wrong password. Example URLs:

```
rtsp://user:pass@192.168.1.50:554/Streaming/Channels/101   # Hikvision
rtsp://user:pass@192.168.1.50:554/cam/realmonitor?channel=1&subtype=0   # Dahua
http://192.168.1.50/video.mjpg                              # MJPEG
```

Then run the worker:

```bash
python manage.py run_camera --camera <uid>          # one camera
python manage.py run_camera --all --max-frames 500  # every active camera
```

Prefer a **sub-stream** where the camera offers one. Analysis runs at 480×270,
so decoding a 4K main stream spends a core on detail the first resize discards.
ONVIF discovery picks the sub-stream for you.

## How the AI works

### The framework (`apps/aiengine/nn/`)

A complete neural-network library written from first principles:

- **Layers** — `Conv2D` and `DepthwiseConv2D` (im2col + GEMM), `BatchNorm2D`,
  `MaxPool2D`, `GlobalAvgPool2D`, `Dense`, `Dropout`, `L2Normalize`, and the
  usual activations. Explicit `forward`/`backward` per layer, no tape.
- **Losses** — cross-entropy (with label smoothing and class weighting),
  binary cross-entropy, focal, MSE, Huber and **batch-hard triplet**.
- **Optimisers** — SGD with Nesterov momentum, Adam, AdamW with decoupled
  decay, global-norm gradient clipping, step and cosine schedules.
- **Checkpoints** — one self-describing `.npz` per model version carrying the
  architecture alongside the weights, so a deployed model can always be
  rebuilt from its file alone.

**Every backward pass is verified against float64 central differences** in
`apps/aiengine/tests/test_gradients.py`. If a gradient were wrong, training
would silently produce a worse model rather than failing — so we prove each
one.

### The vision stack (`apps/aiengine/vision/`)

Bilinear resampling, HSV/YCrCb conversion, an adaptive per-pixel Gaussian
background model, morphology, run-based connected components, Lucas–Kanade
optical flow, HOG/LBP descriptors, polygon geometry, and a Kalman + Hungarian
multi-object tracker. OpenCV is used **only** to decode RTSP; never to analyse.

Three details that matter more than they sound:

- **Ghost rejection.** A subject present when the background model is first
  built leaves a permanent phantom at their starting position once they move.
  Boundary-gradient validation and a static-pixel test reject those: a real
  object is bounded by image edges, a vacated patch of background is not.
- **Tracker ↔ background feedback.** A person standing perfectly still would
  otherwise dissolve into the furniture. Confirmed person tracks freeze the
  background model inside their box — bounded by *demonstrated movement*, so
  the loop cannot sustain a phantom.
- **Proximity association.** IoU matching silently drops anything moving
  faster than its own width per frame — a running person's boxes simply do not
  overlap. A second, size-gated distance pass recovers exactly those.

### The architectures (`apps/aiengine/architectures.py`)

| Family | Purpose | Typical size |
|---|---|---|
| `campynet` | Image classifier (fire/smoke, PPE, object state) | ~45k params |
| `campyface` | L2-normalised identity embedding | ~69k params |
| `campydet` | Single-shot grid detector | ~185k params |
| `campydense` | Crowd density map | ~27k params |
| `campyseq` | Temporal action model | ~11k params |

Small on purpose: many streams per box on ordinary CPU hardware beats one
enormous backbone at 0.4 fps.

## Architecture

```
apps/
  core/        abstract models, utils, template filters, error pages
  accounts/    Organization (tenant), User, Role, Membership, APIKey, AuditLog
  aiengine/    nn/ · vision/ · detectors/ · architectures · pipeline · registry
  cameras/     Site, Camera, Zone, Employee · frame sources · worker
  events/      Event, Incident, AlertRule, Channel · dedupe, grouping, routing
  training/    Dataset, Sample, TrainingJob, ModelVersion · trainer
  analytics/   DailyMetric rollups, Report builder
  billing/     Package, Subscription, Invoice, Payment · Razorpay + Stripe
  cms/         Pages with blocks, blog, media, menus, leads, site settings
  dashboard/   the operator UI and the platform console
  api/         DRF v1 with per-action permissions
```

One `CameraPipeline` per camera. Shared per-frame stages — background model,
detection, tracking — run **once** and feed all seven analytics, which is why
seven cost barely more than one.

```
frame → background model → detection → tracking → analytics → events → alerts
```

## Roles and permissions

54 permissions across 16 modules, with eight built-in roles:

| Role | Rank | For |
|---|---|---|
| Owner | 0 | Full control including billing and closure |
| Administrator | 10 | Everything except closing the account |
| Security Manager | 20 | Cameras, geofences, incidents, escalation |
| HR Manager | 30 | Employee directory, attendance, insight |
| Control-Room Operator | 40 | Live wall, triage, acknowledge |
| AI Engineer | 40 | Datasets, training, model promotion |
| Auditor | 60 | Read-only plus the compliance trail |
| Viewer | 80 | Read-only |

Custom roles are built from the same permission checkboxes. Three properties
are enforced and tested:

- **No privilege escalation.** Nobody may grant a role more senior than their
  own; a workspace can never lose its last owner.
- **Tenancy isolation.** Owning one workspace grants *nothing* in another.
- **Feature gating.** Holding a permission is not enough — the workspace's
  package must also include the feature behind it.

## Packages and payments

Packages combine a **16-feature matrix** with **8 quotas** (cameras, sites,
users, employees, storage, retention, training jobs, API calls; `-1` means
unlimited), priced in both INR and USD with optional per-camera overage.

- **Razorpay** for India — cards, UPI, net banking, wallets. Inline checkout,
  with the returned signature verified **server-side** before anything is
  activated.
- **Stripe** for everywhere else — hosted Checkout, with the session confirmed
  against Stripe's API rather than trusting the browser's redirect.

Both sit behind one interface, so no other code branches on which is in use.
Webhooks are signature-verified and stored with a unique event id, so a
gateway retry is processed exactly once.

A **past-due subscription keeps working for seven days.** Cutting a security
system off the moment a card fails is the wrong behaviour; the dashboard warns
instead.

## Training your own models

1. **Create a dataset** — name the classes, put the "nothing happening" class
   first.
2. **Add samples** — upload frames, import dismissed false alarms with one
   click, or generate bootstrap samples to prove the pipeline works today.
3. **Label** — click-select in a grid and bulk-apply.
4. **Train** — pick an architecture and hyper-parameters, then watch loss and
   accuracy climb live, epoch by epoch.
5. **Deploy** — promote a version to a model slot; every camera using that
   slot picks it up immediately. Roll back just as fast.

Two details that make trained models actually work in the field:

- **Inverse-frequency class weighting** is applied automatically when classes
  are imbalanced — otherwise a model scores 97% by always predicting "normal"
  and never detects the fire.
- **The false-alarm loop closes.** Every alert an operator dismisses becomes a
  hard negative in the next training run, so the noise floor drops *because*
  people used the system.

## REST API

`/api/v1/` — browsable, with OpenAPI at `/api/v1/schema/` and Swagger UI at
`/api/v1/docs/`.

```bash
curl https://your-host/api/v1/events/ \
  -H "Authorization: Api-Key cai_xxxxxxxx.your-secret"
```

Keys are workspace-scoped and carry only the permissions granted at issue.
JWT (`/api/v1/auth/token/`) is also supported for user-context calls.

**Edge ingest** — run Campy AI's engine on your own hardware (or your own
detector entirely) and still get incident grouping, alert routing and the full
dashboard:

```bash
curl -X POST https://your-host/api/v1/events/ingest/ \
  -H "Authorization: Api-Key cai_xxxxxxxx.secret" \
  -H "Content-Type: application/json" \
  -d '{"camera":"<uid>","analytic":"fire","event_type":"fire_detected",
       "severity":"critical","title":"Fire detected","confidence":0.93}'
```

Face embeddings and stream credentials are **never** serialised by the API.

## Privacy and compliance

Monitoring people is a responsibility, not just a feature.

- **Consent-gated biometrics.** Face enrolment stays disabled for a person
  until written consent is recorded against their profile. Erasure is one
  click and is written to the audit log.
- **No photographs retained.** Enrolment stores a unit vector; a face cannot
  be reconstructed from it.
- **Face pixelation.** Unenrolled faces can be mosaicked in every stored
  snapshot — often what makes deployment lawful in a shared space.
- **Enforced retention.** Events and snapshots are deleted on a schedule, with
  the window set per workspace by its package.
- **Audit trail.** Every sensitive action is recorded with actor, IP and
  user-agent, and can be exported.
- **Non-accusatory language.** Theft findings are worded as prompts to review
  footage, never as conclusions about an individual — as the product
  specification requires.

## Deployment

```bash
pip install -r requirements-prod.txt
export DJANGO_SETTINGS_MODULE=campyai.settings.prod

python manage.py migrate
python manage.py collectstatic --noinput
gunicorn campyai.wsgi:application --bind 0.0.0.0:8000 --workers 4
```

Background workers (optional but recommended):

```bash
celery -A campyai worker  -l info
celery -A campyai beat    -l info    # health sweeps, rollups, retention, billing
```

Without Redis, Celery runs every task inline — the platform works, just
synchronously. See `deploy/` for Docker and Compose files, and
`docs/DEPLOYMENT.md` for the full guide including gateway webhook setup.

## Handbooks

Two PDFs in `docs/handbooks/`, both covering how a camera is connected and what
happens to every frame — written for different readers:

| | For | Covers |
|---|---|---|
| [**Technical Handbook**](handbooks/Campy-AI-Technical-Handbook.pdf) | Installers, integrators, engineers | Network topology, the six protocols, ONVIF discovery and its SOAP calls, credential handling, the worker loop and reconnection, the per-frame pipeline stage by stage, zones, capacity planning, training, troubleshooting |
| [**Plain English Handbook**](handbooks/Campy-AI-Plain-English-Handbook.pdf) | Owners, managers, security teams | What we connect to and what we don't need, installation day, how it works in six steps, the seven analytics with their honest weaknesses, alert fatigue, privacy and what to tell staff, limits, FAQ |

Rebuild them after editing the HTML sources beside them:

```bash
pip install playwright pymupdf requests && playwright install chromium
python docs/handbooks/build.py
```

## Testing

```bash
DJANGO_SETTINGS_MODULE=campyai.settings.test python manage.py test
```

131 tests covering:

- **Gradient correctness** — every layer and loss against float64 central
  differences, plus proof that the networks actually learn.
- **Analytic acceptance** — deterministic synthetic scenes that must produce
  specific events, *and* quiet scenes that must stay quiet.
- **RBAC** — per-role permission matrices, view-layer enforcement, tenancy
  isolation, feature gating and privilege-escalation guards.
- **Billing** — quotas, invoicing with GST, gateway signature verification
  (including tampered signatures), and webhook idempotency.
- **Platform** — the camera worker on a live pipeline, event de-duplication,
  incident grouping, alert routing and cooldowns, real model training,
  deployment and rollback, and API authentication and scoping.

## Performance

Run it yourself — the numbers below come from this command, and depend on your
hardware, so measure on the machine you plan to deploy on:

```bash
python manage.py benchmark
python manage.py benchmark --width 320 --height 180
python manage.py benchmark --analytics gesture,geofence,fire
```

On one core of a 2.1 GHz Xeon, all seven analytics at the default 480×270
analysis resolution:

| Scene | ms/frame | fps per core | Cameras per core @ 6 fps |
|---|---|---|---|
| Empty room | 50 | 20 | 3.3 |
| One person walking | 50 | 20 | 3.3 |
| Six people moving | 73 | 14 | 2.3 |
| Fourteen people | 64 | 16 | 2.6 |

**Size for the busy case: roughly 2–3 cameras per core** with everything
switched on. Where that time goes, on the busiest scene:

| Stage | ms |
|---|---|
| Detection (people & objects) | 25.0 |
| Background model | 16.5 |
| Fire & smoke | 13.6 |
| Dangerous objects | 2.7 |
| Face recognition | 1.7 |
| Tracking | 1.7 |
| Gesture | 0.4 |
| Crowd, theft, geofencing | < 0.2 each |

### Reports with evidence

Any period can be downloaded as a PDF — one feature on its own, or all of them
compared side by side — from **Analytics**, from a filtered **Events** view, or
from a saved report defined as PDF. Each carries the frame that produced every
finding, with the detection outlined and the zone it crossed drawn on it, plus
camera, zone, time, confidence and who it was identified as.

Reports state their own limits: theft findings describe behaviour and never
intent, fire detection complements a fire alarm rather than replacing one, and
a report that may contain images of identifiable people says so.

```
/app/reports/analysis.pdf?analytics=fire&days=30          # one feature
/app/reports/analysis.pdf?days=30                         # all of them
/app/reports/analysis.pdf?analytics=fire,geofence&camera=<uid>&evidence=24
```

### Watching a camera

One reader per camera holds the stream open and every viewer reads its newest
frame, so a wall of screens costs the camera a single session rather than one
per image. A camera the analytics worker is already running is not opened a
second time at all — the worker hands its frames over, and encoding only
happens while somebody is watching. Readers close themselves once nobody has
looked for 20 seconds.

| Setting | Default | What it does |
|---|---|---|
| `CAMPY_LIVE_STREAM_SECONDS` | 300 | How long one MJPEG response lasts before the page reopens it |
| `CAMPY_LIVE_FIRST_FRAME_TIMEOUT` | 10 | How long to wait for a stream's first frame |

Two dials change the answer far more than anything else:

- **Analysis resolution.** Cost is roughly quadratic in it. At 320×180 the same
  seven analytics run in 22 ms on a quiet camera and 36 ms on a busy one —
  about 5–7 cameras per core. Set `CAMPY_WORKER_FRAME_WIDTH` / `_HEIGHT`.
- **Which analytics are on.** Fire & smoke is the most expensive single
  analytic; geofencing, crowd and theft are nearly free once tracking has run.
  A camera with gesture, geofencing and fire costs 41 ms quiet, 61 ms busy.

A trained 45k-parameter classifier infers in about 0.7 ms, so the model is
never the bottleneck — reading pixels is.

---

<div align="center">
  <sub>Campy AI · built to make the cameras you already own actually useful</sub>
</div>
