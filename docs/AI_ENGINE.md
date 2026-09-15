# The Campy AI engine

Everything here is implemented from first principles on NumPy. No PyTorch, no
TensorFlow, no third-party vision API.

## Why build it rather than call an API

Three reasons, all commercial rather than ideological:

1. **Footage cannot leave.** Banks, hospitals and plants — the customers who
   most need this — cannot lawfully stream camera feeds to a third-party
   service. That rules the approach out before accuracy is even discussed.
2. **Per-frame pricing does not survive video.** One camera at 6 fps is over
   half a million frames a day. Forty cameras and the API bill exceeds what
   anyone would pay for the product.
3. **Generic models do not know your site.** A model trained on internet
   photographs has no concept of your uniform, your machinery, or what
   "unsafe" means on your floor.

The cost is building the whole stack. The benefit is seven analytics running
together on ordinary CPU cores — roughly two to three cameras per core at the
default analysis resolution, entirely inside the customer's deployment. Measure
it on your own hardware with `python manage.py benchmark`.

## Layout

```
apps/aiengine/
  nn/              the neural-network framework
    config.py      working precision (float32 in production, float64 to verify)
    functional.py  im2col / col2im, stable softmax and sigmoid
    layers.py      Conv2D, DepthwiseConv2D, BatchNorm2D, pooling, Dense, …
    losses.py      cross-entropy, BCE, focal, MSE, Huber, batch-hard triplet
    optim.py       SGD/Nesterov, Adam, AdamW, step and cosine schedules
    model.py       Sequential: fit/predict/evaluate, .npz checkpoints
    metrics.py     accuracy, P/R/F1, confusion matrix, rank-1 retrieval
  vision/          image transforms, background model, optical flow, tracking
  detectors/       the seven analytics
  architectures.py campynet · campyface · campydet · campydense · campyseq
  pipeline.py      per-camera orchestration
  registry.py      model slots and resolution
  simulation.py    synthetic scenes for tests, demos and bootstrapping
```

## Correctness

A wrong gradient does not raise — it silently trains a worse model. So every
backward pass is checked against float64 central differences:

```bash
DJANGO_SETTINGS_MODULE=campyai.settings.test \
  python manage.py test apps.aiengine.tests.test_gradients
```

The framework's working precision is switchable precisely so this check is
meaningful: a 1e-6 perturbation is below float32's resolution, so the test
would otherwise be comparing rounding noise.

The Hungarian assignment solver is checked against brute-force optimal
assignment, and the detectors are checked against deterministic synthetic
scenes that must produce specific events — and quiet scenes that must stay
quiet.

## Three details that matter in the field

**Ghost rejection.** A subject present when the background model is first built
is baked into it. When they move off, the patch they vacated differs from the
model forever, and naive background subtraction tracks it as a motionless
second person. Two tests reject those: a real object is bounded by image edges
on most of its perimeter (`boundary_gradient_support`), and a real object keeps
changing frame to frame.

**Tracker ↔ background feedback.** Conversely, a person standing perfectly
still stops differing from the background and dissolves into the furniture.
Confirmed person tracks freeze the background model inside their box — bounded
by *demonstrated movement*, so a ghost can never earn that protection and the
loop cannot sustain a phantom.

**Proximity association.** IoU matching silently fails for anything moving
faster than its own width per frame: a running person's boxes do not overlap
between frames, so every frame spawns a new track and nothing is confirmed. A
second, size-gated centre-distance pass recovers exactly those cases.

## Physical units

Gesture thresholds are expressed in **body-heights per second**. Dividing by
the subject's own pixel height makes them independent of camera distance;
dividing by frame rate makes them independent of fps. Reference points:
walking ≈ 1.4 m/s and running ≈ 3 m/s against a ~1.7 m person give roughly
0.8 and 1.8 body-heights per second.

This is why the same configuration works on a doorway camera and a wide-angle
warehouse camera without retuning.

## Training

`apps/training/services.py` handles the full loop:

- **Stratified splits** so every class appears in validation — otherwise the
  validation score is meaningless for the rarest and most important class.
- **Inverse-frequency class weighting** applied automatically when classes are
  imbalanced. Without it a model scores 97% by always predicting "normal" and
  never detects the fire.
- **Augmentation** — flip, brightness, contrast, noise, shift. Customers train
  on a few hundred frames from a handful of cameras; without augmentation the
  model memorises one camera's lighting.
- **Live progress** streamed per epoch into the dashboard, and cancellable.
- **Self-describing checkpoints** — one `.npz` carrying the architecture beside
  the weights, so a deployed version can always be rebuilt from its file.

## Model slots

| Slot | Family | Used by |
|---|---|---|
| `campydet` | campydet | person/object detection for every analytic |
| `campyface` | campyface | identity embedding |
| `campynet_face` | campynet | face / not-face verification |
| `campynet_fire` | campynet | fire and smoke |
| `campynet_object` | campynet | object classification |
| `campydense` | campydense | crowd density |
| `campyseq_gesture` | campyseq | gesture and posture sequences |

Resolution order is: the workspace's own deployed model, then the Campy AI base
model, then the classical fallback. Nothing ever hard-fails for want of a
model — the analytic just runs its classical implementation instead.
