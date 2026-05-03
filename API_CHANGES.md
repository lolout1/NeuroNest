# NeuroNest API Changes

## May 2026 — Defect 3 (Sign & Clock Placement) — additive

A new top-level key `"placement"` is being added to the structured JSON returned as element `[4]` of `/analyze_wrapper`. **Existing keys (`"blackspots"`, `"contrast"`) are unchanged** — mobile-app integrations that ignore the new key continue to work.

### New schema

```json
"placement": {
  "count": 4,
  "violations": 2,
  "calibration": "door",
  "scale_factor": 1.03,
  "ada_recommended_range_in": [48, 60],
  "detections": [
    {
      "id": 1,
      "class": "sign",
      "class_id": 43,
      "centroid_px": [220, 410],
      "bbox": [200, 380, 250, 450],
      "area_pixels": 1240,
      "height_in": 38.2,
      "height_in_uncertainty": 4.1,
      "severity": "high",
      "violation_type": "below",
      "calibration_source": "door",
      "confidence": 0.91
    }
  ]
}
```

### Field reference

| Field | Type | Meaning |
|---|---|---|
| `count` | int | Total signs + clocks detected in the image. |
| `violations` | int | Count of detections with `severity != "ok"`. |
| `calibration` | string \| null | `"door"`, `"ceiling"`, or `"prior"` — what was used to scale the depth map. |
| `scale_factor` | float | Multiplier applied to raw depth output (1.0 = no calibration). |
| `ada_recommended_range_in` | `[float, float]` | Threshold band; `[48, 60]` per ADA / dementia design guidelines. |
| `detections[].class` | string | `"sign"` or `"clock"`. |
| `detections[].class_id` | int | Original ADE20K class id (43 for sign, 148 for clock). |
| `detections[].centroid_px` | `[int, int]` | `(cx, cy)` in image coordinates. |
| `detections[].bbox` | `[int, int, int, int]` | `[x1, y1, x2, y2]`. |
| `detections[].height_in` | float | Centroid height above floor plane, in inches. |
| `detections[].height_in_uncertainty` | float | ±inches, 1-sigma. |
| `detections[].severity` | string | `"critical"` (>6″ off), `"high"` (3–6″ off), `"medium"` (0–3″ off), or `"ok"`. |
| `detections[].violation_type` | string \| null | `"below"`, `"above"`, or `null` if `severity == "ok"`. |
| `detections[].calibration_source` | string | Which reference object this detection's scale derives from. |
| `detections[].confidence` | float | Analyzer self-assessed confidence (0..1). |

### Defensive consumption pattern

```python
data = result[4]

placement = data.get("placement")
if placement and not placement.get("skipped"):
    for det in placement.get("detections", []):
        if det["severity"] == "ok":
            continue
        print(f"{det['class']} #{det['id']}: {det['height_in']:.1f}\" "
              f"(\u00b1{det['height_in_uncertainty']:.1f}) "
              f"\u2014 {det['severity'].upper()} {det['violation_type']}")
```

### Skipped placement analysis

If the image lacks any sign/clock pixels, or the floor isn't visible enough to fit a plane, the field is:

```json
"placement": { "skipped": true, "reason": "no floor visible" }
```

Always check `placement.get("skipped")` before iterating `detections`.

---

## April 2026 — Breaking Change: `/analyze_wrapper` now returns 5 elements (was 4)

**Endpoint:** `/analyze_wrapper`  
**Base URL:** `https://lolout1-txstneuronest.hf.space/`

### Before (old)

```
Returns tuple of 4 elements:
[0] Image  — segmentation visualization
[1] Image  — blackspot visualization
[2] Image  — contrast visualization
[3] str    — markdown report
```

### After (new)

```
Returns tuple of 5 elements:
[0] Image  — segmentation visualization
[1] Image  — blackspot visualization
[2] Image  — contrast visualization
[3] str    — markdown report (redesigned, now includes color info + tables)
[4] dict   — STRUCTURED JSON (new) — machine-readable analysis data
```

**Element [4] is the new structured JSON.** This is what you want for the mobile app.

---

## Quick Start — Python

```python
from gradio_client import Client, handle_file

client = Client("https://lolout1-txstneuronest.hf.space/")
result = client.predict(
    image_path=handle_file("path/to/room_photo.jpg"),
    blackspot_threshold=0.5,
    contrast_threshold=4.5,
    enable_blackspot=True,
    enable_contrast=True,
    api_name="/analyze_wrapper"
)

seg_image    = result[0]   # file path to segmentation image
bs_image     = result[1]   # file path to blackspot image
ct_image     = result[2]   # file path to contrast image
report_md    = result[3]   # markdown string (human-readable report)
data         = result[4]   # dict — structured JSON (see schema below)
```

---

## Structured JSON Schema (element [4])

```json
{
  "blackspots": {
    "count": 3,
    "coverage_pct": 2.5,
    "avg_confidence": 0.87,
    "floor_area_pixels": 50000,
    "blackspot_area_pixels": 1250,
    "detections": [
      {
        "id": 1,
        "centroid": [420, 310],
        "area_pixels": 1500,
        "confidence": 0.92
      },
      {
        "id": 2,
        "centroid": [180, 520],
        "area_pixels": 800,
        "confidence": 0.85
      }
    ]
  },
  "contrast": {
    "total_issues": 5,
    "by_severity": {
      "critical": 1,
      "high": 2,
      "medium": 2
    },
    "issues": [
      {
        "severity": "critical",
        "surfaces": ["floor", "stairs"],
        "wcag_ratio": 2.3,
        "colors": {
          "surface_1": {
            "rgb": [180, 160, 140],
            "hex": "#b4a08c",
            "name": "tan"
          },
          "surface_2": {
            "rgb": [170, 155, 135],
            "hex": "#aa9b87",
            "name": "tan"
          }
        },
        "hue_difference": 5.2,
        "saturation_difference": 12,
        "boundary_pixels": 850
      }
    ]
  }
}
```

---

## How to Extract Blackspot Count

```python
data = result[4]

# Total count
num_blackspots = data["blackspots"]["count"]

# Coverage as percentage of floor area
coverage = data["blackspots"]["coverage_pct"]

# Per-instance details
for det in data["blackspots"]["detections"]:
    print(f"Blackspot #{det['id']}")
    print(f"  Location: ({det['centroid'][0]}, {det['centroid'][1]})")
    print(f"  Area: {det['area_pixels']} pixels")
    print(f"  Confidence: {det['confidence']:.0%}")
```

**Note:** If blackspot detection is unavailable (model not loaded), the `"blackspots"` key will be absent from the dict. Always check:

```python
if "blackspots" in data:
    count = data["blackspots"]["count"]
else:
    count = None  # detector not available
```

---

## How to Extract Contrast Issues

```python
data = result[4]

# Summary counts
total = data["contrast"]["total_issues"]
critical = data["contrast"]["by_severity"]["critical"]
high = data["contrast"]["by_severity"]["high"]
medium = data["contrast"]["by_severity"]["medium"]

# Iterate issues (sorted by severity: critical first, then high, then medium)
for issue in data["contrast"]["issues"]:
    severity = issue["severity"]          # "critical" | "high" | "medium"
    surfaces = issue["surfaces"]          # e.g. ["floor", "wall"]
    ratio    = issue["wcag_ratio"]        # e.g. 2.3 (lower = worse)

    # Colors causing the problem
    c1 = issue["colors"]["surface_1"]
    c2 = issue["colors"]["surface_2"]

    print(f"[{severity.upper()}] {surfaces[0]} <-> {surfaces[1]}")
    print(f"  Contrast ratio: {ratio}:1")
    print(f"  Color 1: {c1['name']} ({c1['hex']}) rgb{tuple(c1['rgb'])}")
    print(f"  Color 2: {c2['name']} ({c2['hex']}) rgb{tuple(c2['rgb'])}")
```

**Same check:** if contrast was disabled, `"contrast"` key will be absent.

---

## Severity Levels & Required Ratios

| Severity | Required Ratio | Triggered By |
|----------|---------------|-------------|
| `critical` | 7.0:1 | floor-stairs, floor-door, stairs-wall |
| `high` | 4.5:1 | floor-wall, floor-furniture, floor-fixtures, floor-objects, wall-door, wall-furniture, stairs-door |
| `medium` | 3.0:1 | all other adjacent surface pairs |

---

## Surface Categories

The `surfaces` array uses these category names:

| Category | What it includes |
|----------|-----------------|
| `floor` | floor, rug |
| `wall` | wall |
| `ceiling` | ceiling |
| `furniture` | bed, chair, sofa, table, desk, cabinet, bookcase, etc. (24 types) |
| `door` | door, screen door |
| `window` | window |
| `stairs` | stairs, stairway, bannister, escalator, step |
| `objects` | lamp, cushion, towel, toy, basket, bag, bottle, etc. (24 types) |
| `fixtures` | sink, toilet, tub, stove, refrigerator, shower, etc. (14 types) |
| `decorative` | painting, mirror, curtain, chandelier, blind, etc. (16 types) |

---

## Color Name Reference

The `name` field gives a human-readable approximation:

- **Neutrals:** black, dark gray, gray, light gray, white
- **Warm neutrals:** dark brown, brown, tan, beige
- **Chromatic:** red, orange, yellow, green, teal, blue, purple, pink
- **Modified:** dark [hue], light [hue], pale [hue]

---

## Other Endpoints (unchanged)

All other endpoints (`/lambda`, `/xai_wrapper`, `/defect_xai_wrapper`, `/load_sample_gallery`, `/_run_agent`, etc.) are unchanged.
