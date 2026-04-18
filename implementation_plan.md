# AI Healing Gabès — Command Dashboard Implementation Plan

## Overview

A full-stack **"Command Center"** web application for the Gabès AI hackathon. The frontend is a Next.js (React) app styled with Tailwind CSS, built on top of the **TemplateMo 607 Glass Admin** design system (dark glassmorphism theme, neon emerald & cyan accents). The backend is a **Python Django + Django REST Framework (DRF)** server with:
- Role-based authentication (Worker / Technician / Admin)
- Incident log storage with photo attachments
- A full Django Admin panel for the Admin role
- Stub AI inference endpoints ready for real YOLO plugging

The three AI models already exist as `.pt` files:
| Model | File | Task |
|---|---|---|
| Industrial Safety (PPE) | `gazmask_detection.pt` + `helmet_best_yolov8s.pt` | Detect helmets & gas masks on workers |
| Marine Contamination | `FISH_DETECTION.pt` | Track live fish; flag phosphogypsum contamination |
| Land Regeneration | `mask_detection.pt` | Segment satellite imagery into oasis / dead-saline zones |

---

## Three User Roles

| Role | What they do |
|---|---|
| **Worker** | Runs AI agents, reviews results, presses "Report Incident" to save a log (with photo/video) to DB |
| **Technician** | Receives incident logs, views attached photos, labels each log: `Resolved` / `Non-Resolved` / `In Progress` |
| **Admin** | Full access — manages all user accounts, reads/creates/deletes any log, sees all activity |

---

## Proposed Architecture

```
/gabes-dashboard/
├── frontend/                    ← Next.js 14 App Router + Tailwind CSS
│   ├── app/
│   │   ├── page.tsx             (Overview dashboard)
│   │   ├── ppe/page.tsx         (Industrial Safety module)
│   │   ├── marine/page.tsx      (Marine Contamination module)
│   │   ├── land/page.tsx        (Land Regeneration module)
│   │   ├── logs/page.tsx        (Technician: incident log inbox)
│   │   ├── admin/page.tsx       (Admin: user & log management)
│   │   └── layout.tsx           (Sidebar + shared shell)
│   ├── components/
│   │   ├── Sidebar.tsx
│   │   ├── Navbar.tsx
│   │   ├── StatusCard.tsx
│   │   ├── AlertFeed.tsx
│   │   ├── GabesMap.tsx         (React-Leaflet map with zone pins)
│   │   ├── UploadPanel.tsx      (Reusable file/video upload)
│   │   ├── ResultViewer.tsx     (Annotated image/video result)
│   │   ├── ReportButton.tsx     (Worker: saves result as incident log)
│   │   └── LogCard.tsx          (Technician: shows log + label selector)
│   ├── styles/globals.css
│   └── tailwind.config.ts
│
└── backend/                     ← Python Django + DRF
    ├── manage.py
    ├── gabes_project/
    │   ├── settings.py
    │   ├── urls.py
    │   └── wsgi.py
    ├── apps/
    │   ├── accounts/            (Custom User model + role management)
    │   ├── logs/                (Incident Log model + CRUD API)
    │   └── ai_inference/        (PPE / Marine / Land stub endpoints)
    ├── media/                   (Uploaded photos/videos stored here)
    └── requirements.txt
```

---

## User Review Required

> [!IMPORTANT]
> Backend is switching from **FastAPI → Django + DRF**. This gives us a free Admin panel, built-in Auth, session/token management, ORM migrations, and file upload handling — all critical for the 3-role system.

> [!WARNING]
> The AI inference endpoints return **mock JSON stubs**. Real YOLO inference via `ultralytics` can be plugged in later — the `.pt` model files are already in `/models/`.

> [!NOTE]
> Auth will use **Django REST Framework + SimpleJWT** (token-based) so the Next.js frontend can call the API securely across origins.

---

---

## Step-by-Step Build Plan

### PHASE 1 — Django Backend

#### Step 1 — Scaffold the Django Project
```
pip install django djangorestframework djangorestframework-simplejwt django-cors-headers pillow
django-admin startproject gabes_project backend/
cd backend
python manage.py startapp accounts
python manage.py startapp logs
python manage.py startapp ai_inference
```

#### Step 2 — Custom User Model (`accounts` app)
Django's built-in User extended with a **role field**.

```python
# accounts/models.py
class CustomUser(AbstractUser):
    ROLE_CHOICES = [
        ('worker',     'Worker'),
        ('technician', 'Technician'),
        ('admin',      'Admin'),
    ]
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='worker')
```
- `settings.py` → `AUTH_USER_MODEL = 'accounts.CustomUser'`
- Register in Django Admin so the Admin can create/edit/delete accounts
- API endpoints: `POST /api/auth/login/` → returns JWT access + refresh tokens

#### Step 3 — Incident Log Model (`logs` app)

```python
# logs/models.py
class IncidentLog(models.Model):
    STATUS_CHOICES = [
        ('in_progress',   'In Progress'),
        ('resolved',      'Resolved'),
        ('non_resolved',  'Non Resolved'),
    ]
    MODULE_CHOICES = [
        ('ppe',    'Industrial Safety'),
        ('marine', 'Marine Contamination'),
        ('land',   'Land Regeneration'),
    ]
    created_by    = models.ForeignKey(CustomUser, on_delete=CASCADE, related_name='logs')
    module        = models.CharField(max_length=20, choices=MODULE_CHOICES)
    title         = models.CharField(max_length=255)
    description   = models.TextField(blank=True)
    photo         = models.ImageField(upload_to='logs/photos/', blank=True, null=True)
    result_data   = models.JSONField(default=dict)   # raw AI result JSON
    status        = models.CharField(max_length=20, choices=STATUS_CHOICES, default='in_progress')
    labeled_by    = models.ForeignKey(CustomUser, null=True, blank=True, on_delete=SET_NULL, related_name='labeled_logs')
    created_at    = models.DateTimeField(auto_now_add=True)
    updated_at    = models.DateTimeField(auto_now=True)
```

#### Step 4 — API Endpoints (DRF ViewSets)

| Method | URL | Role | Action |
|---|---|---|---|
| POST | `/api/auth/login/` | All | Get JWT token |
| POST | `/api/auth/refresh/` | All | Refresh JWT |
| GET | `/api/logs/` | Technician, Admin | List all logs |
| POST | `/api/logs/` | Worker, Admin | Create new log |
| GET | `/api/logs/{id}/` | Technician, Admin | Detail view |
| PATCH | `/api/logs/{id}/label/` | Technician, Admin | Set status label |
| DELETE | `/api/logs/{id}/` | Admin only | Delete log |
| GET | `/api/users/` | Admin only | List all users |
| POST | `/api/users/` | Admin only | Create user |
| DELETE | `/api/users/{id}/` | Admin only | Delete user |
| POST | `/api/ppe/analyze/` | Worker | Run PPE stub |
| POST | `/api/marine/analyze/` | Worker | Run Marine stub |
| POST | `/api/land/analyze/` | Worker | Run Land stub |

#### Step 5 — Permissions Layer (DRF custom permissions)
```python
# permissions.py
class IsWorker(BasePermission):      # can create logs, run AI
class IsTechnician(BasePermission):  # can read & label logs
class IsAdminRole(BasePermission):   # full access
```

#### Step 6 — Django Admin Panel Configuration
- Register `CustomUser` with UserAdmin (shows role, can change it)
- Register `IncidentLog` with list display: module, status, created_by, created_at
- Admin can filter by module, status, date
- Admin can inline-view attached photos

#### Step 7 — Media File Handling
- `settings.py`: `MEDIA_ROOT = BASE_DIR / 'media'`, `MEDIA_URL = '/media/'`
- `urls.py`: serve media in development
- Workers upload photo/video when creating a log; stored in `media/logs/photos/`

#### Step 8 — CORS & settings
- `django-cors-headers` allows Next.js on `localhost:3000` to call `localhost:8000`
- `settings.py`: `CORS_ALLOWED_ORIGINS = ['http://localhost:3000']`

#### Step 9 — Run migrations & seed data
```
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser  # creates the Admin user
```
Seed script creates 1 Worker + 1 Technician account for demo.

---

### PHASE 2 — Frontend (Next.js)

#### Step 10 — Auth Flow
- Login page (`/login`) → calls `POST /api/auth/login/` → stores JWT in httpOnly cookie or localStorage
- `middleware.ts` guards routes: redirects based on `role` in JWT payload

#### Step 11 — Worker View
- Runs AI agents on PPE / Marine / Land pages
- Sees results → clicks **"🚨 Report Incident"** button
- Opens a modal: pre-fills module/result, allows adding description + attaches photo
- Calls `POST /api/logs/` → log saved in DB

#### Step 12 — Technician View (`/logs`)
- See all incident logs in a card grid (photo thumbnail, module badge, status)
- Clicks a log → full detail modal with photo viewer
- 3 action buttons: **✅ Resolved** | **🔄 In Progress** | **❌ Non-Resolved**
- Calls `PATCH /api/logs/{id}/label/` → updates status in DB

#### Step 13 — Admin View (`/admin`)
- Tab 1: **User Management** — table of all users, add/delete, change role
- Tab 2: **All Logs** — filterable table, can delete any log, download photo
- Also has access to the Django Admin panel at `/django-admin/`

---

### PHASE 3 — Integration & Polish

#### Step 14 — Wire up Overview Dashboard
- Fetch log counts from `/api/logs/stats/` → show in KPI cards
- Alert feed pulls latest 10 logs with `status=in_progress`

#### Step 15 — Browser verification
- Test full login flow for each role
- Test Worker creating a log with photo
- Test Technician labeling a log
- Test Admin deleting a user

---

## Proposed Changes

### 1. Project Scaffold

#### [NEW] `gabes-dashboard/` (root)
Bootstrap the project with a creation script that sets up both `frontend/` and `backend/` directories.

---

### 2. Frontend — Next.js App

#### [NEW] `frontend/app/layout.tsx`
Global shell: animated background orbs, fixed Sidebar, top Navbar. Wraps all pages.

#### [NEW] `frontend/app/page.tsx` — Overview Dashboard
- **4 KPI cards**: Workers Monitored, Marine Checks Today, Satellites Analyzed, Active Alerts
- **Gabès interactive map** (React-Leaflet) with 3 zone pins (Industrial, Marine, Land) each showing live status pulse animations
- **Status Alerts feed**: scrollable list of real-time alerts (e.g., "⚠ Contamination Detected in Marine Zone")
- **System Status bar**: uptime, last scan timestamps for each module

#### [NEW] `frontend/app/ppe/page.tsx` — Industrial Safety (PPE)
- Drag-and-drop upload zone for **image or video**
- Calls `POST /api/ppe/analyze` with `multipart/form-data`
- Displays annotated result image with bounding boxes highlighted (using canvas overlay)
- Detection summary table: Worker ID, Helmet ✓/✗, Gas Mask ✓/✗, Risk Level badge
- Simulated "Live Camera Feed" toggle (webcam via `getUserMedia`)

#### [NEW] `frontend/app/marine/page.tsx` — Marine Contamination
- Live video feed simulation panel (with upload fallback)
- Fish tracking visualization with motion vector overlays
- Contamination status gauge (animated ring chart: Safe / Warning / Critical)
- Erratic motion threshold slider for demo tuning
- Historical contamination log table

#### [NEW] `frontend/app/land/page.tsx` — Land Regeneration
- Satellite image uploader (accept `.jpg`, `.png`, `.tif`)
- Side-by-side view: Original satellite image | Segmented overlay
- Legend: 🟢 Healthy Oasis, 🟡 Marginal, 🔴 Dead/Saline
- Zone statistics panel: % coverage, area in km², regeneration potential score
- Export results as PNG/CSV button

#### [NEW] `frontend/components/Sidebar.tsx`
Glassmorphism sidebar with logo "🌿 Gabès AI", nav sections: Overview, AI Modules (PPE, Marine, Land), Settings. Active state glow animation.

#### [NEW] `frontend/components/GabesMap.tsx`
React-Leaflet map centered on Gabès. Custom animated markers per zone, clicking opens a tooltip with module status.

#### [NEW] `frontend/components/UploadPanel.tsx`
Reusable drag-and-drop upload with progress bar, preview, and submit button.

#### [NEW] `frontend/components/ResultViewer.tsx`
Canvas-based annotated image viewer. Shows bounding boxes and labels from API response.

#### [NEW] `frontend/tailwind.config.ts`
Custom color tokens extending the glassmorphism palette:
- `emerald`: `#059669`, `emerald-light`: `#34d399`
- `neon-blue`: `#0ea5e9`, `neon-green`: `#22c55e`
- `glass-bg`, `glass-border` via CSS variables

---

### 3. Backend — FastAPI

#### [NEW] `backend/main.py`
FastAPI app with CORS enabled for `localhost:3000`. Mounts all routers.

#### [NEW] `backend/routers/ppe.py`
```
POST /api/ppe/analyze
  Input: multipart image or video file
  Output: {
    "status": "safe" | "violation",
    "detections": [
      { "worker_id": 1, "helmet": true, "gas_mask": false, "confidence": 0.92 }
    ],
    "annotated_image_url": "/static/results/ppe_result.jpg",
    "summary": "1 violation detected"
  }
```
Stub returns randomized mock data. TODO comment marks where `ultralytics` YOLO inference goes.

#### [NEW] `backend/routers/marine.py`
```
POST /api/marine/analyze
  Input: multipart video file
  Output: {
    "status": "contaminated" | "normal",
    "fish_count": 12,
    "erratic_count": 4,
    "contamination_probability": 0.87,
    "alert": "Phosphogypsum contamination likely"
  }
```

#### [NEW] `backend/routers/land.py`
```
POST /api/land/analyze
  Input: multipart satellite image
  Output: {
    "healthy_pct": 34.2,
    "marginal_pct": 28.5,
    "dead_pct": 37.3,
    "regeneration_score": 61,
    "annotated_image_url": "/static/results/land_result.jpg"
  }
```

#### [NEW] `backend/schemas.py`
Pydantic models for all request/response types.

#### [NEW] `backend/requirements.txt`
```
fastapi
uvicorn[standard]
python-multipart
pillow
```

---

## Theming Strategy

The glassmorphism CSS from `templatemo-glass-admin-style.css` will be adapted:
- CSS custom properties (`--emerald`, `--glass-bg`, etc.) defined in `globals.css`
- Tailwind extended with matching tokens
- Accent colors shifted: **emerald-green** (environment) + **neon-cyan** (tech) as primary accents
- Background: deep dark `#0a0f0d` with animated floating orbs in green/blue
- Cards: `backdrop-filter: blur(20px)`, glassmorphism borders

---

## Verification Plan

### Automated
- `npm run dev` — verify Next.js builds and serves on port 3000
- `uvicorn backend.main:app --reload` — verify FastAPI on port 8000
- API test: `curl -X POST http://localhost:8000/api/ppe/analyze -F "file=@test.jpg"` → valid JSON

### Browser Testing (via browser subagent)
1. Navigate to `http://localhost:3000` → verify Overview dashboard renders correctly
2. Click map zone markers → verify tooltip/status interaction
3. Navigate to `/ppe` → upload a test image → verify mock result renders with bounding boxes
4. Navigate to `/marine` → play simulated feed → verify contamination gauge animates
5. Navigate to `/land` → upload satellite image → verify segmentation overlay renders
6. Verify the UI looks premium (glassmorphism, animations, neon accents, responsive layout)

---

## Open Questions

> [!IMPORTANT]
> **Real model inference**: Do you want the backend stubs to also attempt real YOLO inference using `ultralytics` if a `.pt` model is present, or keep them purely as stubs for the hackathon demo?

> [!NOTE]
> **Map data**: I'll use OpenStreetMap / React-Leaflet for the Gabès map. Do you have specific GPS coordinates for the industrial zone, marine monitoring site, or oasis area you want pinned?

> [!NOTE]
> **Live webcam**: The PPE and Marine modules can optionally use the browser webcam (`getUserMedia`) for a live-feed demo. Should this be enabled by default?
