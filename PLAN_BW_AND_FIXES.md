# ASTRA: Black & White Theme + Frontend-Backend Connection Fixes

## Part 1: Black & White Theme

### Design Token System — Replace `globals.css :root` (Lines 8-17)

Replace all color accent variables with B&W tokens:

```css
:root {
  --bg-primary: #000000;
  --bg-secondary: #0a0a0a;
  --bg-card: rgba(15, 15, 15, 0.85);
  --bg-card-hover: rgba(25, 25, 25, 0.9);
  --border-default: rgba(100, 100, 100, 0.25);
  --border-hover: rgba(180, 180, 180, 0.4);
  --text-primary: #ffffff;
  --text-secondary: #aaaaaa;
  --text-muted: #666666;
  --glow-card: 0 0 20px rgba(255, 255, 255, 0.03);
  --glow-card-hover: 0 0 30px rgba(255, 255, 255, 0.06);
}
```

Remove `--accent-indigo`, `--accent-emerald`, `--accent-violet`, `--glow-indigo`, `--glow-emerald`.

### Component Classes — Rewrite in `globals.css` (Lines 60-268)

**`.glass-card:hover`** (L66-78): Use `--glow-card-hover` and `--border-hover` (auto-resolved once vars change).

**`.stat-card` accents** (L99-107): All 6 accent variants become gray gradients:
```
.indigo   → linear-gradient(90deg, #888, #aaa)
.emerald  → linear-gradient(90deg, #aaa, #ccc)
.blue     → linear-gradient(90deg, #999, #bbb)
.amber    → linear-gradient(90deg, #777, #999)
.rose     → linear-gradient(90deg, #666, #888)
.violet   → linear-gradient(90deg, #888, #aaa)
```

**Gradient text** (L110-128): Collapse all 4 gradient-text variants to single B&W:
```css
.gradient-text {
  background: linear-gradient(135deg, #ffffff 0%, #aaaaaa 100%);
}
```
Remove `.gradient-text-brand`, `.gradient-text-emerald` — replace usages with `.gradient-text`.

**Pulse dots** (L131-137): Remove green/red — all dots become white:
```css
.pulse-dot::after { background: rgba(255, 255, 255, 0.12); }
.pulse-dot.green::after, .pulse-dot.red::after { background: rgba(255, 255, 255, 0.12); }
```

**Nav items** (L140-156): Active states become white-based:
```css
.nav-item.active {
  background: rgba(255, 255, 255, 0.06);
  color: #ffffff;
  border: 1px solid rgba(255, 255, 255, 0.15);
}
.nav-item.active-emerald {
  background: rgba(255, 255, 255, 0.06);
  color: #ffffff;
  border: 1px solid rgba(255, 255, 255, 0.15);
}
```

**Buttons** (L158-175):
```css
.btn-primary {
  background: #ffffff;
  color: #000000;
  box-shadow: 0 2px 10px rgba(255, 255, 255, 0.08);
}
.btn-primary:hover { box-shadow: 0 4px 20px rgba(255, 255, 255, 0.15); }
.btn-emerald {
  background: transparent;
  color: #ffffff;
  border: 1px solid rgba(255, 255, 255, 0.3);
  box-shadow: none;
}
.btn-emerald:hover { background: rgba(255, 255, 255, 0.08); }
```

**Input focus** (L183-188): `border-color: rgba(99,102,241,0.5)` → `rgba(255,255,255,0.35)`, `box-shadow` → white 10%.

**Login background** (L233-241): Remove all colored radial gradients:
```css
.login-bg { background: #000000; }
```
Grid pattern (L243-252) is already grayscale — keep as-is.

### Tailwind Config — `dashboard/tailwind.config.js`

Remove the entire `brand` palette block (lines 12-24). No changes needed to animations.

### TSX Files — Systematic Color Class Replacement

Apply these find-and-replace patterns across all 14 TSX files:

| Find Pattern | Replace With |
|---|---|
| `text-indigo-400`, `text-brand-300`, `text-brand-400` | `text-gray-300` |
| `text-emerald-400`, `text-emerald-300` | `text-gray-300` |
| `text-rose-400`, `text-red-400`, `text-orange-400` | `text-gray-400` |
| `text-amber-400`, `text-yellow-400` | `text-gray-400` |
| `text-blue-400` | `text-gray-300` |
| `text-purple-400`, `text-violet-400` | `text-gray-300` |
| `text-green-400` | `text-gray-300` |
| `text-yellow-300` | `text-gray-300` |
| `bg-indigo-900/20`, `bg-indigo-900/30` | `bg-white/5` |
| `bg-emerald-900/20`, `bg-emerald-900/30` | `bg-white/5` |
| `bg-purple-900/50`, `bg-blue-900/50`, `bg-yellow-900/50` | `bg-white/5` |
| `bg-green-900/30`, `bg-green-900/20` | `bg-white/5` |
| `bg-red-900/20`, `bg-red-900/30` | `bg-white/5` |
| `bg-rose-500`, `bg-rose-500/10` | `bg-white/10` |
| `bg-emerald-500`, `bg-emerald-600` | `bg-white` with `text-black` |
| `bg-indigo-600`, `bg-indigo-600/...` | `bg-white` with `text-black` |
| `bg-yellow-600`, `bg-purple-600` | `bg-white` with `text-black` |
| `border-indigo-500`, `border-purple-500` | `border-white/40` |
| `border-green-500`, `border-green-600` | `border-white/40` |
| `border-red-800`, `border-yellow-800`, `border-green-800` | `border-white/15` |
| `bg-gradient-to-r from-green-600 to-emerald-600` | `bg-white/20` |
| `bg-gradient-to-r from-... to-...` | `bg-white/15` |
| `hover:text-indigo-400`, `hover:text-emerald-400` | `hover:text-white` |
| `ring-indigo-500`, `ring-emerald-500` | `ring-white/30` |
| `shadow-indigo-500/25` | `shadow-white/10` |

### Per-File Critical Specific Changes

#### `dashboard/app/dashboard/layout.tsx`
- L72: Logo gradient bg → `#ffffff` with black text
- L113: `text-brand-300` → `text-gray-300`
- L116: Gradient avatar → `rgba(255,255,255,0.06)`
- L122: `bg-emerald-500` dot → `bg-white/40`
- L127: `bg-rose-500` badge → `bg-white/10`

#### `dashboard/app/dashboard/page.tsx`
- L62-67: All `iconColor` → `text-gray-300`, `iconBg` → `rgba(255,255,255,0.06)`
- L80-83: All accent props → `accent-white`
- L95-96: Remove `color: 'emerald'/'blue'`, use grayscale
- L110-128: Action icons → white-based

#### `dashboard/app/dashboard/groups/page.tsx`
- `getStatusStyle()`: Replace color mapping with grayscale, add text prefixes:
  - TRAINING: `■` prefix, white/10 bg, white border
  - COMPLETED: `✓` prefix, white/5 bg, gray border
  - PAUSED: `⏸` prefix, white/3 bg, gray border
  - FAILED: `✖` prefix, white/3 bg, gray border
- L140: `text-indigo-400` → `text-gray-300`
- L159-165: Action buttons → all `bg-white/10 hover:bg-white/20`

#### `dashboard/app/dashboard/create/page.tsx`
- Replace `bg-gray-900`/`border-gray-800` with `bg-black`/`border-white/10`
- AI recommendation card: `bg-purple-900/50` → `bg-white/5`
- Badge colors (purple/yellow/blue sources) → all `bg-white/5 text-gray-300`
- Add model button: `bg-indigo-600` → `bg-white text-black`
- Register button: `bg-emerald-600` → `bg-white text-black`

#### `dashboard/app/dashboard/logs/page.tsx`
- Event color mapping: Replace emerald/blue/violet/rose → all `text-gray-300`
- Distinguish event types by left border thickness instead of color

#### `dashboard/app/login/page.tsx`
- L35-38: Login card `border-indigo-500/20 shadow-indigo-500/10` → white equivalents
- Role toggle: `bg-indigo-600` → `bg-white text-black`

#### `dashboard/app/client/layout.tsx`
- Mirror admin layout changes
- Active nav: `bg-white/6 border-l-2 border-white`
- Logo: Same B&W treatment
- Pulse dot: `bg-white/40`
- Badge: `bg-white/10`

#### `dashboard/app/client/page.tsx`
- Stat cards: Same B&W icon treatment as admin
- Trust score bar: `from-green-600 to-emerald-600` → `from-gray-600 to-gray-300`

#### `dashboard/app/client/groups/page.tsx`
- Group cards: `bg-gray-900 border-gray-800` → `bg-black border-white/10`
- Status badges: Same text-prefix grayscale system
- Join button: `bg-emerald-600` → `bg-white text-black`

#### `dashboard/app/client/training/page.tsx`
- Trust score conditional colors: Replace ternary with gray scale
  - Good trust: `text-gray-200 bg-white/5`
  - Medium trust: `text-gray-400 bg-white/3`
  - Low trust: `text-gray-500 bg-white/2`
- Status dots: `bg-green-500` → `bg-white/40`, `bg-red-500` → `bg-white/20`

#### `dashboard/app/client/recommendations/page.tsx`
- Source badges (purple/blue/yellow): All → `bg-white/5 text-gray-300`
- Apply button: `bg-emerald-600` → `bg-white text-black`

#### `dashboard/app/client/trust/page.tsx`
- Trust gauge gradient: `from-red/emerald/green` → `from-gray-700 to-gray-200`
- Score text: Color conditional → all `text-gray-200`
- Quarantine indicator: Red → `text-gray-400`

#### `dashboard/app/client/notifications/page.tsx`
- Priority backgrounds (red/yellow/green/blue): All → `border-white/15 bg-white/3`
- Read indicator: `bg-emerald-500` → `bg-white/40`

---

## Part 2: Frontend-Backend-Database Connection Fixes

### FIX 1: Docker — NEXT_PUBLIC_API_URL (CRITICAL)

**File:** `docker-compose.yml` (line ~49)

```yaml
# ADD this line next to the existing REACT_APP_API_URL:
environment:
  - REACT_APP_API_URL=http://fl_server:8000
  - NEXT_PUBLIC_API_URL=http://fl_server:8000    # ← ADD THIS
```

### FIX 2: Database — DB_PATH env var ignored (CRITICAL)

**File:** `src/astra/app/database.py` (lines 549-558)

```python
def get_db() -> AstraDB:
    global _db
    if _db is None:
        db_path = os.getenv("DB_PATH", "./astra.db")   # ← ADD THIS
        _db = AstraDB(db_path=db_path)                  # ← PASS IT
    return _db
```

### FIX 3: CORS — Allow Docker network origins (CRITICAL)

**File:** `src/astra/app/server_api.py` (lines 87-94)

Add `os.getenv("FRONTEND_URL", "")` to `allow_origins` list (filter empty strings). Set `FRONTEND_URL` env var for Docker deployments.

### FIX 4: Extended endpoints initialized with empty config (HIGH)

**File:** `src/astra/app/server_api.py` (lines 40-82)

Change `_register_extended_endpoints(app, {})` to `_register_extended_endpoints(app, config)` to pass the loaded config with Gemini API key.

### FIX 5: JWT env var naming mismatch (HIGH)

**File:** `.env.example` — Change `JWT_SECRET=...` to `SECRET_KEY=...`

**File:** `src/astra/infra/security/auth.py` — Add fallback: `os.getenv("SECRET_KEY", os.getenv("JWT_SECRET", "..."))`

### FIX 6: Health endpoint broken import (CRITICAL - from known bugs)

**File:** `src/astra/app/routes/system.py` — Fix broken `from networking.state import ...` import.

---

## Verification Steps

### Theme Verification
1. Start dashboard: `cd dashboard && npm run dev`
2. Check login page: Pure black background, no colored glows, B&W form
3. Check admin dashboard: Gray stat card accents, gray icons, no indigo
4. Check client dashboard: No emerald, trust bar grayscale
5. Check groups page: Status badges use text prefixes (■, ✓, ⏸, ✖) not color
6. Check training page: Trust score grayscale only
7. Check logs page: Events distinguished by border thickness, not color
8. Check notifications: Priority backgrounds all same gray
9. Hover states: All white/gray opacity changes, no colored glows

### Connection Verification
1. Docker: `docker-compose up`, dashboard loads without CORS errors
2. Database: Data persists across container restarts (check `/data/experiments.db`)
3. Health: `curl http://localhost:8000/health` returns 200
4. Auth: Login works with credentials
5. API: `curl http://localhost:8000/api/system/metrics` returns JSON

### Files Affected (19 files total)

| Category | File | Changes |
|---|---|---|
| Theme — CSS | `dashboard/app/globals.css` | Rewrite :root tokens, all @layer component classes |
| Theme — Config | `dashboard/tailwind.config.js` | Remove brand palette |
| Theme — Admin | `dashboard/app/dashboard/layout.tsx` | Logo, nav, avatar, pulse dot, badge |
| Theme — Admin | `dashboard/app/dashboard/page.tsx` | Stat cards, action icons |
| Theme — Admin | `dashboard/app/dashboard/groups/page.tsx` | Status badges, action buttons |
| Theme — Admin | `dashboard/app/dashboard/create/page.tsx` | Forms, AI cards, buttons, badges |
| Theme — Admin | `dashboard/app/dashboard/logs/page.tsx` | Event type indicators |
| Theme — Auth | `dashboard/app/login/page.tsx` | Login bg, role toggle, card |
| Theme — Client | `dashboard/app/client/layout.tsx` | Logo, nav, dot, badge |
| Theme — Client | `dashboard/app/client/page.tsx` | Stat cards, trust bar |
| Theme — Client | `dashboard/app/client/groups/page.tsx` | Group cards, status, buttons |
| Theme — Client | `dashboard/app/client/training/page.tsx` | Trust scores, status dots |
| Theme — Client | `dashboard/app/client/recommendations/page.tsx` | Source badges, buttons |
| Theme — Client | `dashboard/app/client/trust/page.tsx` | Trust gauge, conditional colors |
| Theme — Client | `dashboard/app/client/notifications/page.tsx` | Priority backgrounds, read markers |
| Backend | `docker-compose.yml` | Add NEXT_PUBLIC_API_URL |
| Backend | `src/astra/app/database.py` | get_db() reads DB_PATH |
| Backend | `src/astra/app/server_api.py` | CORS origins, pass config to extended endpoints |
| Backend | `.env.example` + `auth.py` | Standardize JWT env var name |
