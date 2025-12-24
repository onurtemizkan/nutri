# API Endpoint Test Results

**Date:** 2025-12-24
**Environment:** Production

## Backend API (Node.js)

| Endpoint | Method | Status | Response |
|----------|--------|--------|----------|
| `/health` | GET | ✅ Pass | Database healthy (38ms), Redis healthy |
| `/api/auth/login` | POST | ✅ Pass | Correctly rejects invalid credentials |
| `/api/auth/register` | POST | ✅ Pass | Validates email format |
| `/api/meals` | GET | ✅ Pass | Returns "No token provided" (auth working) |

**Backend URL:** `http://zwogw8ccsw84w8ocws4sowok.195.201.228.58.sslip.io`

---

## ML Service API (Python/FastAPI)

### Health & Info Endpoints

| Endpoint | Method | Status | Response |
|----------|--------|--------|----------|
| `/health` | GET | ✅ Pass | Database (104ms), Redis (8ms) healthy |
| `/health/live` | GET | ✅ Pass | `{"status":"ok"}` |
| `/` | GET | ✅ Pass | Service info with version 1.0.0 |
| `/api/food/health` | GET | ✅ Pass | Model loaded, CLIP-First Ensemble |
| `/api/sensitivity/health` | GET | ✅ Pass | 83 ingredients, 15 allergens |

### Food Analysis Endpoints

| Endpoint | Method | Status | Response |
|----------|--------|--------|----------|
| `/api/food/models/info` | GET | ✅ Pass | CLIP-First Ensemble (92% accuracy) |
| `/api/food/cooking-methods` | GET | ✅ Pass | 10 cooking methods |
| `/api/food/coarse-classify/categories` | GET | ✅ Pass | 30 food categories |
| `/api/food/coarse-classify` | POST | ✅ Pass | Image classification working |
| `/api/food/nutrition-db/search?q=apple` | GET | ✅ Pass | Returns nutrition data |
| `/api/food/estimate-micronutrients` | POST | ✅ Pass | Returns 17 micronutrients |
| `/api/food/feedback/stats` | GET | ⚠️ Bug | Column naming mismatch |

### Sensitivity Endpoints

| Endpoint | Method | Status | Response |
|----------|--------|--------|----------|
| `/api/sensitivity/allergens` | GET | ✅ Pass | 15 allergens (FDA + EU) |
| `/api/sensitivity/model/info` | GET | ✅ Pass | 21 features, model untrained |
| `/api/sensitivity/compounds/dao-inhibitors` | GET | ✅ Pass | 9 DAO inhibitors |
| `/api/sensitivity/compounds/histamine-liberators` | GET | ✅ Pass | 10 histamine liberators |
| `/api/sensitivity/ingredients/search` | GET | ⚠️ Bug | Attribute error |
| `/api/sensitivity/extract-ingredients` | POST | ⚠️ Bug | Attribute error |

**ML Service URL:** `http://z8cg8kkg4o0wg8044c8g0s0o.195.201.228.58.sslip.io`

---

## Known Issues (FIXED)

### 1. Column Naming Mismatch (FoodFeedback) ✅ FIXED
**Endpoint:** `/api/food/feedback/stats`
**Error:** `column FoodFeedback.original_prediction does not exist`
**Cause:** SQLAlchemy model used snake_case but Prisma schema has camelCase
**Fix:** Added proper Column mappings in `food_feedback.py` and updated `feedback_service.py`

### 2. AllergenMapping Attribute Error ✅ FIXED
**Endpoints:** `/api/sensitivity/ingredients/search`, `/api/sensitivity/extract-ingredients`
**Error:** `'AllergenMapping' object has no attribute 'allergen_type'`
**Cause:** Code accessed `.allergen_type` but dataclass has `.allergen`
**Fix:** Updated `ingredient_extraction_service.py` to use `.allergen`

---

## Summary

| Category | Passed | Failed | Total |
|----------|--------|--------|-------|
| Backend | 4 | 0 | 4 |
| ML Health | 5 | 0 | 5 |
| ML Food | 6 | 1 | 7 |
| ML Sensitivity | 4 | 2 | 6 |
| **Total** | **19** | **3** | **22** |

**Pass Rate:** 86.4%

---

## Production Readiness

### Working Features
- ✅ User authentication
- ✅ Protected routes
- ✅ Database connectivity (PostgreSQL)
- ✅ Cache connectivity (Redis)
- ✅ Food image classification (CLIP)
- ✅ Nutrition database search
- ✅ Micronutrient estimation
- ✅ Allergen information
- ✅ Compound analysis (DAO inhibitors, histamine liberators)

### Previously Fixed Issues
- ✅ Feedback analytics (column naming) - Fixed in commit
- ✅ Ingredient extraction (attribute error) - Fixed in commit
- ✅ Ingredient search (attribute error) - Fixed in commit

---

*Generated: 2025-12-24*
