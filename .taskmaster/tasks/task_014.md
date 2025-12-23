# Task ID: 14

**Title:** Implement Barcode Scanner with Open Food Facts Integration

**Status:** done

**Dependencies:** None

**Priority:** high

**Description:** Add barcode scanning capability to the food logging flow using the device camera and integrate with Open Food Facts API for instant nutrition lookup. This addresses a critical competitive gap - MyFitnessPal's barcode scanner is one of their most-used features.

**Details:**

## Overview
Barcode scanning is the #1 most requested feature for calorie tracking apps. Users expect instant food lookup by simply pointing their camera at a product barcode.

## Technical Implementation

### 1. Camera Barcode Scanner
- Use `expo-camera` with barcode scanning enabled
- Support EAN-13, EAN-8, UPC-A, UPC-E formats (standard food barcodes)
- Implement scanning UI with viewfinder overlay and haptic feedback
- Handle low-light conditions with torch toggle

### 2. Open Food Facts API Integration
- API endpoint: https://world.openfoodfacts.org/api/v0/product/{barcode}.json
- Free, open-source database with 2M+ products
- No API key required (rate limit: be respectful)
- Implement caching layer to reduce API calls
- Store successfully scanned products locally for offline access

### 3. Data Mapping
Map Open Food Facts response to Nutri's nutrition schema:
- nutriments.energy-kcal_100g → calories (per 100g)
- nutriments.proteins_100g → protein
- nutriments.carbohydrates_100g → carbs
- nutriments.fat_100g → fat
- nutriments.fiber_100g → fiber
- nutriments.sugars_100g → sugar
- nutriments.sodium_100g → sodium
- serving_size → portion reference

### 4. UI/UX Flow
1. User taps barcode icon in add-meal screen
2. Camera opens with scanning viewfinder
3. Barcode detected → API lookup
4. Results shown: product name, image, nutrition per serving
5. User can adjust serving size
6. One-tap to add to meal log

### 5. Fallback Handling
- Product not found: Offer manual entry or AI food scanner
- Network error: Check local cache first
- Invalid barcode: Show helpful error message

### 6. Files to Create/Modify
- `app/scan-barcode.tsx` - New barcode scanner screen
- `lib/api/openfoodfacts.ts` - API client
- `lib/types/barcode.ts` - Type definitions
- `app/add-meal.tsx` - Add barcode scanner button
- `server/src/routes/foods.ts` - Cache endpoint (optional)

## Success Metrics
- Scanner accuracy: >95% successful reads
- API hit rate: >80% products found
- User adoption: 50%+ of food logs use barcode

## Dependencies
- expo-camera (already installed)
- expo-haptics (for feedback)

**Test Strategy:**

1. Unit tests for Open Food Facts API client (mock responses)
2. Unit tests for nutrition data mapping
3. Integration tests for barcode detection
4. E2E test: scan sample barcode → verify nutrition displayed
5. Test offline caching behavior
6. Test fallback flows (product not found, network error)
