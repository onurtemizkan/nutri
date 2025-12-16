"""
Ingredient Extraction Service

Extracts ingredients from meal text/descriptions using:
- Exact matching against ingredient database
- Fuzzy string matching (RapidFuzz/Levenshtein)
- Hidden allergen keyword detection
- Compound inference from food categories

Based on FDA Big 9 allergens and comprehensive compound database.
"""

import re
import time
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from difflib import SequenceMatcher

# Try to import rapidfuzz, fall back to difflib
try:
    from rapidfuzz import fuzz, process

    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

from app.data.allergen_database import (
    INGREDIENT_DATABASE,
    CROSS_REACTIVITY,
    IngredientData,
    check_hidden_allergen,
    AllergenType,
    CompoundLevel,
    DerivationType,
    FodmapLevel,
)
from app.schemas.sensitivity import (
    IngredientExtractionRequest,
    IngredientExtractionResponse,
    ExtractedIngredient,
    AllergenWarning,
    CompoundWarning,
    AllergenTypeSchema,
    CompoundLevelSchema,
    DerivationTypeSchema,
    SeveritySchema,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Common words to ignore when extracting ingredients
STOP_WORDS = {
    "with",
    "and",
    "or",
    "in",
    "on",
    "the",
    "a",
    "an",
    "of",
    "for",
    "to",
    "from",
    "by",
    "at",
    "into",
    "over",
    "under",
    "above",
    "below",
    "fresh",
    "organic",
    "natural",
    "homemade",
    "cooked",
    "raw",
    "baked",
    "fried",
    "grilled",
    "roasted",
    "steamed",
    "boiled",
    "sauteed",
    "sliced",
    "diced",
    "chopped",
    "minced",
    "crushed",
    "ground",
    "large",
    "small",
    "medium",
    "extra",
    "lite",
    "light",
    "heavy",
    "cup",
    "cups",
    "tbsp",
    "tsp",
    "oz",
    "ounce",
    "ounces",
    "lb",
    "lbs",
    "gram",
    "grams",
    "kg",
    "ml",
    "liter",
    "liters",
    "serving",
    "servings",
    "piece",
    "pieces",
    "slice",
    "slices",
    "whole",
    "half",
    "quarter",
}

# Patterns for cleaning ingredient text
QUANTITY_PATTERN = re.compile(
    r"\b\d+[\d./]*\s*(g|kg|oz|lb|ml|l|cup|tbsp|tsp|serving)s?\b", re.IGNORECASE
)
NUMBER_PATTERN = re.compile(r"^\d+[\d./]*\s*")

# Allergen display names
ALLERGEN_DISPLAY_NAMES = {
    AllergenType.MILK: "Milk/Dairy",
    AllergenType.EGGS: "Eggs",
    AllergenType.FISH: "Fish",
    AllergenType.SHELLFISH_CRUSTACEAN: "Crustacean Shellfish",
    AllergenType.TREE_NUTS: "Tree Nuts",
    AllergenType.PEANUTS: "Peanuts",
    AllergenType.WHEAT: "Wheat",
    AllergenType.SOY: "Soy",
    AllergenType.SESAME: "Sesame",
    AllergenType.GLUTEN_CEREALS: "Gluten/Cereals",
    AllergenType.SHELLFISH_MOLLUSCAN: "Molluscan Shellfish",
    AllergenType.MUSTARD: "Mustard",
    AllergenType.CELERY: "Celery",
    AllergenType.LUPIN: "Lupin",
    AllergenType.SULFITES: "Sulfites",
}

# Severity mapping for allergen derivation types
DERIVATION_SEVERITY = {
    DerivationType.DIRECTLY_CONTAINS: SeveritySchema.SEVERE,
    DerivationType.DERIVED_FROM: SeveritySchema.MODERATE,
    DerivationType.MAY_CONTAIN: SeveritySchema.MILD,
    DerivationType.LIKELY_CONTAINS: SeveritySchema.MODERATE,
    DerivationType.PROCESSED_WITH: SeveritySchema.MILD,
    DerivationType.FREE_FROM: SeveritySchema.NONE,
}

# Compound level severity mapping
COMPOUND_SEVERITY = {
    CompoundLevel.VERY_HIGH: SeveritySchema.SEVERE,
    CompoundLevel.HIGH: SeveritySchema.MODERATE,
    CompoundLevel.MEDIUM: SeveritySchema.MILD,
    CompoundLevel.LOW: SeveritySchema.NONE,
    CompoundLevel.NEGLIGIBLE: SeveritySchema.NONE,
}


# =============================================================================
# FUZZY MATCHING HELPERS
# =============================================================================


def fuzzy_ratio(s1: str, s2: str) -> float:
    """
    Calculate fuzzy match ratio between two strings.
    Uses RapidFuzz if available, falls back to difflib.

    Returns:
        Similarity ratio 0.0 - 1.0
    """
    if RAPIDFUZZ_AVAILABLE:
        return fuzz.ratio(s1.lower(), s2.lower()) / 100.0
    else:
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def fuzzy_partial_ratio(s1: str, s2: str) -> float:
    """
    Calculate partial fuzzy match ratio (substring matching).

    Returns:
        Partial similarity ratio 0.0 - 1.0
    """
    if RAPIDFUZZ_AVAILABLE:
        return fuzz.partial_ratio(s1.lower(), s2.lower()) / 100.0
    else:
        # Simple substring check for fallback
        s1_lower, s2_lower = s1.lower(), s2.lower()
        if s1_lower in s2_lower or s2_lower in s1_lower:
            return 1.0
        return SequenceMatcher(None, s1_lower, s2_lower).ratio()


def fuzzy_token_sort_ratio(s1: str, s2: str) -> float:
    """
    Calculate token sort ratio (order-independent word matching).

    Returns:
        Token sort similarity ratio 0.0 - 1.0
    """
    if RAPIDFUZZ_AVAILABLE:
        return fuzz.token_sort_ratio(s1.lower(), s2.lower()) / 100.0
    else:
        # Sort words and compare
        words1 = sorted(s1.lower().split())
        words2 = sorted(s2.lower().split())
        return SequenceMatcher(None, " ".join(words1), " ".join(words2)).ratio()


def find_best_match(
    query: str, choices: List[str], threshold: float = 0.75
) -> Optional[Tuple[str, float]]:
    """
    Find the best fuzzy match for a query from a list of choices.

    Args:
        query: The search string
        choices: List of possible matches
        threshold: Minimum score threshold

    Returns:
        Tuple of (best_match, score) or None if no match above threshold
    """
    if RAPIDFUZZ_AVAILABLE:
        result = process.extractOne(
            query.lower(),
            [c.lower() for c in choices],
            scorer=fuzz.WRatio,
            score_cutoff=threshold * 100,
        )
        if result:
            # Return original case version
            idx = [c.lower() for c in choices].index(result[0])
            return (choices[idx], result[1] / 100.0)
        return None
    else:
        best_match = None
        best_score = 0.0
        query_lower = query.lower()

        for choice in choices:
            # Combine multiple scoring methods
            ratio = SequenceMatcher(None, query_lower, choice.lower()).ratio()

            if ratio > best_score and ratio >= threshold:
                best_score = ratio
                best_match = choice

        return (best_match, best_score) if best_match else None


# =============================================================================
# TEXT PARSING HELPERS
# =============================================================================


def clean_ingredient_text(text: str) -> str:
    """Clean and normalize ingredient text."""
    # Remove quantities
    text = QUANTITY_PATTERN.sub("", text)
    text = NUMBER_PATTERN.sub("", text)

    # Remove parenthetical content like "(optional)" or "(diced)"
    text = re.sub(r"\([^)]*\)", "", text)

    # Remove special characters but keep hyphens and apostrophes
    text = re.sub(r"[^\w\s'-]", " ", text)

    # Normalize whitespace
    text = " ".join(text.split())

    return text.strip().lower()


def extract_ingredient_candidates(text: str) -> List[str]:
    """
    Extract potential ingredient names from text.

    Handles various formats:
    - "Chicken breast with mushrooms"
    - "eggs, milk, flour, sugar"
    - "Grilled salmon, steamed broccoli"
    - "Caesar salad with parmesan"
    """
    candidates = []

    # Split by common separators
    parts = re.split(
        r"[,;|&]+|\band\b|\bwith\b|\bover\b|\bon\b", text, flags=re.IGNORECASE
    )

    for part in parts:
        cleaned = clean_ingredient_text(part)
        if not cleaned:
            continue

        # Filter out stop words only if the entire term is a stop word
        words = cleaned.split()
        filtered_words = [w for w in words if w not in STOP_WORDS]

        if filtered_words:
            # Keep the cleaned version
            candidate = " ".join(filtered_words)
            if len(candidate) > 1:  # Skip single characters
                candidates.append(candidate)

                # Also add individual words if multi-word
                if len(filtered_words) > 1:
                    for word in filtered_words:
                        if len(word) > 2 and word not in STOP_WORDS:
                            candidates.append(word)

    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    return unique_candidates


# =============================================================================
# MAIN SERVICE CLASS
# =============================================================================


@dataclass
class IngredientMatch:
    """Internal representation of an ingredient match."""

    original_text: str
    ingredient_key: str
    ingredient_data: Optional[IngredientData]
    confidence: float
    match_type: str  # exact, fuzzy, hidden_keyword, category_inference


class IngredientExtractionService:
    """
    Service for extracting ingredients from meal text and detecting allergens.

    Features:
    - Exact and fuzzy matching against ingredient database
    - Hidden allergen keyword detection
    - Compound level aggregation (histamine, tyramine, FODMAP)
    - Cross-reactivity warnings
    - Severity-based allergen warnings
    """

    def __init__(self):
        """Initialize the service with precomputed search structures."""
        # Build search indices
        self._ingredient_names: List[str] = []
        self._display_names: List[str] = []
        self._all_variants: Dict[str, str] = {}  # variant -> ingredient_key

        self._build_search_indices()

        logger.info(
            f"Initialized IngredientExtractionService with "
            f"{len(INGREDIENT_DATABASE)} ingredients, "
            f"{len(self._all_variants)} name variants, "
            f"RapidFuzz: {'enabled' if RAPIDFUZZ_AVAILABLE else 'disabled'}"
        )

    def _build_search_indices(self):
        """Build indices for fast ingredient lookup."""
        for key, data in INGREDIENT_DATABASE.items():
            self._ingredient_names.append(key)
            self._display_names.append(data.display_name.lower())

            # Map exact name
            self._all_variants[key.lower()] = key
            self._all_variants[data.display_name.lower()] = key

            # Map all name variants
            if data.name_variants:
                for variant in data.name_variants:
                    self._all_variants[variant.lower()] = key

    async def extract_ingredients(
        self, request: IngredientExtractionRequest
    ) -> IngredientExtractionResponse:
        """
        Extract ingredients and detect allergens from text.

        Args:
            request: Extraction request with text and options

        Returns:
            Response with ingredients, allergen warnings, and compound warnings
        """
        start_time = time.time()

        try:
            # 1. Extract candidate ingredients from text
            candidates = extract_ingredient_candidates(request.text)
            logger.debug(f"Extracted {len(candidates)} candidates from text")

            # 2. Match candidates against database
            matches = self._match_ingredients(
                candidates, request.fuzzy_threshold, request.max_results
            )

            # 3. Check for hidden allergens in original text
            hidden_allergens = []
            if request.include_hidden_allergens:
                hidden_allergens = check_hidden_allergen(request.text)

            # 4. Build response components
            extracted_ingredients = self._build_extracted_ingredients(matches)
            allergen_warnings = self._build_allergen_warnings(matches, hidden_allergens)
            compound_warnings = self._build_compound_warnings(matches)

            # 5. Calculate compound totals
            totals = self._calculate_compound_totals(matches)

            # 6. Generate suggestions
            suggestions = self._generate_suggestions(
                matches, allergen_warnings, compound_warnings
            )

            processing_time = (time.time() - start_time) * 1000

            return IngredientExtractionResponse(
                success=True,
                ingredients=extracted_ingredients[: request.max_results],
                allergen_warnings=allergen_warnings,
                compound_warnings=compound_warnings,
                total_histamine_mg=totals.get("histamine"),
                total_tyramine_mg=totals.get("tyramine"),
                fodmap_summary=totals.get("fodmap"),
                processing_time_ms=round(processing_time, 2),
                suggestions=suggestions,
            )

        except Exception as e:
            logger.error(f"Ingredient extraction error: {e}", exc_info=True)
            raise

    def _match_ingredients(
        self, candidates: List[str], threshold: float, max_results: int
    ) -> List[IngredientMatch]:
        """
        Match candidate strings to ingredients in database.

        Uses a cascading approach:
        1. Exact match on name/variants
        2. Fuzzy match on names
        3. Hidden keyword detection
        """
        matches: List[IngredientMatch] = []
        matched_keys: Set[str] = set()

        for candidate in candidates:
            candidate_lower = candidate.lower()

            # Try exact match first
            if candidate_lower in self._all_variants:
                key = self._all_variants[candidate_lower]
                if key not in matched_keys:
                    matches.append(
                        IngredientMatch(
                            original_text=candidate,
                            ingredient_key=key,
                            ingredient_data=INGREDIENT_DATABASE.get(key),
                            confidence=1.0,
                            match_type="exact",
                        )
                    )
                    matched_keys.add(key)
                continue

            # Try fuzzy match
            all_searchable = list(self._all_variants.keys())
            best_match = find_best_match(candidate, all_searchable, threshold)

            if best_match:
                matched_text, score = best_match
                key = self._all_variants[matched_text.lower()]
                if key not in matched_keys:
                    matches.append(
                        IngredientMatch(
                            original_text=candidate,
                            ingredient_key=key,
                            ingredient_data=INGREDIENT_DATABASE.get(key),
                            confidence=score,
                            match_type="fuzzy",
                        )
                    )
                    matched_keys.add(key)
                continue

            # Check for hidden allergen keywords in this candidate
            hidden = check_hidden_allergen(candidate)
            if hidden:
                # Create a synthetic match for the hidden allergen
                for allergen in hidden:
                    matches.append(
                        IngredientMatch(
                            original_text=candidate,
                            ingredient_key=f"hidden_{allergen.value}",
                            ingredient_data=None,
                            confidence=0.9,  # High confidence for keyword match
                            match_type="hidden_keyword",
                        )
                    )

        # Sort by confidence
        matches.sort(key=lambda m: m.confidence, reverse=True)

        return matches[: max_results * 2]  # Return more for filtering

    def _build_extracted_ingredients(
        self, matches: List[IngredientMatch]
    ) -> List[ExtractedIngredient]:
        """Build ExtractedIngredient objects from matches."""
        ingredients = []
        seen_keys = set()

        for match in matches:
            if match.match_type == "hidden_keyword":
                continue  # Skip hidden allergens (they go in warnings)

            if match.ingredient_key in seen_keys:
                continue
            seen_keys.add(match.ingredient_key)

            data = match.ingredient_data

            ingredients.append(
                ExtractedIngredient(
                    matched_text=match.original_text,
                    ingredient_id=match.ingredient_key if data else None,
                    ingredient_name=match.ingredient_key,
                    display_name=(
                        data.display_name if data else match.original_text.title()
                    ),
                    confidence=round(match.confidence, 3),
                    match_type=match.match_type,
                    category=data.category.value if data else None,
                )
            )

        return ingredients

    def _build_allergen_warnings(
        self, matches: List[IngredientMatch], hidden_allergens: List[AllergenType]
    ) -> List[AllergenWarning]:
        """Build allergen warnings from matches and hidden allergens."""
        warnings: List[AllergenWarning] = []
        seen_allergens: Dict[
            AllergenType, float
        ] = {}  # Track highest confidence per allergen

        # Process ingredient matches
        for match in matches:
            if not match.ingredient_data:
                continue

            for allergen_mapping in match.ingredient_data.allergens:
                allergen = allergen_mapping.allergen_type
                confidence = match.confidence * allergen_mapping.confidence

                # Keep highest confidence warning for each allergen
                if allergen in seen_allergens:
                    if confidence <= seen_allergens[allergen]:
                        continue

                seen_allergens[allergen] = confidence

                warnings.append(
                    AllergenWarning(
                        allergen_type=AllergenTypeSchema(allergen.value),
                        display_name=ALLERGEN_DISPLAY_NAMES.get(
                            allergen, allergen.value
                        ),
                        derivation=DerivationTypeSchema(
                            allergen_mapping.derivation.value
                        ),
                        confidence=round(confidence, 3),
                        source_ingredient=match.ingredient_data.display_name,
                        is_hidden=False,
                        warning_level=DERIVATION_SEVERITY.get(
                            allergen_mapping.derivation, SeveritySchema.MODERATE
                        ),
                    )
                )

        # Add hidden allergen warnings
        for allergen in hidden_allergens:
            if allergen not in seen_allergens or seen_allergens[allergen] < 0.9:
                warnings.append(
                    AllergenWarning(
                        allergen_type=AllergenTypeSchema(allergen.value),
                        display_name=ALLERGEN_DISPLAY_NAMES.get(
                            allergen, allergen.value
                        ),
                        derivation=DerivationTypeSchema.LIKELY_CONTAINS,
                        confidence=0.9,
                        source_ingredient="Hidden in ingredient name",
                        is_hidden=True,
                        warning_level=SeveritySchema.MODERATE,
                    )
                )

        # Sort by warning level and confidence
        severity_order = {
            SeveritySchema.LIFE_THREATENING: 0,
            SeveritySchema.EMERGENCY: 1,
            SeveritySchema.SEVERE: 2,
            SeveritySchema.MODERATE: 3,
            SeveritySchema.MILD: 4,
            SeveritySchema.NONE: 5,
        }
        warnings.sort(
            key=lambda w: (severity_order.get(w.warning_level, 5), -w.confidence)
        )

        # De-duplicate keeping highest severity/confidence
        seen = set()
        unique_warnings = []
        for w in warnings:
            if w.allergen_type not in seen:
                seen.add(w.allergen_type)
                unique_warnings.append(w)

        return unique_warnings

    def _build_compound_warnings(
        self, matches: List[IngredientMatch]
    ) -> List[CompoundWarning]:
        """Build compound warnings (histamine, tyramine, FODMAP, etc.)."""
        warnings: List[CompoundWarning] = []

        for match in matches:
            if not match.ingredient_data:
                continue

            data = match.ingredient_data

            # Histamine warning
            if data.histamine_level and data.histamine_level in (
                CompoundLevel.HIGH,
                CompoundLevel.VERY_HIGH,
            ):
                warnings.append(
                    CompoundWarning(
                        compound_type="histamine",
                        level=CompoundLevelSchema(data.histamine_level.value),
                        amount_mg=data.histamine_mg,
                        source_ingredient=data.display_name,
                        warning_message=self._histamine_warning_message(data),
                    )
                )

            # Tyramine warning
            if data.tyramine_level and data.tyramine_level in (
                CompoundLevel.HIGH,
                CompoundLevel.VERY_HIGH,
            ):
                warnings.append(
                    CompoundWarning(
                        compound_type="tyramine",
                        level=CompoundLevelSchema(data.tyramine_level.value),
                        amount_mg=data.tyramine_mg,
                        source_ingredient=data.display_name,
                        warning_message=self._tyramine_warning_message(data),
                    )
                )

            # FODMAP warning
            if data.fodmap_level and data.fodmap_level == FodmapLevel.HIGH:
                fodmap_types = (
                    ", ".join(data.fodmap_types) if data.fodmap_types else "various"
                )
                warnings.append(
                    CompoundWarning(
                        compound_type="fodmap",
                        level=CompoundLevelSchema.HIGH,
                        amount_mg=None,
                        source_ingredient=data.display_name,
                        warning_message=f"High FODMAP ({fodmap_types}). May cause IBS symptoms.",
                    )
                )

            # Nightshade warning
            if data.is_nightshade:
                warnings.append(
                    CompoundWarning(
                        compound_type="nightshade",
                        level=CompoundLevelSchema.HIGH,
                        amount_mg=None,
                        source_ingredient=data.display_name,
                        warning_message="Nightshade vegetable. May trigger inflammation in sensitive individuals.",
                    )
                )

        return warnings

    def _histamine_warning_message(self, data: IngredientData) -> str:
        """Generate histamine warning message."""
        if data.histamine_mg and data.histamine_mg > 100:
            return f"Very high histamine ({data.histamine_mg}mg/100g). May trigger histamine intolerance symptoms."
        elif data.histamine_mg:
            return f"High histamine ({data.histamine_mg}mg/100g). Monitor for symptoms if histamine sensitive."
        return "High histamine content. May trigger symptoms in histamine-sensitive individuals."

    def _tyramine_warning_message(self, data: IngredientData) -> str:
        """Generate tyramine warning message."""
        if data.tyramine_mg and data.tyramine_mg > 40:
            return f"Very high tyramine ({data.tyramine_mg}mg/100g). Avoid if taking MAOIs."
        elif data.tyramine_mg:
            return f"High tyramine ({data.tyramine_mg}mg/100g). May cause migraines in sensitive individuals."
        return "High tyramine content. May trigger migraines or interact with MAOIs."

    def _calculate_compound_totals(
        self, matches: List[IngredientMatch]
    ) -> Dict[str, any]:
        """Calculate total compound amounts from matched ingredients."""
        totals = {
            "histamine": 0.0,
            "tyramine": 0.0,
            "fodmap": {},
        }

        for match in matches:
            if not match.ingredient_data:
                continue

            data = match.ingredient_data

            # Assume 100g per ingredient for estimation
            if data.histamine_mg:
                totals["histamine"] += data.histamine_mg

            if data.tyramine_mg:
                totals["tyramine"] += data.tyramine_mg

            if data.fodmap_level and data.fodmap_types:
                for fodmap_type in data.fodmap_types:
                    current = totals["fodmap"].get(fodmap_type, "low")
                    if data.fodmap_level == FodmapLevel.HIGH:
                        totals["fodmap"][fodmap_type] = "high"
                    elif data.fodmap_level == FodmapLevel.MEDIUM and current != "high":
                        totals["fodmap"][fodmap_type] = "medium"

        # Round values
        totals["histamine"] = (
            round(totals["histamine"], 1) if totals["histamine"] > 0 else None
        )
        totals["tyramine"] = (
            round(totals["tyramine"], 1) if totals["tyramine"] > 0 else None
        )
        totals["fodmap"] = totals["fodmap"] if totals["fodmap"] else None

        return totals

    def _generate_suggestions(
        self,
        matches: List[IngredientMatch],
        allergen_warnings: List[AllergenWarning],
        compound_warnings: List[CompoundWarning],
    ) -> List[str]:
        """Generate helpful suggestions based on analysis."""
        suggestions = []

        # Suggestion for severe allergens
        severe_allergens = [
            w for w in allergen_warnings if w.warning_level == SeveritySchema.SEVERE
        ]
        if severe_allergens:
            allergen_names = ", ".join([w.display_name for w in severe_allergens[:3]])
            suggestions.append(
                f"Contains {allergen_names}. Verify ingredients if you have allergies."
            )

        # Suggestion for hidden allergens
        hidden = [w for w in allergen_warnings if w.is_hidden]
        if hidden:
            suggestions.append(
                "Hidden allergens detected in ingredient names. Check labels carefully."
            )

        # Suggestion for high histamine
        high_histamine = [
            w for w in compound_warnings if w.compound_type == "histamine"
        ]
        if high_histamine:
            suggestions.append(
                "High histamine foods detected. Space meals apart if histamine-sensitive."
            )

        # Suggestion for high tyramine
        high_tyramine = [w for w in compound_warnings if w.compound_type == "tyramine"]
        if high_tyramine:
            suggestions.append(
                "High tyramine foods detected. Avoid if taking MAOIs or prone to migraines."
            )

        # Low match confidence suggestion
        low_confidence = [
            m for m in matches if m.confidence < 0.85 and m.match_type == "fuzzy"
        ]
        if low_confidence:
            suggestions.append(
                "Some ingredients matched with lower confidence. Verify correctness."
            )

        return suggestions

    def get_all_allergen_types(self) -> List[Dict[str, str]]:
        """Get list of all supported allergen types."""
        return [
            {
                "value": allergen.value,
                "display_name": ALLERGEN_DISPLAY_NAMES.get(allergen, allergen.value),
            }
            for allergen in AllergenType
        ]

    def search_ingredients(self, query: str, limit: int = 20) -> List[Dict[str, any]]:
        """
        Search ingredients database by name.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching ingredients
        """
        results = []
        query_lower = query.lower()

        for key, data in INGREDIENT_DATABASE.items():
            score = 0.0

            # Exact match gets highest score
            if query_lower == key or query_lower == data.display_name.lower():
                score = 1.0
            # Check name variants
            elif data.name_variants and any(
                query_lower == v.lower() for v in data.name_variants
            ):
                score = 0.95
            # Partial match
            elif query_lower in key or query_lower in data.display_name.lower():
                score = 0.8
            # Fuzzy match
            else:
                score = fuzzy_ratio(query, data.display_name)

            if score >= 0.5:
                results.append(
                    {
                        "id": key,
                        "name": data.display_name,
                        "category": data.category.value,
                        "score": round(score, 3),
                        "allergens": [a.allergen_type.value for a in data.allergens],
                        "histamine_level": (
                            data.histamine_level.value if data.histamine_level else None
                        ),
                        "fodmap_level": (
                            data.fodmap_level.value if data.fodmap_level else None
                        ),
                    }
                )

        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:limit]

    def get_cross_reactivity(self, allergen_type: AllergenType) -> List[Dict[str, str]]:
        """Get cross-reactive allergens for a given allergen."""
        cross_reactive = CROSS_REACTIVITY.get(allergen_type, [])
        return [
            {
                "allergen": a.value,
                "display_name": ALLERGEN_DISPLAY_NAMES.get(a, a.value),
            }
            for a in cross_reactive
        ]


# Singleton instance
ingredient_extraction_service = IngredientExtractionService()
