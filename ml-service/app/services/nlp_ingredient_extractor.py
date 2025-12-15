"""
Advanced NLP-Based Ingredient Extraction Service.

This module provides sophisticated natural language processing for extracting
ingredients from unstructured text using:
- Custom spaCy NER model for food entities
- Dependency parsing for quantity and unit extraction
- Semantic similarity using word embeddings
- Rule-based patterns for cooking terminology
- Contextual understanding of preparation methods

Architecture:
1. Text preprocessing and normalization
2. Entity recognition (ingredients, quantities, units)
3. Dependency parsing for relationships
4. Post-processing and validation
5. Confidence scoring and ranking
"""

import re
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
from collections import defaultdict
import logging

# Graceful spaCy import
try:
    import spacy
    from spacy.tokens import Doc, Span, Token
    from spacy.language import Language
    from spacy.matcher import Matcher, PhraseMatcher
    from spacy.vocab import Vocab
    SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    Doc = Span = Token = Language = Matcher = PhraseMatcher = Vocab = None
    SPACY_AVAILABLE = False

# Graceful numpy import for embeddings
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of entities we extract from text."""
    INGREDIENT = "INGREDIENT"
    QUANTITY = "QUANTITY"
    UNIT = "UNIT"
    PREPARATION = "PREPARATION"
    COOKING_METHOD = "COOKING_METHOD"
    ALLERGEN = "ALLERGEN"
    BRAND = "BRAND"
    MODIFIER = "MODIFIER"


class MeasurementSystem(Enum):
    """Measurement systems."""
    METRIC = "metric"
    IMPERIAL = "imperial"
    VOLUMETRIC = "volumetric"
    COUNT = "count"
    DESCRIPTIVE = "descriptive"


@dataclass
class QuantityInfo:
    """Extracted quantity information."""
    value: float
    unit: str
    original_text: str
    system: MeasurementSystem
    normalized_grams: Optional[float] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit,
            "original_text": self.original_text,
            "system": self.system.value,
            "normalized_grams": self.normalized_grams,
            "confidence": self.confidence,
        }


@dataclass
class ExtractedIngredient:
    """A fully extracted ingredient with all associated information."""
    name: str
    original_text: str
    quantity: Optional[QuantityInfo] = None
    preparation: Optional[str] = None
    cooking_method: Optional[str] = None
    modifiers: List[str] = field(default_factory=list)
    allergens: List[str] = field(default_factory=list)
    confidence: float = 1.0
    start_char: int = 0
    end_char: int = 0
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "original_text": self.original_text,
            "confidence": self.confidence,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "modifiers": self.modifiers,
            "allergens": self.allergens,
        }
        if self.quantity:
            result["quantity"] = self.quantity.to_dict()
        if self.preparation:
            result["preparation"] = self.preparation
        if self.cooking_method:
            result["cooking_method"] = self.cooking_method
        return result


@dataclass
class ExtractionResult:
    """Complete extraction result from text."""
    text: str
    ingredients: List[ExtractedIngredient]
    entities: Dict[str, List[Tuple[str, int, int]]]
    parse_tree: Optional[Dict[str, Any]] = None
    confidence: float = 1.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "ingredients": [i.to_dict() for i in self.ingredients],
            "entity_count": {k: len(v) for k, v in self.entities.items()},
            "confidence": self.confidence,
            "warnings": self.warnings,
        }


class UnitConverter:
    """
    Convert between measurement units.

    Handles metric, imperial, and volumetric conversions
    with ingredient-specific density adjustments.
    """

    # Standard unit conversions to grams
    UNIT_TO_GRAMS = {
        # Mass - Metric
        "g": 1.0,
        "gram": 1.0,
        "grams": 1.0,
        "kg": 1000.0,
        "kilogram": 1000.0,
        "kilograms": 1000.0,
        "mg": 0.001,
        "milligram": 0.001,
        "milligrams": 0.001,
        # Mass - Imperial
        "oz": 28.3495,
        "ounce": 28.3495,
        "ounces": 28.3495,
        "lb": 453.592,
        "lbs": 453.592,
        "pound": 453.592,
        "pounds": 453.592,
        # Volume - Metric (using water density ~1g/ml)
        "ml": 1.0,
        "milliliter": 1.0,
        "milliliters": 1.0,
        "l": 1000.0,
        "liter": 1000.0,
        "liters": 1000.0,
        "litre": 1000.0,
        "litres": 1000.0,
        # Volume - US
        "cup": 236.588,
        "cups": 236.588,
        "tbsp": 14.787,
        "tablespoon": 14.787,
        "tablespoons": 14.787,
        "tsp": 4.929,
        "teaspoon": 4.929,
        "teaspoons": 4.929,
        "fl oz": 29.5735,
        "fluid ounce": 29.5735,
        "fluid ounces": 29.5735,
        "pint": 473.176,
        "pints": 473.176,
        "quart": 946.353,
        "quarts": 946.353,
        "gallon": 3785.41,
        "gallons": 3785.41,
    }

    # Density adjustments for common ingredients (g/ml)
    INGREDIENT_DENSITY = {
        "flour": 0.593,
        "sugar": 0.845,
        "brown sugar": 0.82,
        "powdered sugar": 0.56,
        "butter": 0.911,
        "oil": 0.92,
        "olive oil": 0.918,
        "vegetable oil": 0.92,
        "honey": 1.42,
        "maple syrup": 1.37,
        "milk": 1.03,
        "cream": 1.008,
        "heavy cream": 0.994,
        "sour cream": 1.06,
        "yogurt": 1.03,
        "water": 1.0,
        "salt": 1.217,
        "rice": 0.85,
        "oats": 0.41,
        "cocoa powder": 0.52,
        "baking powder": 0.9,
        "baking soda": 1.0,
        "cornstarch": 0.58,
        "peanut butter": 1.09,
        "nuts": 0.65,
        "cheese": 1.0,
    }

    @classmethod
    def get_system(cls, unit: str) -> MeasurementSystem:
        """Determine measurement system from unit."""
        unit_lower = unit.lower().strip()

        if unit_lower in ["g", "gram", "grams", "kg", "kilogram", "kilograms",
                          "mg", "milligram", "milligrams", "ml", "milliliter",
                          "milliliters", "l", "liter", "liters", "litre", "litres"]:
            return MeasurementSystem.METRIC

        if unit_lower in ["oz", "ounce", "ounces", "lb", "lbs", "pound", "pounds",
                          "fl oz", "fluid ounce", "fluid ounces"]:
            return MeasurementSystem.IMPERIAL

        if unit_lower in ["cup", "cups", "tbsp", "tablespoon", "tablespoons",
                          "tsp", "teaspoon", "teaspoons", "pint", "pints",
                          "quart", "quarts", "gallon", "gallons"]:
            return MeasurementSystem.VOLUMETRIC

        if unit_lower in ["", "piece", "pieces", "whole", "slice", "slices",
                          "clove", "cloves", "bunch", "bunches", "head", "heads",
                          "sprig", "sprigs", "stick", "sticks", "can", "cans",
                          "package", "packages", "box", "boxes"]:
            return MeasurementSystem.COUNT

        return MeasurementSystem.DESCRIPTIVE

    @classmethod
    def to_grams(
        cls,
        value: float,
        unit: str,
        ingredient: Optional[str] = None
    ) -> Optional[float]:
        """Convert quantity to grams."""
        unit_lower = unit.lower().strip()

        if unit_lower not in cls.UNIT_TO_GRAMS:
            return None

        base_grams = value * cls.UNIT_TO_GRAMS[unit_lower]

        # Apply ingredient-specific density for volumetric measures
        if cls.get_system(unit) == MeasurementSystem.VOLUMETRIC and ingredient:
            ingredient_lower = ingredient.lower()
            for ing_name, density in cls.INGREDIENT_DENSITY.items():
                if ing_name in ingredient_lower:
                    base_grams *= density
                    break

        return round(base_grams, 2)


class PatternLibrary:
    """
    Pattern library for ingredient extraction.

    Contains regex patterns and rules for:
    - Quantity patterns (fractions, ranges, etc.)
    - Unit patterns
    - Preparation terms
    - Cooking methods
    - Allergen indicators
    """

    # Fraction patterns
    FRACTION_MAP = {
        "½": 0.5, "⅓": 0.333, "⅔": 0.667, "¼": 0.25, "¾": 0.75,
        "⅕": 0.2, "⅖": 0.4, "⅗": 0.6, "⅘": 0.8, "⅙": 0.167,
        "⅚": 0.833, "⅛": 0.125, "⅜": 0.375, "⅝": 0.625, "⅞": 0.875,
        "1/2": 0.5, "1/3": 0.333, "2/3": 0.667, "1/4": 0.25, "3/4": 0.75,
        "1/8": 0.125, "3/8": 0.375, "5/8": 0.625, "7/8": 0.875,
    }

    # Quantity regex patterns
    QUANTITY_PATTERNS = [
        # Unicode fractions: ½, ¼, etc.
        r"(\d+\s*)?([½⅓⅔¼¾⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞])",
        # Written fractions: 1/2, 3/4, etc.
        r"(\d+\s*)?(\d+/\d+)",
        # Decimal: 1.5, 0.25, etc.
        r"(\d+\.?\d*)",
        # Range: 2-3, 1 to 2, etc.
        r"(\d+\.?\d*)\s*[-–to]+\s*(\d+\.?\d*)",
        # Written numbers
        r"(one|two|three|four|five|six|seven|eight|nine|ten|dozen|half|quarter)",
    ]

    WRITTEN_NUMBERS = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "dozen": 12,
        "half": 0.5, "quarter": 0.25,
    }

    # Units organized by category
    UNITS = {
        "mass_metric": ["g", "gram", "grams", "kg", "kilogram", "kilograms", "mg"],
        "mass_imperial": ["oz", "ounce", "ounces", "lb", "lbs", "pound", "pounds"],
        "volume_metric": ["ml", "milliliter", "milliliters", "l", "liter", "liters", "litre", "litres"],
        "volume_imperial": ["cup", "cups", "tbsp", "tablespoon", "tablespoons",
                           "tsp", "teaspoon", "teaspoons", "fl oz", "fluid ounce",
                           "pint", "pints", "quart", "quarts", "gallon", "gallons"],
        "count": ["piece", "pieces", "whole", "slice", "slices", "clove", "cloves",
                  "bunch", "bunches", "head", "heads", "sprig", "sprigs", "stick",
                  "sticks", "can", "cans", "package", "packages", "jar", "jars",
                  "bag", "bags", "box", "boxes", "bottle", "bottles"],
        "descriptive": ["pinch", "dash", "handful", "splash", "drizzle",
                       "small", "medium", "large", "extra large"],
    }

    # Preparation terms
    PREPARATIONS = [
        "chopped", "diced", "minced", "sliced", "grated", "shredded",
        "crushed", "ground", "mashed", "pureed", "julienned", "cubed",
        "quartered", "halved", "torn", "crumbled", "zested", "peeled",
        "deveined", "deboned", "trimmed", "cored", "seeded", "pitted",
        "blanched", "soaked", "drained", "rinsed", "dried", "toasted",
        "roasted", "sauteed", "caramelized", "melted", "softened",
        "chilled", "frozen", "thawed", "room temperature", "at room temp",
        "freshly", "fresh", "finely", "coarsely", "roughly", "thinly",
    ]

    # Cooking methods
    COOKING_METHODS = [
        "bake", "baked", "baking", "roast", "roasted", "roasting",
        "grill", "grilled", "grilling", "broil", "broiled", "broiling",
        "fry", "fried", "frying", "deep fry", "deep fried", "deep frying",
        "saute", "sauteed", "sauteing", "stir fry", "stir fried",
        "boil", "boiled", "boiling", "simmer", "simmered", "simmering",
        "steam", "steamed", "steaming", "poach", "poached", "poaching",
        "braise", "braised", "braising", "stew", "stewed", "stewing",
        "smoke", "smoked", "smoking", "cure", "cured", "curing",
        "ferment", "fermented", "fermenting", "pickle", "pickled", "pickling",
    ]

    # Modifiers (descriptive adjectives)
    MODIFIERS = [
        "organic", "natural", "fresh", "frozen", "canned", "dried",
        "raw", "cooked", "uncooked", "ripe", "unripe", "green",
        "red", "yellow", "white", "dark", "light", "extra",
        "virgin", "extra virgin", "pure", "refined", "unrefined",
        "salted", "unsalted", "sweetened", "unsweetened", "flavored",
        "plain", "vanilla", "chocolate", "whole", "skim", "low-fat",
        "fat-free", "reduced", "lite", "light", "sugar-free", "gluten-free",
        "vegan", "vegetarian", "kosher", "halal", "non-gmo",
    ]

    @classmethod
    def parse_quantity(cls, text: str) -> Optional[Tuple[float, str]]:
        """
        Parse quantity string into numeric value and remaining text.

        Returns:
            Tuple of (numeric_value, remaining_text) or None
        """
        text = text.strip()

        # Try unicode fractions first
        for frac, value in cls.FRACTION_MAP.items():
            if frac in text:
                match = re.match(r"(\d+)?\s*" + re.escape(frac), text)
                if match:
                    whole = float(match.group(1) or 0)
                    remaining = text[match.end():].strip()
                    return (whole + value, remaining)

        # Try written numbers
        for word, value in cls.WRITTEN_NUMBERS.items():
            if text.lower().startswith(word):
                remaining = text[len(word):].strip()
                return (value, remaining)

        # Try numeric patterns
        # Range pattern: 2-3, 1 to 2
        range_match = re.match(r"(\d+\.?\d*)\s*[-–to]+\s*(\d+\.?\d*)", text)
        if range_match:
            low, high = float(range_match.group(1)), float(range_match.group(2))
            remaining = text[range_match.end():].strip()
            return ((low + high) / 2, remaining)  # Use midpoint

        # Simple decimal
        decimal_match = re.match(r"(\d+\.?\d*)", text)
        if decimal_match:
            value = float(decimal_match.group(1))
            remaining = text[decimal_match.end():].strip()
            return (value, remaining)

        return None

    @classmethod
    def extract_unit(cls, text: str) -> Optional[Tuple[str, str]]:
        """
        Extract unit from text.

        Returns:
            Tuple of (unit, remaining_text) or None
        """
        text = text.strip().lower()

        # Check all unit categories
        all_units = []
        for units in cls.UNITS.values():
            all_units.extend(units)

        # Sort by length (longest first) for greedy matching
        all_units.sort(key=len, reverse=True)

        for unit in all_units:
            if text.startswith(unit):
                # Check word boundary
                remaining_start = len(unit)
                if remaining_start >= len(text) or not text[remaining_start].isalpha():
                    return (unit, text[remaining_start:].strip())

        return None

    @classmethod
    def extract_preparation(cls, text: str) -> List[str]:
        """Extract preparation terms from text."""
        text_lower = text.lower()
        found = []

        for prep in cls.PREPARATIONS:
            if prep in text_lower:
                found.append(prep)

        return found

    @classmethod
    def extract_cooking_method(cls, text: str) -> List[str]:
        """Extract cooking method terms from text."""
        text_lower = text.lower()
        found = []

        for method in cls.COOKING_METHODS:
            if method in text_lower:
                found.append(method)

        return found

    @classmethod
    def extract_modifiers(cls, text: str) -> List[str]:
        """Extract modifier terms from text."""
        text_lower = text.lower()
        found = []

        for modifier in cls.MODIFIERS:
            if modifier in text_lower:
                found.append(modifier)

        return found


class SpaCyNLPProcessor:
    """
    spaCy-based NLP processor for advanced entity extraction.

    Uses:
    - Pre-trained NER for base entity recognition
    - Custom entity ruler for food-specific patterns
    - Dependency parsing for relationship extraction
    - Word vectors for semantic similarity
    """

    def __init__(self, model_name: str = "en_core_web_md"):
        """
        Initialize the NLP processor.

        Args:
            model_name: spaCy model to load (md or lg recommended for vectors)
        """
        self.model_name = model_name
        self.nlp: Optional[Language] = None
        self.matcher: Optional[Matcher] = None
        self.phrase_matcher: Optional[PhraseMatcher] = None
        self._initialized = False

        # Common food terms for phrase matching
        self._food_terms: Set[str] = set()
        self._allergen_terms: Set[str] = set()

    def initialize(self) -> bool:
        """Initialize spaCy model and matchers."""
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not available, using fallback extraction")
            return False

        try:
            # Try to load the specified model
            try:
                self.nlp = spacy.load(self.model_name)
            except OSError:
                # Fall back to smaller model
                logger.warning(f"Model {self.model_name} not found, trying en_core_web_sm")
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    logger.warning("No spaCy model found, using blank model")
                    self.nlp = spacy.blank("en")

            # Initialize matchers
            self.matcher = Matcher(self.nlp.vocab)
            self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")

            # Add custom patterns
            self._add_quantity_patterns()
            self._add_food_patterns()

            self._initialized = True
            logger.info(f"Initialized spaCy NLP processor with model: {self.nlp.meta.get('name', 'blank')}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize spaCy: {e}")
            return False

    def _add_quantity_patterns(self) -> None:
        """Add patterns for quantity extraction."""
        if not self.matcher:
            return

        # Pattern: NUMBER + UNIT
        self.matcher.add("QUANTITY", [
            [{"LIKE_NUM": True}, {"LOWER": {"IN": list(UnitConverter.UNIT_TO_GRAMS.keys())}}],
        ])

        # Pattern: Fraction
        self.matcher.add("FRACTION", [
            [{"TEXT": {"REGEX": r"^\d+/\d+$"}}],
            [{"LIKE_NUM": True}, {"TEXT": {"REGEX": r"^\d+/\d+$"}}],
        ])

    def _add_food_patterns(self) -> None:
        """Add patterns for food entity extraction."""
        if not self.phrase_matcher:
            return

        # Add allergen terms
        allergens = [
            "milk", "dairy", "lactose", "casein", "whey",
            "egg", "eggs", "egg white", "egg yolk",
            "peanut", "peanuts", "peanut butter",
            "tree nut", "tree nuts", "almond", "almonds", "walnut", "walnuts",
            "cashew", "cashews", "pecan", "pecans", "pistachio", "pistachios",
            "wheat", "gluten", "flour", "bread", "pasta",
            "soy", "soybean", "tofu", "edamame", "miso",
            "fish", "salmon", "tuna", "cod", "tilapia",
            "shellfish", "shrimp", "crab", "lobster", "oyster", "clam", "mussel",
            "sesame", "sesame seed", "tahini",
        ]

        self._allergen_terms = set(allergens)
        allergen_patterns = [self.nlp(text) for text in allergens]
        self.phrase_matcher.add("ALLERGEN", allergen_patterns)

    def set_food_vocabulary(self, food_terms: List[str]) -> None:
        """Set custom food vocabulary for phrase matching."""
        if not self.phrase_matcher or not self.nlp:
            return

        self._food_terms = set(food_terms)
        food_patterns = [self.nlp(term) for term in food_terms[:10000]]  # Limit size
        self.phrase_matcher.add("FOOD", food_patterns)

    def process(self, text: str) -> Optional[Doc]:
        """Process text through spaCy pipeline."""
        if not self._initialized or not self.nlp:
            return None

        return self.nlp(text)

    def extract_entities(self, doc: Doc) -> Dict[str, List[Tuple[str, int, int]]]:
        """Extract all entities from processed document."""
        entities: Dict[str, List[Tuple[str, int, int]]] = defaultdict(list)

        # Standard NER entities
        for ent in doc.ents:
            if ent.label_ in ["PRODUCT", "ORG"]:  # Potential food/brand names
                entities["POTENTIAL_FOOD"].append((ent.text, ent.start_char, ent.end_char))
            elif ent.label_ == "QUANTITY":
                entities["QUANTITY"].append((ent.text, ent.start_char, ent.end_char))
            elif ent.label_ == "CARDINAL":
                entities["NUMBER"].append((ent.text, ent.start_char, ent.end_char))

        # Custom matcher entities
        if self.matcher:
            matches = self.matcher(doc)
            for match_id, start, end in matches:
                span = doc[start:end]
                label = self.nlp.vocab.strings[match_id]
                entities[label].append((span.text, span.start_char, span.end_char))

        # Phrase matcher entities
        if self.phrase_matcher:
            phrase_matches = self.phrase_matcher(doc)
            for match_id, start, end in phrase_matches:
                span = doc[start:end]
                label = self.nlp.vocab.strings[match_id]
                entities[label].append((span.text, span.start_char, span.end_char))

        return dict(entities)

    def get_noun_chunks(self, doc: Doc) -> List[Tuple[str, str, int, int]]:
        """Extract noun chunks with their root."""
        chunks = []
        for chunk in doc.noun_chunks:
            chunks.append((
                chunk.text,
                chunk.root.text,
                chunk.start_char,
                chunk.end_char
            ))
        return chunks

    def get_dependencies(self, doc: Doc) -> List[Dict[str, Any]]:
        """Extract dependency relationships."""
        deps = []
        for token in doc:
            deps.append({
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "dep": token.dep_,
                "head": token.head.text,
                "children": [child.text for child in token.children],
            })
        return deps

    def get_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        if not self._initialized or not self.nlp:
            return 0.0

        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)

        # Check if vectors are available
        if not doc1.has_vector or not doc2.has_vector:
            return 0.0

        return doc1.similarity(doc2)


class NLPIngredientExtractor:
    """
    Main NLP-based ingredient extraction service.

    Combines:
    - spaCy NLP processing
    - Pattern-based extraction
    - Post-processing and validation
    - Confidence scoring
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_md",
        confidence_threshold: float = 0.3,
    ):
        """
        Initialize the extractor.

        Args:
            spacy_model: spaCy model to use
            confidence_threshold: Minimum confidence for extraction
        """
        self.confidence_threshold = confidence_threshold
        self.nlp_processor = SpaCyNLPProcessor(spacy_model)
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize the extractor."""
        self._initialized = self.nlp_processor.initialize()
        return self._initialized

    def set_food_vocabulary(self, food_terms: List[str]) -> None:
        """Set custom food vocabulary."""
        self.nlp_processor.set_food_vocabulary(food_terms)

    def extract(self, text: str) -> ExtractionResult:
        """
        Extract ingredients from text.

        Args:
            text: Input text (recipe, ingredient list, etc.)

        Returns:
            ExtractionResult with all extracted information
        """
        # Preprocess text
        processed_text = self._preprocess(text)

        # Initialize result
        result = ExtractionResult(
            text=text,
            ingredients=[],
            entities={},
            warnings=[],
        )

        # Try spaCy extraction first
        if self._initialized:
            result = self._extract_with_spacy(processed_text, result)
        else:
            result.warnings.append("spaCy not available, using fallback extraction")

        # Fallback/supplemental pattern extraction
        pattern_ingredients = self._extract_with_patterns(processed_text)

        # Merge results (avoid duplicates)
        existing_names = {i.name.lower() for i in result.ingredients}
        for ing in pattern_ingredients:
            if ing.name.lower() not in existing_names:
                result.ingredients.append(ing)
                existing_names.add(ing.name.lower())

        # Post-process
        result = self._postprocess(result)

        # Calculate overall confidence
        if result.ingredients:
            result.confidence = sum(i.confidence for i in result.ingredients) / len(result.ingredients)

        return result

    def extract_from_lines(self, lines: List[str]) -> List[ExtractionResult]:
        """Extract ingredients from multiple lines (typical ingredient list)."""
        results = []
        for line in lines:
            if line.strip():
                results.append(self.extract(line.strip()))
        return results

    def _preprocess(self, text: str) -> str:
        """Preprocess text for extraction."""
        # Normalize whitespace
        text = " ".join(text.split())

        # Normalize dashes and quotes
        text = text.replace("–", "-").replace("—", "-")
        text = text.replace(""", '"').replace(""", '"')
        text = text.replace("'", "'").replace("'", "'")

        # Remove parenthetical notes at end (e.g., "(about 2 cups)")
        text = re.sub(r"\s*\([^)]*\)\s*$", "", text)

        return text.strip()

    def _extract_with_spacy(
        self,
        text: str,
        result: ExtractionResult
    ) -> ExtractionResult:
        """Extract using spaCy NLP."""
        doc = self.nlp_processor.process(text)
        if not doc:
            return result

        # Get entities
        result.entities = self.nlp_processor.extract_entities(doc)

        # Get noun chunks as potential ingredients
        noun_chunks = self.nlp_processor.get_noun_chunks(doc)

        # Process each noun chunk as potential ingredient
        for chunk_text, root, start, end in noun_chunks:
            # Skip very short chunks
            if len(chunk_text) < 2:
                continue

            # Create ingredient from chunk
            ingredient = self._create_ingredient_from_chunk(
                chunk_text, root, start, end, text
            )

            if ingredient and ingredient.confidence >= self.confidence_threshold:
                result.ingredients.append(ingredient)

        # Also check for food entities from phrase matcher
        if "FOOD" in result.entities:
            for food_text, start, end in result.entities["FOOD"]:
                # Check if already captured in ingredients
                if not any(food_text.lower() in i.name.lower() for i in result.ingredients):
                    ingredient = ExtractedIngredient(
                        name=food_text,
                        original_text=food_text,
                        start_char=start,
                        end_char=end,
                        confidence=0.9,  # High confidence for vocabulary match
                    )
                    result.ingredients.append(ingredient)

        # Extract allergens
        if "ALLERGEN" in result.entities:
            allergen_set = {a[0].lower() for a in result.entities["ALLERGEN"]}
            for ingredient in result.ingredients:
                for allergen in allergen_set:
                    if allergen in ingredient.name.lower():
                        if allergen not in ingredient.allergens:
                            ingredient.allergens.append(allergen)

        return result

    def _create_ingredient_from_chunk(
        self,
        chunk_text: str,
        root: str,
        start: int,
        end: int,
        full_text: str
    ) -> Optional[ExtractedIngredient]:
        """Create ExtractedIngredient from noun chunk."""
        # Try to parse quantity from the beginning
        quantity_result = PatternLibrary.parse_quantity(chunk_text)

        name = chunk_text
        quantity_info = None

        if quantity_result:
            qty_value, remaining = quantity_result
            unit_result = PatternLibrary.extract_unit(remaining)

            if unit_result:
                unit, name = unit_result
                quantity_info = QuantityInfo(
                    value=qty_value,
                    unit=unit,
                    original_text=chunk_text[:len(chunk_text) - len(name)].strip(),
                    system=UnitConverter.get_system(unit),
                    normalized_grams=UnitConverter.to_grams(qty_value, unit, name),
                )
            else:
                name = remaining

        # Clean up ingredient name
        name = self._clean_ingredient_name(name)

        if not name or len(name) < 2:
            return None

        # Extract additional information
        preparations = PatternLibrary.extract_preparation(chunk_text)
        cooking_methods = PatternLibrary.extract_cooking_method(chunk_text)
        modifiers = PatternLibrary.extract_modifiers(chunk_text)

        # Calculate confidence
        confidence = self._calculate_confidence(name, quantity_info, preparations)

        return ExtractedIngredient(
            name=name,
            original_text=chunk_text,
            quantity=quantity_info,
            preparation=preparations[0] if preparations else None,
            cooking_method=cooking_methods[0] if cooking_methods else None,
            modifiers=modifiers,
            confidence=confidence,
            start_char=start,
            end_char=end,
        )

    def _extract_with_patterns(self, text: str) -> List[ExtractedIngredient]:
        """Fallback pattern-based extraction."""
        ingredients = []

        # Pattern: QUANTITY UNIT INGREDIENT
        pattern = r"(?:(\d+\.?\d*|\d+/\d+|[½⅓⅔¼¾⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞])\s*)?(?:(cup|cups|tbsp|tablespoon|tsp|teaspoon|oz|ounce|pound|lb|g|gram|kg|ml|liter|l|piece|pieces|slice|slices|clove|cloves|bunch|can|jar|package)s?\s+)?(?:of\s+)?(.+)"

        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            qty_str, unit, name = match.groups()

            quantity_info = None
            if qty_str:
                qty_result = PatternLibrary.parse_quantity(qty_str)
                if qty_result:
                    qty_value, _ = qty_result
                    quantity_info = QuantityInfo(
                        value=qty_value,
                        unit=unit or "",
                        original_text=f"{qty_str} {unit or ''}".strip(),
                        system=UnitConverter.get_system(unit or ""),
                        normalized_grams=UnitConverter.to_grams(
                            qty_value, unit or "", name
                        ) if unit else None,
                    )

            name = self._clean_ingredient_name(name)

            if name and len(name) >= 2:
                preparations = PatternLibrary.extract_preparation(text)
                modifiers = PatternLibrary.extract_modifiers(text)

                ingredient = ExtractedIngredient(
                    name=name,
                    original_text=text,
                    quantity=quantity_info,
                    preparation=preparations[0] if preparations else None,
                    modifiers=modifiers,
                    confidence=0.6,  # Lower confidence for pattern extraction
                    start_char=0,
                    end_char=len(text),
                )
                ingredients.append(ingredient)

        return ingredients

    def _clean_ingredient_name(self, name: str) -> str:
        """Clean and normalize ingredient name."""
        # Remove leading/trailing punctuation and whitespace
        name = name.strip(" ,;:.-")

        # Remove common filler words at beginning
        filler_words = ["of", "the", "a", "an", "some", "few", "optional"]
        words = name.split()
        while words and words[0].lower() in filler_words:
            words.pop(0)
        name = " ".join(words)

        # Remove preparation terms from name (they're captured separately)
        for prep in PatternLibrary.PREPARATIONS:
            name = re.sub(rf"\b{prep}\b", "", name, flags=re.IGNORECASE)

        # Clean up whitespace
        name = " ".join(name.split())

        return name.strip()

    def _calculate_confidence(
        self,
        name: str,
        quantity: Optional[QuantityInfo],
        preparations: List[str]
    ) -> float:
        """Calculate extraction confidence score."""
        confidence = 0.5  # Base confidence

        # Boost for having quantity
        if quantity:
            confidence += 0.2
            if quantity.normalized_grams:
                confidence += 0.1

        # Boost for having preparation
        if preparations:
            confidence += 0.1

        # Penalty for very short names
        if len(name) < 3:
            confidence -= 0.2

        # Penalty for names that are mostly numbers
        if sum(c.isdigit() for c in name) / max(len(name), 1) > 0.5:
            confidence -= 0.3

        # Boost for names that look like ingredients
        if any(
            term in name.lower()
            for term in ["chicken", "beef", "fish", "egg", "milk", "butter",
                        "oil", "flour", "sugar", "salt", "pepper", "onion",
                        "garlic", "tomato", "potato", "carrot", "celery"]
        ):
            confidence += 0.2

        return max(0.0, min(1.0, confidence))

    def _postprocess(self, result: ExtractionResult) -> ExtractionResult:
        """Post-process extraction results."""
        # Remove duplicates (keep higher confidence)
        seen: Dict[str, ExtractedIngredient] = {}
        for ing in result.ingredients:
            key = ing.name.lower()
            if key not in seen or ing.confidence > seen[key].confidence:
                seen[key] = ing

        result.ingredients = list(seen.values())

        # Sort by position in text
        result.ingredients.sort(key=lambda x: x.start_char)

        return result

    def get_semantic_similarity(self, text1: str, text2: str) -> float:
        """Get semantic similarity between two ingredient names."""
        return self.nlp_processor.get_similarity(text1, text2)

    def find_similar_ingredients(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5,
        threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        """Find semantically similar ingredients."""
        if not self._initialized:
            return []

        similarities = []
        for candidate in candidates:
            sim = self.get_semantic_similarity(query, candidate)
            if sim >= threshold:
                similarities.append((candidate, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


# ==================== Singleton Instance ====================

_nlp_extractor: Optional[NLPIngredientExtractor] = None


def get_nlp_extractor() -> NLPIngredientExtractor:
    """Get or create the NLP extractor singleton."""
    global _nlp_extractor

    if _nlp_extractor is None:
        _nlp_extractor = NLPIngredientExtractor()
        _nlp_extractor.initialize()

    return _nlp_extractor


# ==================== Convenience Functions ====================

def extract_ingredients(text: str) -> ExtractionResult:
    """Extract ingredients from text (convenience function)."""
    extractor = get_nlp_extractor()
    return extractor.extract(text)


def extract_ingredients_from_list(lines: List[str]) -> List[ExtractedIngredient]:
    """Extract ingredients from a list of lines."""
    extractor = get_nlp_extractor()
    results = extractor.extract_from_lines(lines)

    all_ingredients = []
    for result in results:
        all_ingredients.extend(result.ingredients)

    return all_ingredients
