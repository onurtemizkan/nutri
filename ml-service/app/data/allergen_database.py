"""
Comprehensive Allergen and Food Sensitivity Database

This module contains research-backed data for:
- FDA Big 9 allergens (2023)
- EU Big 14 allergens
- Hidden allergen ingredient mappings
- Histamine content (mg/100g)
- Tyramine content (mg/100g)
- FODMAP levels
- Oxalate, salicylate, lectin information
- Cross-reactivity mappings

Sources:
- USDA FoodData Central
- Monash University FODMAP Database
- SIGHI Histamine List
- Research: PMC7305651 (Biogenic Amines in Food)
- Research: PMC10830535 (Biogenic Amines Review)
- FDA Food Allergen Labeling Guidelines (2025)
"""
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from app.models.sensitivity import (
    AllergenType,
    CompoundLevel,
    FodmapLevel,
    IngredientCategory,
    DerivationType,
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class AllergenMapping:
    """Maps an ingredient to an allergen type"""

    allergen: AllergenType
    confidence: float = 1.0
    derivation: DerivationType = DerivationType.DIRECTLY_CONTAINS


@dataclass
class IngredientData:
    """Complete ingredient entry with sensitivity data"""

    name: str
    display_name: str
    category: IngredientCategory
    name_variants: List[str] = field(default_factory=list)

    # Allergen mappings
    allergens: List[AllergenMapping] = field(default_factory=list)

    # FODMAP
    fodmap_level: Optional[FodmapLevel] = None
    fodmap_types: List[str] = field(default_factory=list)

    # Biogenic amines (mg per 100g)
    histamine_mg: Optional[float] = None
    histamine_level: Optional[CompoundLevel] = None
    tyramine_mg: Optional[float] = None
    tyramine_level: Optional[CompoundLevel] = None

    # Other compounds
    oxalate_mg: Optional[float] = None
    oxalate_level: Optional[CompoundLevel] = None
    salicylate_mg: Optional[float] = None
    salicylate_level: Optional[CompoundLevel] = None
    lectin_level: Optional[CompoundLevel] = None

    # Flags
    is_nightshade: bool = False
    is_fermented: bool = False
    is_aged: bool = False
    is_histamine_liberator: bool = False  # Foods that trigger histamine release

    # Data source
    sources: List[str] = field(default_factory=lambda: ["research"])


# =============================================================================
# HELPER FUNCTIONS FOR COMPOUND LEVELS
# =============================================================================


def histamine_to_level(mg: Optional[float]) -> Optional[CompoundLevel]:
    """Convert histamine mg/100g to CompoundLevel"""
    if mg is None:
        return None
    if mg < 1:
        return CompoundLevel.NEGLIGIBLE
    if mg < 10:
        return CompoundLevel.LOW
    if mg < 50:
        return CompoundLevel.MEDIUM
    if mg < 200:
        return CompoundLevel.HIGH
    return CompoundLevel.VERY_HIGH


def tyramine_to_level(mg: Optional[float]) -> Optional[CompoundLevel]:
    """Convert tyramine mg/100g to CompoundLevel"""
    if mg is None:
        return None
    if mg < 1:
        return CompoundLevel.NEGLIGIBLE
    if mg < 10:
        return CompoundLevel.LOW
    if mg < 50:
        return CompoundLevel.MEDIUM
    if mg < 200:
        return CompoundLevel.HIGH
    return CompoundLevel.VERY_HIGH


def oxalate_to_level(mg: Optional[float]) -> Optional[CompoundLevel]:
    """Convert oxalate mg/100g to CompoundLevel (per serving thresholds)"""
    if mg is None:
        return None
    if mg < 5:
        return CompoundLevel.NEGLIGIBLE
    if mg < 10:
        return CompoundLevel.LOW
    if mg < 30:
        return CompoundLevel.MEDIUM
    if mg < 100:
        return CompoundLevel.HIGH
    return CompoundLevel.VERY_HIGH


# =============================================================================
# HIDDEN ALLERGEN SYNONYMS
# =============================================================================

# Maps hidden ingredient names to their allergen type
HIDDEN_ALLERGEN_KEYWORDS: Dict[str, AllergenType] = {
    # MILK
    "casein": AllergenType.MILK,
    "caseinate": AllergenType.MILK,
    "sodium caseinate": AllergenType.MILK,
    "calcium caseinate": AllergenType.MILK,
    "whey": AllergenType.MILK,
    "whey protein": AllergenType.MILK,
    "whey protein concentrate": AllergenType.MILK,
    "whey protein isolate": AllergenType.MILK,
    "lactose": AllergenType.MILK,
    "lactalbumin": AllergenType.MILK,
    "lactoglobulin": AllergenType.MILK,
    "lactoferrin": AllergenType.MILK,
    "curds": AllergenType.MILK,
    "ghee": AllergenType.MILK,
    "paneer": AllergenType.MILK,
    "cream": AllergenType.MILK,
    "butter": AllergenType.MILK,
    "butterfat": AllergenType.MILK,
    "buttermilk": AllergenType.MILK,
    "half and half": AllergenType.MILK,
    "half-and-half": AllergenType.MILK,
    "sour cream": AllergenType.MILK,
    "yogurt": AllergenType.MILK,
    "kefir": AllergenType.MILK,
    "cottage cheese": AllergenType.MILK,
    "ricotta": AllergenType.MILK,
    "custard": AllergenType.MILK,
    "pudding": AllergenType.MILK,
    "ice cream": AllergenType.MILK,
    "gelato": AllergenType.MILK,
    "nougat": AllergenType.MILK,
    "recaldent": AllergenType.MILK,
    "simplesse": AllergenType.MILK,
    # EGGS
    "albumin": AllergenType.EGGS,
    "ovalbumin": AllergenType.EGGS,
    "ovomucin": AllergenType.EGGS,
    "ovomucoid": AllergenType.EGGS,
    "ovovitellin": AllergenType.EGGS,
    "globulin": AllergenType.EGGS,
    "livetin": AllergenType.EGGS,
    "lysozyme": AllergenType.EGGS,
    "meringue": AllergenType.EGGS,
    "mayonnaise": AllergenType.EGGS,
    "eggnog": AllergenType.EGGS,
    "egg lecithin": AllergenType.EGGS,
    "egg wash": AllergenType.EGGS,
    # WHEAT / GLUTEN
    "wheat flour": AllergenType.WHEAT,
    "enriched flour": AllergenType.WHEAT,
    "all-purpose flour": AllergenType.WHEAT,
    "bread flour": AllergenType.WHEAT,
    "cake flour": AllergenType.WHEAT,
    "pastry flour": AllergenType.WHEAT,
    "semolina": AllergenType.WHEAT,
    "durum": AllergenType.WHEAT,
    "spelt": AllergenType.WHEAT,
    "kamut": AllergenType.WHEAT,
    "einkorn": AllergenType.WHEAT,
    "emmer": AllergenType.WHEAT,
    "farina": AllergenType.WHEAT,
    "couscous": AllergenType.WHEAT,
    "bulgur": AllergenType.WHEAT,
    "seitan": AllergenType.GLUTEN_CEREALS,
    "vital wheat gluten": AllergenType.GLUTEN_CEREALS,
    "gluten": AllergenType.GLUTEN_CEREALS,
    "hydrolyzed wheat protein": AllergenType.WHEAT,
    "wheat starch": AllergenType.WHEAT,
    "modified wheat starch": AllergenType.WHEAT,
    "wheat germ": AllergenType.WHEAT,
    "wheat bran": AllergenType.WHEAT,
    "triticale": AllergenType.WHEAT,
    "malt": AllergenType.GLUTEN_CEREALS,
    "malt extract": AllergenType.GLUTEN_CEREALS,
    "malt flavoring": AllergenType.GLUTEN_CEREALS,
    "malt vinegar": AllergenType.GLUTEN_CEREALS,
    "barley": AllergenType.GLUTEN_CEREALS,
    "rye": AllergenType.GLUTEN_CEREALS,
    "surimi": AllergenType.WHEAT,  # Imitation crab often contains wheat
    # SOY
    "soy": AllergenType.SOY,
    "soya": AllergenType.SOY,
    "soybean": AllergenType.SOY,
    "soy lecithin": AllergenType.SOY,
    "soy protein": AllergenType.SOY,
    "soy protein isolate": AllergenType.SOY,
    "soy protein concentrate": AllergenType.SOY,
    "hydrolyzed soy protein": AllergenType.SOY,
    "textured vegetable protein": AllergenType.SOY,
    "tvp": AllergenType.SOY,
    "tofu": AllergenType.SOY,
    "tempeh": AllergenType.SOY,
    "miso": AllergenType.SOY,
    "natto": AllergenType.SOY,
    "edamame": AllergenType.SOY,
    "soy sauce": AllergenType.SOY,
    "shoyu": AllergenType.SOY,
    "tamari": AllergenType.SOY,
    "teriyaki": AllergenType.SOY,
    "soy milk": AllergenType.SOY,
    # PEANUTS
    "peanut": AllergenType.PEANUTS,
    "peanuts": AllergenType.PEANUTS,
    "groundnut": AllergenType.PEANUTS,
    "groundnuts": AllergenType.PEANUTS,
    "arachis oil": AllergenType.PEANUTS,
    "arachis hypogaea": AllergenType.PEANUTS,
    "monkey nuts": AllergenType.PEANUTS,
    "peanut butter": AllergenType.PEANUTS,
    "peanut flour": AllergenType.PEANUTS,
    "peanut oil": AllergenType.PEANUTS,
    # TREE NUTS
    "almond": AllergenType.TREE_NUTS,
    "almonds": AllergenType.TREE_NUTS,
    "cashew": AllergenType.TREE_NUTS,
    "cashews": AllergenType.TREE_NUTS,
    "walnut": AllergenType.TREE_NUTS,
    "walnuts": AllergenType.TREE_NUTS,
    "pecan": AllergenType.TREE_NUTS,
    "pecans": AllergenType.TREE_NUTS,
    "pistachio": AllergenType.TREE_NUTS,
    "pistachios": AllergenType.TREE_NUTS,
    "hazelnut": AllergenType.TREE_NUTS,
    "hazelnuts": AllergenType.TREE_NUTS,
    "filbert": AllergenType.TREE_NUTS,
    "macadamia": AllergenType.TREE_NUTS,
    "macadamia nut": AllergenType.TREE_NUTS,
    "brazil nut": AllergenType.TREE_NUTS,
    "brazil nuts": AllergenType.TREE_NUTS,
    "pine nut": AllergenType.TREE_NUTS,
    "pine nuts": AllergenType.TREE_NUTS,
    "pignoli": AllergenType.TREE_NUTS,
    "praline": AllergenType.TREE_NUTS,
    "marzipan": AllergenType.TREE_NUTS,
    "almond paste": AllergenType.TREE_NUTS,
    "nougat": AllergenType.TREE_NUTS,
    "gianduja": AllergenType.TREE_NUTS,
    "nutella": AllergenType.TREE_NUTS,
    # FISH
    "anchovy": AllergenType.FISH,
    "anchovies": AllergenType.FISH,
    "bass": AllergenType.FISH,
    "catfish": AllergenType.FISH,
    "cod": AllergenType.FISH,
    "flounder": AllergenType.FISH,
    "haddock": AllergenType.FISH,
    "halibut": AllergenType.FISH,
    "herring": AllergenType.FISH,
    "mackerel": AllergenType.FISH,
    "mahi mahi": AllergenType.FISH,
    "perch": AllergenType.FISH,
    "pike": AllergenType.FISH,
    "pollock": AllergenType.FISH,
    "salmon": AllergenType.FISH,
    "sardine": AllergenType.FISH,
    "sardines": AllergenType.FISH,
    "snapper": AllergenType.FISH,
    "sole": AllergenType.FISH,
    "swordfish": AllergenType.FISH,
    "tilapia": AllergenType.FISH,
    "trout": AllergenType.FISH,
    "tuna": AllergenType.FISH,
    "fish sauce": AllergenType.FISH,
    "fish oil": AllergenType.FISH,
    "omega-3": AllergenType.FISH,  # Often fish-derived
    "worcestershire": AllergenType.FISH,  # Contains anchovies
    "caesar dressing": AllergenType.FISH,  # Contains anchovies
    # SHELLFISH - CRUSTACEANS
    "shrimp": AllergenType.SHELLFISH_CRUSTACEAN,
    "prawn": AllergenType.SHELLFISH_CRUSTACEAN,
    "prawns": AllergenType.SHELLFISH_CRUSTACEAN,
    "crab": AllergenType.SHELLFISH_CRUSTACEAN,
    "lobster": AllergenType.SHELLFISH_CRUSTACEAN,
    "crayfish": AllergenType.SHELLFISH_CRUSTACEAN,
    "crawfish": AllergenType.SHELLFISH_CRUSTACEAN,
    "langoustine": AllergenType.SHELLFISH_CRUSTACEAN,
    "krill": AllergenType.SHELLFISH_CRUSTACEAN,
    # SHELLFISH - MOLLUSKS
    "clam": AllergenType.SHELLFISH_MOLLUSCAN,
    "clams": AllergenType.SHELLFISH_MOLLUSCAN,
    "mussel": AllergenType.SHELLFISH_MOLLUSCAN,
    "mussels": AllergenType.SHELLFISH_MOLLUSCAN,
    "oyster": AllergenType.SHELLFISH_MOLLUSCAN,
    "oysters": AllergenType.SHELLFISH_MOLLUSCAN,
    "scallop": AllergenType.SHELLFISH_MOLLUSCAN,
    "scallops": AllergenType.SHELLFISH_MOLLUSCAN,
    "squid": AllergenType.SHELLFISH_MOLLUSCAN,
    "calamari": AllergenType.SHELLFISH_MOLLUSCAN,
    "octopus": AllergenType.SHELLFISH_MOLLUSCAN,
    "snail": AllergenType.SHELLFISH_MOLLUSCAN,
    "escargot": AllergenType.SHELLFISH_MOLLUSCAN,
    "abalone": AllergenType.SHELLFISH_MOLLUSCAN,
    # SESAME
    "sesame": AllergenType.SESAME,
    "sesame seeds": AllergenType.SESAME,
    "sesame oil": AllergenType.SESAME,
    "tahini": AllergenType.SESAME,
    "halvah": AllergenType.SESAME,
    "halva": AllergenType.SESAME,
    "hummus": AllergenType.SESAME,  # Contains tahini
    "benne seeds": AllergenType.SESAME,
    "gingelly oil": AllergenType.SESAME,
    # MUSTARD
    "mustard": AllergenType.MUSTARD,
    "mustard seed": AllergenType.MUSTARD,
    "mustard oil": AllergenType.MUSTARD,
    "mustard flour": AllergenType.MUSTARD,
    "mustard powder": AllergenType.MUSTARD,
    "dijon": AllergenType.MUSTARD,
    # CELERY
    "celery": AllergenType.CELERY,
    "celeriac": AllergenType.CELERY,
    "celery salt": AllergenType.CELERY,
    "celery seed": AllergenType.CELERY,
    # LUPIN
    "lupin": AllergenType.LUPIN,
    "lupini": AllergenType.LUPIN,
    "lupine": AllergenType.LUPIN,
    "lupin flour": AllergenType.LUPIN,
    # SULFITES
    "sulfite": AllergenType.SULFITES,
    "sulfites": AllergenType.SULFITES,
    "sulphite": AllergenType.SULFITES,
    "sulphites": AllergenType.SULFITES,
    "sulfur dioxide": AllergenType.SULFITES,
    "sodium sulfite": AllergenType.SULFITES,
    "sodium bisulfite": AllergenType.SULFITES,
    "sodium metabisulfite": AllergenType.SULFITES,
    "potassium bisulfite": AllergenType.SULFITES,
    "potassium metabisulfite": AllergenType.SULFITES,
}


# =============================================================================
# CROSS-REACTIVITY MAPPINGS
# =============================================================================

# If user is allergic to X, they should also watch for Y
CROSS_REACTIVITY: Dict[AllergenType, List[AllergenType]] = {
    AllergenType.PEANUTS: [AllergenType.TREE_NUTS, AllergenType.LUPIN],
    AllergenType.TREE_NUTS: [AllergenType.PEANUTS],
    AllergenType.SHELLFISH_CRUSTACEAN: [AllergenType.SHELLFISH_MOLLUSCAN],
    AllergenType.SHELLFISH_MOLLUSCAN: [AllergenType.SHELLFISH_CRUSTACEAN],
    AllergenType.MILK: [],  # Goat/sheep milk should be separate entries
    AllergenType.WHEAT: [AllergenType.GLUTEN_CEREALS],
    AllergenType.GLUTEN_CEREALS: [AllergenType.WHEAT],
    AllergenType.LUPIN: [AllergenType.PEANUTS],  # Legume family
}


# =============================================================================
# COMPREHENSIVE INGREDIENT DATABASE
# =============================================================================

INGREDIENT_DATABASE: Dict[str, IngredientData] = {
    # =========================================================================
    # DAIRY PRODUCTS
    # =========================================================================
    "milk": IngredientData(
        name="milk",
        display_name="Milk",
        category=IngredientCategory.DAIRY,
        name_variants=["cow's milk", "whole milk", "2% milk", "skim milk", "low-fat milk"],
        allergens=[AllergenMapping(AllergenType.MILK)],
        fodmap_level=FodmapLevel.HIGH,
        fodmap_types=["lactose"],
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
    ),
    "cheese_aged": IngredientData(
        name="cheese_aged",
        display_name="Aged Cheese",
        category=IngredientCategory.DAIRY,
        name_variants=[
            "cheddar", "parmesan", "parmigiano", "gouda", "gruyere",
            "blue cheese", "gorgonzola", "roquefort", "stilton",
            "aged cheddar", "sharp cheddar", "romano", "asiago"
        ],
        allergens=[AllergenMapping(AllergenType.MILK)],
        fodmap_level=FodmapLevel.LOW,  # Aged cheeses are low FODMAP
        histamine_mg=500,  # Can range 10-2500 mg/kg
        histamine_level=CompoundLevel.VERY_HIGH,
        tyramine_mg=200,  # Can range 10-2210 mg/kg
        tyramine_level=CompoundLevel.VERY_HIGH,
        is_aged=True,
    ),
    "cheese_fresh": IngredientData(
        name="cheese_fresh",
        display_name="Fresh Cheese",
        category=IngredientCategory.DAIRY,
        name_variants=[
            "mozzarella", "cottage cheese", "cream cheese", "ricotta",
            "mascarpone", "queso fresco", "feta", "brie", "camembert"
        ],
        allergens=[AllergenMapping(AllergenType.MILK)],
        fodmap_level=FodmapLevel.MEDIUM,
        fodmap_types=["lactose"],
        histamine_mg=5,
        histamine_level=CompoundLevel.LOW,
        tyramine_mg=5,
        tyramine_level=CompoundLevel.LOW,
    ),
    "yogurt": IngredientData(
        name="yogurt",
        display_name="Yogurt",
        category=IngredientCategory.DAIRY,
        name_variants=["greek yogurt", "plain yogurt", "natural yogurt"],
        allergens=[AllergenMapping(AllergenType.MILK)],
        fodmap_level=FodmapLevel.HIGH,
        fodmap_types=["lactose"],
        histamine_mg=10,
        histamine_level=CompoundLevel.LOW,
        is_fermented=True,
    ),
    "butter": IngredientData(
        name="butter",
        display_name="Butter",
        category=IngredientCategory.DAIRY,
        name_variants=["unsalted butter", "salted butter", "clarified butter"],
        allergens=[AllergenMapping(AllergenType.MILK)],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
    ),
    "ghee": IngredientData(
        name="ghee",
        display_name="Ghee",
        category=IngredientCategory.DAIRY,
        name_variants=["clarified butter"],
        allergens=[
            AllergenMapping(AllergenType.MILK, confidence=0.3, derivation=DerivationType.DERIVED_FROM)
        ],  # Very low milk protein
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=0.1,
        histamine_level=CompoundLevel.NEGLIGIBLE,
    ),
    "whey_protein": IngredientData(
        name="whey_protein",
        display_name="Whey Protein",
        category=IngredientCategory.DAIRY,
        name_variants=["whey", "whey protein isolate", "whey protein concentrate"],
        allergens=[AllergenMapping(AllergenType.MILK, derivation=DerivationType.DERIVED_FROM)],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=1,
        histamine_level=CompoundLevel.NEGLIGIBLE,
    ),
    "casein": IngredientData(
        name="casein",
        display_name="Casein",
        category=IngredientCategory.DAIRY,
        name_variants=["sodium caseinate", "calcium caseinate", "casein protein"],
        allergens=[AllergenMapping(AllergenType.MILK, derivation=DerivationType.DERIVED_FROM)],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=1,
        histamine_level=CompoundLevel.NEGLIGIBLE,
    ),

    # =========================================================================
    # EGGS
    # =========================================================================
    "egg": IngredientData(
        name="egg",
        display_name="Egg",
        category=IngredientCategory.EGGS,
        name_variants=["eggs", "whole egg", "chicken egg", "hen's egg"],
        allergens=[AllergenMapping(AllergenType.EGGS)],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
    ),
    "egg_white": IngredientData(
        name="egg_white",
        display_name="Egg White",
        category=IngredientCategory.EGGS,
        name_variants=["egg whites", "albumin", "dried egg white"],
        allergens=[AllergenMapping(AllergenType.EGGS)],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
    ),
    "egg_yolk": IngredientData(
        name="egg_yolk",
        display_name="Egg Yolk",
        category=IngredientCategory.EGGS,
        name_variants=["egg yolks", "yolk"],
        allergens=[AllergenMapping(AllergenType.EGGS)],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
    ),
    "mayonnaise": IngredientData(
        name="mayonnaise",
        display_name="Mayonnaise",
        category=IngredientCategory.EGGS,
        name_variants=["mayo", "aioli"],
        allergens=[
            AllergenMapping(AllergenType.EGGS),
            AllergenMapping(AllergenType.SOY, confidence=0.7, derivation=DerivationType.MAY_CONTAIN),
        ],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=1,
        histamine_level=CompoundLevel.NEGLIGIBLE,
    ),

    # =========================================================================
    # WHEAT & GLUTEN
    # =========================================================================
    "wheat_flour": IngredientData(
        name="wheat_flour",
        display_name="Wheat Flour",
        category=IngredientCategory.GRAINS,
        name_variants=[
            "flour", "all-purpose flour", "bread flour", "cake flour",
            "pastry flour", "enriched flour", "bleached flour", "unbleached flour"
        ],
        allergens=[
            AllergenMapping(AllergenType.WHEAT),
            AllergenMapping(AllergenType.GLUTEN_CEREALS),
        ],
        fodmap_level=FodmapLevel.HIGH,
        fodmap_types=["fructans"],
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
        lectin_level=CompoundLevel.MEDIUM,
    ),
    "bread": IngredientData(
        name="bread",
        display_name="Bread",
        category=IngredientCategory.GRAINS,
        name_variants=["white bread", "whole wheat bread", "toast", "baguette", "ciabatta"],
        allergens=[
            AllergenMapping(AllergenType.WHEAT),
            AllergenMapping(AllergenType.GLUTEN_CEREALS),
            AllergenMapping(AllergenType.SOY, confidence=0.5, derivation=DerivationType.MAY_CONTAIN),
        ],
        fodmap_level=FodmapLevel.HIGH,
        fodmap_types=["fructans"],
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
        lectin_level=CompoundLevel.LOW,  # Reduced after baking
    ),
    "pasta": IngredientData(
        name="pasta",
        display_name="Pasta",
        category=IngredientCategory.GRAINS,
        name_variants=[
            "spaghetti", "penne", "fettuccine", "linguine", "macaroni",
            "noodles", "egg noodles", "lasagna"
        ],
        allergens=[
            AllergenMapping(AllergenType.WHEAT),
            AllergenMapping(AllergenType.GLUTEN_CEREALS),
            AllergenMapping(AllergenType.EGGS, confidence=0.6, derivation=DerivationType.MAY_CONTAIN),
        ],
        fodmap_level=FodmapLevel.HIGH,
        fodmap_types=["fructans"],
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
    ),
    "seitan": IngredientData(
        name="seitan",
        display_name="Seitan",
        category=IngredientCategory.GRAINS,
        name_variants=["wheat gluten", "vital wheat gluten", "wheat meat"],
        allergens=[
            AllergenMapping(AllergenType.WHEAT),
            AllergenMapping(AllergenType.GLUTEN_CEREALS),
        ],
        fodmap_level=FodmapLevel.LOW,  # Pure gluten, no fructans
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
    ),
    "soy_sauce": IngredientData(
        name="soy_sauce",
        display_name="Soy Sauce",
        category=IngredientCategory.FERMENTED,
        name_variants=["shoyu", "tamari", "light soy sauce", "dark soy sauce"],
        allergens=[
            AllergenMapping(AllergenType.SOY),
            AllergenMapping(AllergenType.WHEAT),
            AllergenMapping(AllergenType.GLUTEN_CEREALS),
        ],
        fodmap_level=FodmapLevel.LOW,  # Small serving
        histamine_mg=20,
        histamine_level=CompoundLevel.MEDIUM,
        tyramine_mg=50,
        tyramine_level=CompoundLevel.MEDIUM,
        is_fermented=True,
    ),

    # =========================================================================
    # SOY PRODUCTS
    # =========================================================================
    "soy": IngredientData(
        name="soy",
        display_name="Soy",
        category=IngredientCategory.LEGUMES,
        name_variants=["soya", "soybeans", "soy beans", "edamame"],
        allergens=[AllergenMapping(AllergenType.SOY)],
        fodmap_level=FodmapLevel.HIGH,
        fodmap_types=["galactans", "fructans"],
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
        lectin_level=CompoundLevel.HIGH,
    ),
    "tofu": IngredientData(
        name="tofu",
        display_name="Tofu",
        category=IngredientCategory.LEGUMES,
        name_variants=["bean curd", "firm tofu", "silken tofu", "soft tofu"],
        allergens=[AllergenMapping(AllergenType.SOY)],
        fodmap_level=FodmapLevel.LOW,  # Firm tofu is low FODMAP
        histamine_mg=5,
        histamine_level=CompoundLevel.LOW,
        lectin_level=CompoundLevel.LOW,  # Reduced in processing
    ),
    "tempeh": IngredientData(
        name="tempeh",
        display_name="Tempeh",
        category=IngredientCategory.LEGUMES,
        name_variants=["fermented soy", "soy tempeh"],
        allergens=[AllergenMapping(AllergenType.SOY)],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=30,
        histamine_level=CompoundLevel.MEDIUM,
        tyramine_mg=20,
        tyramine_level=CompoundLevel.MEDIUM,
        is_fermented=True,
        lectin_level=CompoundLevel.LOW,
    ),
    "miso": IngredientData(
        name="miso",
        display_name="Miso",
        category=IngredientCategory.FERMENTED,
        name_variants=["miso paste", "white miso", "red miso", "brown miso"],
        allergens=[
            AllergenMapping(AllergenType.SOY),
            AllergenMapping(AllergenType.GLUTEN_CEREALS, confidence=0.7),
        ],
        fodmap_level=FodmapLevel.LOW,  # Small serving
        histamine_mg=40,
        histamine_level=CompoundLevel.MEDIUM,
        tyramine_mg=100,
        tyramine_level=CompoundLevel.HIGH,
        is_fermented=True,
        is_aged=True,
    ),
    "soy_lecithin": IngredientData(
        name="soy_lecithin",
        display_name="Soy Lecithin",
        category=IngredientCategory.ADDITIVES,
        name_variants=["lecithin (soy)", "E322"],
        allergens=[
            AllergenMapping(AllergenType.SOY, confidence=0.7, derivation=DerivationType.DERIVED_FROM)
        ],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=0.1,
        histamine_level=CompoundLevel.NEGLIGIBLE,
    ),

    # =========================================================================
    # PEANUTS
    # =========================================================================
    "peanut": IngredientData(
        name="peanut",
        display_name="Peanut",
        category=IngredientCategory.LEGUMES,  # Peanuts are legumes!
        name_variants=["peanuts", "groundnut", "groundnuts", "monkey nuts"],
        allergens=[AllergenMapping(AllergenType.PEANUTS)],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
        lectin_level=CompoundLevel.HIGH,
    ),
    "peanut_butter": IngredientData(
        name="peanut_butter",
        display_name="Peanut Butter",
        category=IngredientCategory.LEGUMES,
        name_variants=["peanut spread", "groundnut butter"],
        allergens=[AllergenMapping(AllergenType.PEANUTS)],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
        lectin_level=CompoundLevel.MEDIUM,
    ),

    # =========================================================================
    # TREE NUTS
    # =========================================================================
    "almond": IngredientData(
        name="almond",
        display_name="Almond",
        category=IngredientCategory.NUTS_SEEDS,
        name_variants=["almonds", "almond flour", "almond meal", "almond butter"],
        allergens=[AllergenMapping(AllergenType.TREE_NUTS)],
        fodmap_level=FodmapLevel.LOW,  # Up to 10 nuts
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
        oxalate_mg=469,
        oxalate_level=CompoundLevel.VERY_HIGH,
        lectin_level=CompoundLevel.MEDIUM,
    ),
    "cashew": IngredientData(
        name="cashew",
        display_name="Cashew",
        category=IngredientCategory.NUTS_SEEDS,
        name_variants=["cashews", "cashew nuts", "cashew butter"],
        allergens=[AllergenMapping(AllergenType.TREE_NUTS)],
        fodmap_level=FodmapLevel.HIGH,
        fodmap_types=["galactans"],
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
        oxalate_mg=262,
        oxalate_level=CompoundLevel.VERY_HIGH,
    ),
    "walnut": IngredientData(
        name="walnut",
        display_name="Walnut",
        category=IngredientCategory.NUTS_SEEDS,
        name_variants=["walnuts", "walnut halves", "walnut pieces"],
        allergens=[AllergenMapping(AllergenType.TREE_NUTS)],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
        oxalate_mg=74,
        oxalate_level=CompoundLevel.HIGH,
    ),
    "hazelnut": IngredientData(
        name="hazelnut",
        display_name="Hazelnut",
        category=IngredientCategory.NUTS_SEEDS,
        name_variants=["hazelnuts", "filbert", "filberts", "cob nuts"],
        allergens=[AllergenMapping(AllergenType.TREE_NUTS)],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
    ),
    "pistachio": IngredientData(
        name="pistachio",
        display_name="Pistachio",
        category=IngredientCategory.NUTS_SEEDS,
        name_variants=["pistachios", "pistachio nuts"],
        allergens=[AllergenMapping(AllergenType.TREE_NUTS)],
        fodmap_level=FodmapLevel.HIGH,
        fodmap_types=["fructans"],
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
    ),
    "macadamia": IngredientData(
        name="macadamia",
        display_name="Macadamia",
        category=IngredientCategory.NUTS_SEEDS,
        name_variants=["macadamia nuts", "macadamia nut"],
        allergens=[AllergenMapping(AllergenType.TREE_NUTS)],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
    ),

    # =========================================================================
    # FISH & SEAFOOD
    # =========================================================================
    "tuna": IngredientData(
        name="tuna",
        display_name="Tuna",
        category=IngredientCategory.SEAFOOD,
        name_variants=["tuna fish", "canned tuna", "fresh tuna", "ahi tuna"],
        allergens=[AllergenMapping(AllergenType.FISH)],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=100,  # Can be 0-300+ depending on freshness
        histamine_level=CompoundLevel.HIGH,
        tyramine_mg=10,
        tyramine_level=CompoundLevel.LOW,
    ),
    "salmon": IngredientData(
        name="salmon",
        display_name="Salmon",
        category=IngredientCategory.SEAFOOD,
        name_variants=["fresh salmon", "smoked salmon", "salmon fillet"],
        allergens=[AllergenMapping(AllergenType.FISH)],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=10,  # Fresh salmon is low
        histamine_level=CompoundLevel.LOW,
    ),
    "smoked_salmon": IngredientData(
        name="smoked_salmon",
        display_name="Smoked Salmon",
        category=IngredientCategory.SEAFOOD,
        name_variants=["lox", "nova", "gravlax"],
        allergens=[AllergenMapping(AllergenType.FISH)],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=50,
        histamine_level=CompoundLevel.MEDIUM,
        tyramine_mg=30,
        tyramine_level=CompoundLevel.MEDIUM,
        is_aged=True,
    ),
    "sardine": IngredientData(
        name="sardine",
        display_name="Sardine",
        category=IngredientCategory.SEAFOOD,
        name_variants=["sardines", "canned sardines"],
        allergens=[AllergenMapping(AllergenType.FISH)],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=150,
        histamine_level=CompoundLevel.HIGH,
    ),
    "anchovy": IngredientData(
        name="anchovy",
        display_name="Anchovy",
        category=IngredientCategory.SEAFOOD,
        name_variants=["anchovies", "anchovy paste", "anchovy fillet"],
        allergens=[AllergenMapping(AllergenType.FISH)],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=200,
        histamine_level=CompoundLevel.VERY_HIGH,
        tyramine_mg=50,
        tyramine_level=CompoundLevel.MEDIUM,
        is_aged=True,
    ),
    "mackerel": IngredientData(
        name="mackerel",
        display_name="Mackerel",
        category=IngredientCategory.SEAFOOD,
        name_variants=["canned mackerel", "fresh mackerel"],
        allergens=[AllergenMapping(AllergenType.FISH)],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=150,
        histamine_level=CompoundLevel.HIGH,
    ),
    "shrimp": IngredientData(
        name="shrimp",
        display_name="Shrimp",
        category=IngredientCategory.SEAFOOD,
        name_variants=["prawns", "prawn", "jumbo shrimp"],
        allergens=[AllergenMapping(AllergenType.SHELLFISH_CRUSTACEAN)],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=5,
        histamine_level=CompoundLevel.LOW,
    ),
    "crab": IngredientData(
        name="crab",
        display_name="Crab",
        category=IngredientCategory.SEAFOOD,
        name_variants=["crab meat", "king crab", "snow crab", "dungeness"],
        allergens=[AllergenMapping(AllergenType.SHELLFISH_CRUSTACEAN)],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=5,
        histamine_level=CompoundLevel.LOW,
    ),
    "lobster": IngredientData(
        name="lobster",
        display_name="Lobster",
        category=IngredientCategory.SEAFOOD,
        name_variants=["lobster tail", "maine lobster"],
        allergens=[AllergenMapping(AllergenType.SHELLFISH_CRUSTACEAN)],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=5,
        histamine_level=CompoundLevel.LOW,
    ),
    "scallop": IngredientData(
        name="scallop",
        display_name="Scallop",
        category=IngredientCategory.SEAFOOD,
        name_variants=["scallops", "sea scallop", "bay scallop"],
        allergens=[AllergenMapping(AllergenType.SHELLFISH_MOLLUSCAN)],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=5,
        histamine_level=CompoundLevel.LOW,
    ),
    "mussel": IngredientData(
        name="mussel",
        display_name="Mussel",
        category=IngredientCategory.SEAFOOD,
        name_variants=["mussels", "blue mussels", "green-lipped mussels"],
        allergens=[AllergenMapping(AllergenType.SHELLFISH_MOLLUSCAN)],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=20,
        histamine_level=CompoundLevel.MEDIUM,
    ),
    "oyster": IngredientData(
        name="oyster",
        display_name="Oyster",
        category=IngredientCategory.SEAFOOD,
        name_variants=["oysters", "raw oysters", "cooked oysters"],
        allergens=[AllergenMapping(AllergenType.SHELLFISH_MOLLUSCAN)],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=10,
        histamine_level=CompoundLevel.LOW,
    ),

    # =========================================================================
    # SESAME
    # =========================================================================
    "sesame": IngredientData(
        name="sesame",
        display_name="Sesame",
        category=IngredientCategory.NUTS_SEEDS,
        name_variants=["sesame seeds", "benne seeds"],
        allergens=[AllergenMapping(AllergenType.SESAME)],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
    ),
    "tahini": IngredientData(
        name="tahini",
        display_name="Tahini",
        category=IngredientCategory.NUTS_SEEDS,
        name_variants=["sesame paste", "sesame butter", "tahina"],
        allergens=[AllergenMapping(AllergenType.SESAME)],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
    ),
    "hummus": IngredientData(
        name="hummus",
        display_name="Hummus",
        category=IngredientCategory.LEGUMES,
        name_variants=["houmous", "humous", "chickpea dip"],
        allergens=[
            AllergenMapping(AllergenType.SESAME),  # Contains tahini
        ],
        fodmap_level=FodmapLevel.HIGH,
        fodmap_types=["galactans"],
        histamine_mg=1,
        histamine_level=CompoundLevel.NEGLIGIBLE,
    ),

    # =========================================================================
    # FERMENTED & HIGH-HISTAMINE FOODS
    # =========================================================================
    "wine_red": IngredientData(
        name="wine_red",
        display_name="Red Wine",
        category=IngredientCategory.BEVERAGES,
        name_variants=["red wine", "merlot", "cabernet", "pinot noir", "shiraz"],
        allergens=[
            AllergenMapping(AllergenType.SULFITES),
            AllergenMapping(AllergenType.EGGS, confidence=0.3, derivation=DerivationType.MAY_CONTAIN),
            AllergenMapping(AllergenType.FISH, confidence=0.2, derivation=DerivationType.MAY_CONTAIN),
        ],
        fodmap_level=FodmapLevel.LOW,  # One glass
        histamine_mg=30,
        histamine_level=CompoundLevel.MEDIUM,
        tyramine_mg=20,
        tyramine_level=CompoundLevel.MEDIUM,
        is_fermented=True,
    ),
    "wine_white": IngredientData(
        name="wine_white",
        display_name="White Wine",
        category=IngredientCategory.BEVERAGES,
        name_variants=["white wine", "chardonnay", "sauvignon blanc", "riesling"],
        allergens=[
            AllergenMapping(AllergenType.SULFITES),
        ],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=10,
        histamine_level=CompoundLevel.LOW,
        is_fermented=True,
    ),
    "beer": IngredientData(
        name="beer",
        display_name="Beer",
        category=IngredientCategory.BEVERAGES,
        name_variants=["lager", "ale", "stout", "pilsner", "ipa"],
        allergens=[
            AllergenMapping(AllergenType.GLUTEN_CEREALS),
            AllergenMapping(AllergenType.WHEAT, confidence=0.7),
        ],
        fodmap_level=FodmapLevel.HIGH,  # Fructans from wheat
        histamine_mg=10,
        histamine_level=CompoundLevel.LOW,
        tyramine_mg=20,
        tyramine_level=CompoundLevel.MEDIUM,
        is_fermented=True,
    ),
    "sauerkraut": IngredientData(
        name="sauerkraut",
        display_name="Sauerkraut",
        category=IngredientCategory.FERMENTED,
        name_variants=["fermented cabbage"],
        allergens=[],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=75,
        histamine_level=CompoundLevel.HIGH,
        tyramine_mg=30,
        tyramine_level=CompoundLevel.MEDIUM,
        is_fermented=True,
    ),
    "kimchi": IngredientData(
        name="kimchi",
        display_name="Kimchi",
        category=IngredientCategory.FERMENTED,
        name_variants=["korean kimchi"],
        allergens=[
            AllergenMapping(AllergenType.FISH, confidence=0.7, derivation=DerivationType.MAY_CONTAIN),
            AllergenMapping(AllergenType.SHELLFISH_CRUSTACEAN, confidence=0.5, derivation=DerivationType.MAY_CONTAIN),
        ],
        fodmap_level=FodmapLevel.LOW,  # Small serving
        histamine_mg=100,
        histamine_level=CompoundLevel.HIGH,
        tyramine_mg=40,
        tyramine_level=CompoundLevel.MEDIUM,
        is_fermented=True,
    ),
    "kombucha": IngredientData(
        name="kombucha",
        display_name="Kombucha",
        category=IngredientCategory.BEVERAGES,
        name_variants=["kombucha tea"],
        allergens=[],
        fodmap_level=FodmapLevel.HIGH,
        fodmap_types=["fructose"],
        histamine_mg=30,
        histamine_level=CompoundLevel.MEDIUM,
        is_fermented=True,
    ),
    "vinegar": IngredientData(
        name="vinegar",
        display_name="Vinegar",
        category=IngredientCategory.FERMENTED,
        name_variants=["apple cider vinegar", "white vinegar", "red wine vinegar", "balsamic"],
        allergens=[
            AllergenMapping(AllergenType.SULFITES, confidence=0.5),
        ],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=20,
        histamine_level=CompoundLevel.MEDIUM,
        is_fermented=True,
    ),
    "pickles": IngredientData(
        name="pickles",
        display_name="Pickles",
        category=IngredientCategory.FERMENTED,
        name_variants=["pickled cucumber", "dill pickles", "gherkins"],
        allergens=[],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=40,
        histamine_level=CompoundLevel.MEDIUM,
        is_fermented=True,
    ),

    # =========================================================================
    # CURED & PROCESSED MEATS
    # =========================================================================
    "bacon": IngredientData(
        name="bacon",
        display_name="Bacon",
        category=IngredientCategory.MEAT,
        name_variants=["pork bacon", "smoked bacon", "streaky bacon"],
        allergens=[
            AllergenMapping(AllergenType.SULFITES, confidence=0.3, derivation=DerivationType.MAY_CONTAIN),
        ],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=30,
        histamine_level=CompoundLevel.MEDIUM,
        tyramine_mg=50,
        tyramine_level=CompoundLevel.MEDIUM,
        is_aged=True,
    ),
    "salami": IngredientData(
        name="salami",
        display_name="Salami",
        category=IngredientCategory.MEAT,
        name_variants=["pepperoni", "soppressata", "genoa salami"],
        allergens=[
            AllergenMapping(AllergenType.MILK, confidence=0.3, derivation=DerivationType.MAY_CONTAIN),
        ],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=100,
        histamine_level=CompoundLevel.HIGH,
        tyramine_mg=200,
        tyramine_level=CompoundLevel.VERY_HIGH,
        is_fermented=True,
        is_aged=True,
    ),
    "prosciutto": IngredientData(
        name="prosciutto",
        display_name="Prosciutto",
        category=IngredientCategory.MEAT,
        name_variants=["parma ham", "serrano ham", "cured ham"],
        allergens=[],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=50,
        histamine_level=CompoundLevel.MEDIUM,
        tyramine_mg=100,
        tyramine_level=CompoundLevel.HIGH,
        is_aged=True,
    ),
    "hot_dog": IngredientData(
        name="hot_dog",
        display_name="Hot Dog",
        category=IngredientCategory.MEAT,
        name_variants=["frankfurter", "wiener", "sausage"],
        allergens=[
            AllergenMapping(AllergenType.MILK, confidence=0.5, derivation=DerivationType.MAY_CONTAIN),
            AllergenMapping(AllergenType.SOY, confidence=0.5, derivation=DerivationType.MAY_CONTAIN),
            AllergenMapping(AllergenType.WHEAT, confidence=0.3, derivation=DerivationType.MAY_CONTAIN),
        ],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=20,
        histamine_level=CompoundLevel.MEDIUM,
    ),

    # =========================================================================
    # NIGHTSHADES
    # =========================================================================
    "tomato": IngredientData(
        name="tomato",
        display_name="Tomato",
        category=IngredientCategory.VEGETABLES,
        name_variants=["tomatoes", "cherry tomato", "roma tomato", "tomato sauce", "tomato paste"],
        allergens=[],
        fodmap_level=FodmapLevel.LOW,  # Fresh, small serving
        histamine_mg=20,
        histamine_level=CompoundLevel.MEDIUM,
        is_nightshade=True,
        is_histamine_liberator=True,
        lectin_level=CompoundLevel.HIGH,  # Raw
    ),
    "potato": IngredientData(
        name="potato",
        display_name="Potato",
        category=IngredientCategory.VEGETABLES,
        name_variants=["potatoes", "white potato", "russet potato", "yukon gold"],
        allergens=[],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
        is_nightshade=True,
        lectin_level=CompoundLevel.MEDIUM,
    ),
    "bell_pepper": IngredientData(
        name="bell_pepper",
        display_name="Bell Pepper",
        category=IngredientCategory.VEGETABLES,
        name_variants=["capsicum", "sweet pepper", "red pepper", "green pepper", "yellow pepper"],
        allergens=[],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
        is_nightshade=True,
    ),
    "eggplant": IngredientData(
        name="eggplant",
        display_name="Eggplant",
        category=IngredientCategory.VEGETABLES,
        name_variants=["aubergine"],
        allergens=[],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=20,
        histamine_level=CompoundLevel.MEDIUM,
        is_nightshade=True,
        is_histamine_liberator=True,
    ),
    "chili_pepper": IngredientData(
        name="chili_pepper",
        display_name="Chili Pepper",
        category=IngredientCategory.VEGETABLES,
        name_variants=["hot pepper", "jalapeno", "cayenne", "habanero", "serrano"],
        allergens=[],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
        is_nightshade=True,
    ),

    # =========================================================================
    # HIGH-FODMAP VEGETABLES
    # =========================================================================
    "onion": IngredientData(
        name="onion",
        display_name="Onion",
        category=IngredientCategory.VEGETABLES,
        name_variants=["onions", "white onion", "yellow onion", "red onion", "shallot"],
        allergens=[],
        fodmap_level=FodmapLevel.HIGH,
        fodmap_types=["fructans"],
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
    ),
    "garlic": IngredientData(
        name="garlic",
        display_name="Garlic",
        category=IngredientCategory.VEGETABLES,
        name_variants=["garlic clove", "minced garlic", "garlic powder"],
        allergens=[],
        fodmap_level=FodmapLevel.HIGH,
        fodmap_types=["fructans"],
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
    ),
    "leek": IngredientData(
        name="leek",
        display_name="Leek",
        category=IngredientCategory.VEGETABLES,
        name_variants=["leeks"],
        allergens=[],
        fodmap_level=FodmapLevel.HIGH,
        fodmap_types=["fructans"],
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
    ),
    "asparagus": IngredientData(
        name="asparagus",
        display_name="Asparagus",
        category=IngredientCategory.VEGETABLES,
        name_variants=[],
        allergens=[],
        fodmap_level=FodmapLevel.HIGH,
        fodmap_types=["fructans"],
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
    ),
    "artichoke": IngredientData(
        name="artichoke",
        display_name="Artichoke",
        category=IngredientCategory.VEGETABLES,
        name_variants=["artichoke heart", "jerusalem artichoke"],
        allergens=[],
        fodmap_level=FodmapLevel.HIGH,
        fodmap_types=["fructans"],
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
    ),
    "mushroom": IngredientData(
        name="mushroom",
        display_name="Mushroom",
        category=IngredientCategory.VEGETABLES,
        name_variants=["mushrooms", "button mushroom", "shiitake", "portobello", "cremini"],
        allergens=[],
        fodmap_level=FodmapLevel.HIGH,
        fodmap_types=["polyols"],  # Mannitol
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
    ),
    "cauliflower": IngredientData(
        name="cauliflower",
        display_name="Cauliflower",
        category=IngredientCategory.VEGETABLES,
        name_variants=[],
        allergens=[],
        fodmap_level=FodmapLevel.HIGH,
        fodmap_types=["polyols"],  # Mannitol
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
    ),

    # =========================================================================
    # HIGH-OXALATE VEGETABLES
    # =========================================================================
    "spinach": IngredientData(
        name="spinach",
        display_name="Spinach",
        category=IngredientCategory.VEGETABLES,
        name_variants=["baby spinach", "fresh spinach"],
        allergens=[],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=30,
        histamine_level=CompoundLevel.MEDIUM,
        oxalate_mg=970,
        oxalate_level=CompoundLevel.VERY_HIGH,
        is_histamine_liberator=True,
    ),
    "beet": IngredientData(
        name="beet",
        display_name="Beet",
        category=IngredientCategory.VEGETABLES,
        name_variants=["beets", "beetroot", "red beet"],
        allergens=[],
        fodmap_level=FodmapLevel.MEDIUM,
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
        oxalate_mg=675,
        oxalate_level=CompoundLevel.VERY_HIGH,
    ),
    "swiss_chard": IngredientData(
        name="swiss_chard",
        display_name="Swiss Chard",
        category=IngredientCategory.VEGETABLES,
        name_variants=["chard", "rainbow chard"],
        allergens=[],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
        oxalate_mg=645,
        oxalate_level=CompoundLevel.VERY_HIGH,
    ),
    "rhubarb": IngredientData(
        name="rhubarb",
        display_name="Rhubarb",
        category=IngredientCategory.VEGETABLES,
        name_variants=[],
        allergens=[],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
        oxalate_mg=860,
        oxalate_level=CompoundLevel.VERY_HIGH,
    ),

    # =========================================================================
    # LEGUMES (HIGH FODMAP)
    # =========================================================================
    "chickpea": IngredientData(
        name="chickpea",
        display_name="Chickpea",
        category=IngredientCategory.LEGUMES,
        name_variants=["chickpeas", "garbanzo beans", "garbanzo"],
        allergens=[],
        fodmap_level=FodmapLevel.HIGH,
        fodmap_types=["galactans", "fructans"],
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
        lectin_level=CompoundLevel.HIGH,
    ),
    "lentil": IngredientData(
        name="lentil",
        display_name="Lentil",
        category=IngredientCategory.LEGUMES,
        name_variants=["lentils", "red lentil", "green lentil", "brown lentil"],
        allergens=[],
        fodmap_level=FodmapLevel.HIGH,
        fodmap_types=["galactans"],
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
        lectin_level=CompoundLevel.HIGH,
    ),
    "black_bean": IngredientData(
        name="black_bean",
        display_name="Black Bean",
        category=IngredientCategory.LEGUMES,
        name_variants=["black beans", "turtle beans"],
        allergens=[],
        fodmap_level=FodmapLevel.HIGH,
        fodmap_types=["galactans"],
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
        lectin_level=CompoundLevel.VERY_HIGH,  # Raw
    ),
    "kidney_bean": IngredientData(
        name="kidney_bean",
        display_name="Kidney Bean",
        category=IngredientCategory.LEGUMES,
        name_variants=["kidney beans", "red kidney beans"],
        allergens=[],
        fodmap_level=FodmapLevel.HIGH,
        fodmap_types=["galactans"],
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
        lectin_level=CompoundLevel.VERY_HIGH,
    ),

    # =========================================================================
    # HISTAMINE LIBERATORS (Not high histamine, but trigger release)
    # =========================================================================
    "citrus": IngredientData(
        name="citrus",
        display_name="Citrus",
        category=IngredientCategory.FRUITS,
        name_variants=["orange", "lemon", "lime", "grapefruit", "tangerine", "mandarin"],
        allergens=[],
        fodmap_level=FodmapLevel.LOW,  # Small serving
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
        is_histamine_liberator=True,
    ),
    "strawberry": IngredientData(
        name="strawberry",
        display_name="Strawberry",
        category=IngredientCategory.FRUITS,
        name_variants=["strawberries"],
        allergens=[],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
        is_histamine_liberator=True,
    ),
    "pineapple": IngredientData(
        name="pineapple",
        display_name="Pineapple",
        category=IngredientCategory.FRUITS,
        name_variants=[],
        allergens=[],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
        is_histamine_liberator=True,
    ),
    "papaya": IngredientData(
        name="papaya",
        display_name="Papaya",
        category=IngredientCategory.FRUITS,
        name_variants=["pawpaw"],
        allergens=[],
        fodmap_level=FodmapLevel.LOW,
        histamine_mg=0.5,
        histamine_level=CompoundLevel.NEGLIGIBLE,
        is_histamine_liberator=True,
    ),
    "chocolate": IngredientData(
        name="chocolate",
        display_name="Chocolate",
        category=IngredientCategory.OTHER,
        name_variants=["dark chocolate", "milk chocolate", "cocoa", "cacao"],
        allergens=[
            AllergenMapping(AllergenType.MILK, confidence=0.7, derivation=DerivationType.MAY_CONTAIN),
            AllergenMapping(AllergenType.SOY, confidence=0.6, derivation=DerivationType.MAY_CONTAIN),
            AllergenMapping(AllergenType.TREE_NUTS, confidence=0.5, derivation=DerivationType.MAY_CONTAIN),
        ],
        fodmap_level=FodmapLevel.LOW,  # Small serving
        histamine_mg=10,
        histamine_level=CompoundLevel.LOW,
        tyramine_mg=5,
        tyramine_level=CompoundLevel.LOW,
        is_histamine_liberator=True,
        oxalate_mg=117,
        oxalate_level=CompoundLevel.HIGH,
    ),

    # =========================================================================
    # ADDITIVES & SULFITES
    # =========================================================================
    "dried_fruit": IngredientData(
        name="dried_fruit",
        display_name="Dried Fruit",
        category=IngredientCategory.FRUITS,
        name_variants=["dried apricots", "raisins", "dried mango", "dried cranberries", "prunes"],
        allergens=[
            AllergenMapping(AllergenType.SULFITES),
        ],
        fodmap_level=FodmapLevel.HIGH,
        fodmap_types=["fructose"],
        histamine_mg=20,
        histamine_level=CompoundLevel.MEDIUM,
    ),
}


# =============================================================================
# LOOKUP FUNCTIONS
# =============================================================================


def get_ingredient(name: str) -> Optional[IngredientData]:
    """Get ingredient data by name (case-insensitive)"""
    name_lower = name.lower().strip()

    # Direct lookup
    if name_lower in INGREDIENT_DATABASE:
        return INGREDIENT_DATABASE[name_lower]

    # Search by name variants
    for key, ingredient in INGREDIENT_DATABASE.items():
        if name_lower in [v.lower() for v in ingredient.name_variants]:
            return ingredient

    return None


def get_allergens_for_ingredient(name: str) -> List[AllergenMapping]:
    """Get all allergen mappings for an ingredient"""
    ingredient = get_ingredient(name)
    if ingredient:
        return ingredient.allergens
    return []


def check_hidden_allergen(ingredient_text: str) -> List[AllergenType]:
    """
    Check if ingredient text contains hidden allergen keywords.

    Args:
        ingredient_text: Raw ingredient text to check

    Returns:
        List of detected allergen types
    """
    text_lower = ingredient_text.lower()
    found_allergens: Set[AllergenType] = set()

    for keyword, allergen in HIDDEN_ALLERGEN_KEYWORDS.items():
        if keyword in text_lower:
            found_allergens.add(allergen)

    return list(found_allergens)


def get_cross_reactive_allergens(allergen: AllergenType) -> List[AllergenType]:
    """Get allergens that may cross-react with the given allergen"""
    return CROSS_REACTIVITY.get(allergen, [])


def get_all_ingredients() -> Dict[str, IngredientData]:
    """Get the complete ingredient database"""
    return INGREDIENT_DATABASE


def get_ingredients_by_category(category: IngredientCategory) -> List[IngredientData]:
    """Get all ingredients in a category"""
    return [
        ing for ing in INGREDIENT_DATABASE.values()
        if ing.category == category
    ]


def get_high_histamine_ingredients() -> List[IngredientData]:
    """Get all ingredients with high or very high histamine"""
    return [
        ing for ing in INGREDIENT_DATABASE.values()
        if ing.histamine_level in [CompoundLevel.HIGH, CompoundLevel.VERY_HIGH]
    ]


def get_high_fodmap_ingredients() -> List[IngredientData]:
    """Get all high FODMAP ingredients"""
    return [
        ing for ing in INGREDIENT_DATABASE.values()
        if ing.fodmap_level == FodmapLevel.HIGH
    ]


def get_nightshade_ingredients() -> List[IngredientData]:
    """Get all nightshade ingredients"""
    return [
        ing for ing in INGREDIENT_DATABASE.values()
        if ing.is_nightshade
    ]


def get_histamine_liberators() -> List[IngredientData]:
    """Get all histamine-liberating ingredients"""
    return [
        ing for ing in INGREDIENT_DATABASE.values()
        if ing.is_histamine_liberator
    ]


def get_histamine_level(ingredient_name: str) -> Optional[CompoundLevel]:
    """
    Get histamine level for an ingredient.

    Args:
        ingredient_name: Name of the ingredient

    Returns:
        CompoundLevel or None if not found
    """
    ingredient = get_ingredient(ingredient_name)
    if ingredient:
        return ingredient.histamine_level
    return None


def get_tyramine_level(ingredient_name: str) -> Optional[CompoundLevel]:
    """
    Get tyramine level for an ingredient.

    Args:
        ingredient_name: Name of the ingredient

    Returns:
        CompoundLevel or None if not found
    """
    ingredient = get_ingredient(ingredient_name)
    if ingredient:
        return ingredient.tyramine_level
    return None
