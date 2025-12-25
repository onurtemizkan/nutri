"""
Sophisticated Ingredient-Based Micronutrient Estimation Service

This service provides intelligent micronutrient estimation by analyzing:
1. Ingredient text (parsed and matched to known nutrient profiles)
2. Product name (matched to food database)
3. Macronutrient profile (protein/fiber levels inform estimates)
4. Category baseline (fallback when other signals weak)

The approach uses position-weighted ingredient analysis since
ingredients are listed in descending order by weight.
"""

import re
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from app.data.food_database import (
    FoodCategory,
    FOOD_DATABASE,
    MICRONUTRIENT_CATEGORY_ESTIMATES,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# INGREDIENT MICRONUTRIENT PROFILES
# ==============================================================================
# Values are per typical portion contribution (not per 100g)
# These represent the micronutrient "boost" when an ingredient is present

INGREDIENT_MICRONUTRIENT_PROFILES: Dict[str, Dict[str, float]] = {
    # -------------------------------------------------------------------------
    # VITAMIN C RICH INGREDIENTS
    # -------------------------------------------------------------------------
    "orange": {"vitamin_c": 50, "folate": 30, "potassium": 180},
    "orange juice": {"vitamin_c": 60, "folate": 35, "potassium": 200},
    "lemon": {"vitamin_c": 30, "folate": 10},
    "lime": {"vitamin_c": 20},
    "grapefruit": {"vitamin_c": 40, "vitamin_a": 60},
    "strawberry": {"vitamin_c": 60, "folate": 25, "manganese": 0.4},
    "strawberries": {"vitamin_c": 60, "folate": 25},
    "kiwi": {"vitamin_c": 70, "vitamin_k": 30, "potassium": 200},
    "bell pepper": {"vitamin_c": 100, "vitamin_a": 150, "vitamin_b6": 0.2},
    "red pepper": {"vitamin_c": 120, "vitamin_a": 200},
    "broccoli": {"vitamin_c": 80, "vitamin_k": 100, "folate": 60, "potassium": 300},
    "tomato": {"vitamin_c": 15, "vitamin_a": 40, "potassium": 200},
    "tomatoes": {"vitamin_c": 15, "vitamin_a": 40, "potassium": 200},
    "papaya": {"vitamin_c": 90, "vitamin_a": 70, "folate": 40},
    "guava": {"vitamin_c": 125, "folate": 50, "potassium": 400},
    "acerola": {"vitamin_c": 800},  # Extremely high vitamin C
    # -------------------------------------------------------------------------
    # VITAMIN A RICH INGREDIENTS
    # -------------------------------------------------------------------------
    "carrot": {"vitamin_a": 500, "vitamin_k": 10, "potassium": 200},
    "carrots": {"vitamin_a": 500, "vitamin_k": 10, "potassium": 200},
    "sweet potato": {"vitamin_a": 700, "vitamin_c": 20, "potassium": 400, "fiber": 3},
    "spinach": {
        "vitamin_a": 400,
        "vitamin_k": 300,
        "iron": 2.0,
        "folate": 100,
        "magnesium": 60,
    },
    "kale": {"vitamin_a": 300, "vitamin_k": 400, "vitamin_c": 50, "calcium": 100},
    "collard": {"vitamin_a": 250, "vitamin_k": 350, "calcium": 150},
    "swiss chard": {"vitamin_a": 200, "vitamin_k": 300, "magnesium": 80},
    "butternut squash": {"vitamin_a": 500, "vitamin_c": 20, "potassium": 350},
    "pumpkin": {"vitamin_a": 400, "vitamin_c": 10, "potassium": 300},
    "cantaloupe": {"vitamin_a": 200, "vitamin_c": 40, "potassium": 250},
    "apricot": {"vitamin_a": 100, "vitamin_c": 10, "potassium": 250},
    "mango": {"vitamin_a": 60, "vitamin_c": 45, "folate": 40},
    # -------------------------------------------------------------------------
    # B12 RICH INGREDIENTS (animal products)
    # -------------------------------------------------------------------------
    "beef": {
        "vitamin_b12": 2.5,
        "zinc": 4.5,
        "iron": 2.5,
        "niacin": 5,
        "vitamin_b6": 0.4,
    },
    "chicken": {"vitamin_b12": 0.3, "vitamin_b6": 0.5, "niacin": 10, "zinc": 1.5},
    "turkey": {"vitamin_b12": 0.4, "vitamin_b6": 0.5, "niacin": 8, "zinc": 2.0},
    "pork": {"vitamin_b12": 0.6, "thiamin": 0.8, "niacin": 5, "zinc": 2.5},
    "lamb": {"vitamin_b12": 2.5, "zinc": 4.0, "iron": 1.8, "niacin": 6},
    "liver": {
        "vitamin_b12": 60,
        "vitamin_a": 6000,
        "iron": 6,
        "folate": 200,
        "riboflavin": 2.5,
    },
    "salmon": {"vitamin_b12": 4.0, "vitamin_d": 15, "niacin": 8, "omega3": 2000},
    "tuna": {"vitamin_b12": 3.0, "vitamin_d": 2, "niacin": 10, "vitamin_b6": 0.5},
    "sardine": {"vitamin_b12": 8.0, "vitamin_d": 5, "calcium": 350, "omega3": 1500},
    "mackerel": {"vitamin_b12": 8.0, "vitamin_d": 15, "niacin": 9, "omega3": 2500},
    "trout": {"vitamin_b12": 5.0, "vitamin_d": 15, "niacin": 5},
    "cod": {"vitamin_b12": 1.0, "vitamin_d": 1, "iodine": 100},
    "shrimp": {"vitamin_b12": 1.5, "zinc": 1.5, "iodine": 35, "selenium": 40},
    "crab": {"vitamin_b12": 6.0, "zinc": 6.0, "copper": 0.6},
    "oyster": {"vitamin_b12": 16, "zinc": 75, "iron": 5, "copper": 4},
    "clam": {"vitamin_b12": 84, "iron": 24, "zinc": 2.5},  # Clams are B12 powerhouses
    "egg": {
        "vitamin_b12": 0.6,
        "vitamin_d": 1,
        "riboflavin": 0.25,
        "selenium": 15,
        "choline": 150,
    },
    "eggs": {"vitamin_b12": 0.6, "vitamin_d": 1, "riboflavin": 0.25, "selenium": 15},
    # -------------------------------------------------------------------------
    # IRON RICH INGREDIENTS
    # -------------------------------------------------------------------------
    "lentil": {"iron": 3.5, "folate": 180, "potassium": 350, "fiber": 8},
    "lentils": {"iron": 3.5, "folate": 180, "potassium": 350, "fiber": 8},
    "chickpea": {"iron": 2.5, "folate": 170, "magnesium": 50, "zinc": 1.5, "fiber": 7},
    "chickpeas": {"iron": 2.5, "folate": 170, "magnesium": 50, "zinc": 1.5},
    "black bean": {"iron": 2.0, "folate": 130, "magnesium": 60, "potassium": 300},
    "kidney bean": {"iron": 2.5, "folate": 115, "potassium": 350},
    "white bean": {"iron": 3.5, "folate": 100, "potassium": 500, "magnesium": 65},
    "navy bean": {"iron": 2.5, "folate": 130, "magnesium": 50},
    "pinto bean": {"iron": 2.0, "folate": 150, "potassium": 400},
    "soybean": {"iron": 5.0, "calcium": 100, "magnesium": 90, "potassium": 500},
    "tofu": {"iron": 2.5, "calcium": 200, "magnesium": 35, "zinc": 1.0},
    "tempeh": {
        "iron": 2.5,
        "calcium": 90,
        "magnesium": 80,
        "zinc": 1.5,
        "vitamin_b12": 0.1,
    },
    "edamame": {"iron": 2.0, "folate": 120, "vitamin_k": 25, "magnesium": 50},
    "quinoa": {"iron": 2.8, "magnesium": 65, "folate": 40, "zinc": 1.5, "fiber": 3},
    # -------------------------------------------------------------------------
    # CALCIUM RICH INGREDIENTS
    # -------------------------------------------------------------------------
    "milk": {
        "calcium": 300,
        "vitamin_d": 2.5,
        "riboflavin": 0.4,
        "vitamin_b12": 1.0,
        "phosphorus": 200,
    },
    "cheese": {"calcium": 200, "vitamin_b12": 0.8, "phosphorus": 150, "zinc": 1.5},
    "cheddar": {"calcium": 300, "vitamin_b12": 0.9, "phosphorus": 200, "zinc": 2.0},
    "mozzarella": {"calcium": 220, "vitamin_b12": 0.7, "phosphorus": 150},
    "parmesan": {"calcium": 350, "vitamin_b12": 1.2, "phosphorus": 250, "zinc": 2.5},
    "yogurt": {"calcium": 250, "vitamin_b12": 0.9, "riboflavin": 0.3, "potassium": 350},
    "greek yogurt": {"calcium": 200, "vitamin_b12": 1.0, "riboflavin": 0.25},
    "quark": {"calcium": 90, "vitamin_b12": 0.8, "riboflavin": 0.2, "phosphorus": 150},
    "fromage frais": {
        "calcium": 85,
        "vitamin_b12": 0.6,
        "riboflavin": 0.2,
    },  # French quark
    "skyr": {
        "calcium": 150,
        "vitamin_b12": 1.0,
        "riboflavin": 0.25,
        "phosphorus": 180,
    },  # Icelandic
    "protein yogurt": {"calcium": 180, "vitamin_b12": 1.0, "riboflavin": 0.25},
    "cottage cheese": {"calcium": 80, "vitamin_b12": 0.6, "riboflavin": 0.2},
    "kefir": {"calcium": 250, "vitamin_b12": 0.8, "vitamin_k": 30, "riboflavin": 0.3},
    "cream": {"vitamin_a": 400, "calcium": 65, "vitamin_b12": 0.2, "riboflavin": 0.1},
    "fortified plant milk": {"calcium": 300, "vitamin_d": 2.5, "vitamin_b12": 1.0},
    "almond milk": {
        "calcium": 300,
        "vitamin_d": 2.5,
        "vitamin_e": 7,
    },  # Usually fortified
    "soy milk": {
        "calcium": 300,
        "vitamin_d": 2.5,
        "vitamin_b12": 1.0,
    },  # Usually fortified
    # -------------------------------------------------------------------------
    # VITAMIN D SOURCES
    # -------------------------------------------------------------------------
    "mushroom": {"vitamin_d": 7, "riboflavin": 0.3, "niacin": 3, "selenium": 10},
    "mushrooms": {"vitamin_d": 7, "riboflavin": 0.3, "niacin": 3},
    "shiitake": {"vitamin_d": 15, "riboflavin": 0.2, "niacin": 4},
    "fortified": {
        "vitamin_d": 2.5,
        "vitamin_b12": 0.6,
        "calcium": 100,
    },  # General fortification
    # -------------------------------------------------------------------------
    # WHOLE GRAIN / FIBER RICH
    # -------------------------------------------------------------------------
    "oat": {"magnesium": 50, "thiamin": 0.2, "iron": 2.0, "zinc": 2.0, "fiber": 4},
    "oats": {"magnesium": 50, "thiamin": 0.2, "iron": 2.0, "zinc": 2.0, "fiber": 4},
    "whole wheat": {"magnesium": 40, "zinc": 1.5, "iron": 1.5, "fiber": 3},
    "whole grain": {"magnesium": 35, "zinc": 1.0, "iron": 1.5, "fiber": 3},
    "brown rice": {"magnesium": 45, "thiamin": 0.2, "niacin": 2.5, "fiber": 2},
    "barley": {"magnesium": 35, "thiamin": 0.15, "niacin": 3, "fiber": 6},
    "buckwheat": {"magnesium": 85, "zinc": 1.0, "iron": 1.5, "fiber": 3},
    "millet": {"magnesium": 50, "thiamin": 0.2, "niacin": 2, "iron": 1.5},
    "flaxseed": {"magnesium": 100, "thiamin": 0.5, "omega3": 2000, "fiber": 8},
    "chia": {"magnesium": 95, "calcium": 180, "iron": 2.0, "fiber": 10, "omega3": 5000},
    "hemp": {"magnesium": 200, "zinc": 3.0, "iron": 2.5, "omega3": 1000},
    # -------------------------------------------------------------------------
    # NUTS AND SEEDS (Vitamin E, Magnesium rich)
    # -------------------------------------------------------------------------
    "almond": {
        "vitamin_e": 7.5,
        "magnesium": 75,
        "calcium": 75,
        "riboflavin": 0.3,
        "fiber": 3,
    },
    "almonds": {"vitamin_e": 7.5, "magnesium": 75, "calcium": 75, "riboflavin": 0.3},
    "walnut": {"omega3": 2500, "magnesium": 45, "vitamin_e": 2, "copper": 0.5},
    "walnuts": {"omega3": 2500, "magnesium": 45, "vitamin_e": 2},
    "cashew": {"magnesium": 85, "zinc": 1.6, "iron": 2.0, "copper": 0.6},
    "cashews": {"magnesium": 85, "zinc": 1.6, "iron": 2.0},
    "peanut": {"niacin": 4, "folate": 60, "magnesium": 50, "vitamin_e": 2},
    "peanuts": {"niacin": 4, "folate": 60, "magnesium": 50},
    "peanut butter": {"niacin": 4, "folate": 30, "magnesium": 50, "vitamin_e": 3},
    "pistachio": {"vitamin_b6": 0.5, "thiamin": 0.25, "potassium": 300, "fiber": 3},
    "sunflower seed": {"vitamin_e": 10, "magnesium": 100, "selenium": 20, "folate": 65},
    "pumpkin seed": {"magnesium": 150, "zinc": 2.5, "iron": 2.5, "phosphorus": 350},
    "sesame": {"calcium": 280, "magnesium": 100, "zinc": 2.5, "iron": 4},
    # -------------------------------------------------------------------------
    # FORTIFICATION INDICATORS
    # -------------------------------------------------------------------------
    "enriched": {
        "thiamin": 0.3,
        "riboflavin": 0.25,
        "niacin": 3,
        "folate": 140,
        "iron": 3,
    },
    "fortified with": {"vitamin_d": 2.5, "vitamin_b12": 1.0, "calcium": 100},
    "added vitamins": {"vitamin_d": 2, "vitamin_b12": 0.5, "vitamin_a": 100},
    "vitamin d added": {"vitamin_d": 2.5},
    "calcium added": {"calcium": 200},
    # -------------------------------------------------------------------------
    # HERBS AND SPICES (concentrated nutrients)
    # -------------------------------------------------------------------------
    "parsley": {"vitamin_k": 500, "vitamin_c": 40, "vitamin_a": 250, "folate": 50},
    "basil": {"vitamin_k": 100, "vitamin_a": 100, "iron": 1.5},
    "cilantro": {"vitamin_k": 80, "vitamin_a": 70, "vitamin_c": 10},
    "thyme": {"vitamin_k": 50, "iron": 5, "vitamin_c": 15},
    "oregano": {"vitamin_k": 100, "iron": 4, "calcium": 150},
    "turmeric": {"iron": 2, "manganese": 1.5},  # Also anti-inflammatory
    "ginger": {"magnesium": 12, "potassium": 100},
    # -------------------------------------------------------------------------
    # COMMON PRODUCE
    # -------------------------------------------------------------------------
    "banana": {"potassium": 400, "vitamin_b6": 0.4, "vitamin_c": 10, "magnesium": 30},
    "avocado": {
        "potassium": 500,
        "vitamin_k": 20,
        "folate": 80,
        "vitamin_e": 2,
        "fiber": 7,
    },
    "potato": {"potassium": 600, "vitamin_c": 20, "vitamin_b6": 0.3, "fiber": 2},
    "apple": {"vitamin_c": 5, "potassium": 100, "fiber": 3},
    "blueberry": {"vitamin_c": 10, "vitamin_k": 20, "manganese": 0.4},
    "blueberries": {"vitamin_c": 10, "vitamin_k": 20, "manganese": 0.4},
    "raspberry": {"vitamin_c": 25, "manganese": 0.7, "fiber": 6},
    "blackberry": {"vitamin_c": 20, "vitamin_k": 20, "manganese": 0.9, "fiber": 5},
    "cranberry": {"vitamin_c": 15, "vitamin_e": 1},
    "pomegranate": {"vitamin_c": 10, "vitamin_k": 15, "folate": 40, "potassium": 200},
    "pineapple": {
        "vitamin_c": 48,
        "manganese": 0.9,
        "vitamin_b6": 0.1,
        "thiamin": 0.08,
    },
    "grape": {"vitamin_k": 15, "vitamin_c": 5, "potassium": 190},
    "grapes": {"vitamin_k": 15, "vitamin_c": 5, "potassium": 190},
    "pear": {"vitamin_c": 5, "vitamin_k": 5, "potassium": 120, "fiber": 4},
    "peach": {"vitamin_c": 8, "vitamin_a": 25, "potassium": 190},
    "plum": {"vitamin_c": 10, "vitamin_k": 6, "potassium": 150},
    "cherry": {"vitamin_c": 10, "potassium": 175, "vitamin_a": 15},
    "fig": {"potassium": 230, "calcium": 35, "magnesium": 17, "fiber": 3},
    "date": {"potassium": 650, "magnesium": 40, "iron": 1, "fiber": 7},
    # -------------------------------------------------------------------------
    # VEGETABLES
    # -------------------------------------------------------------------------
    "asparagus": {"folate": 135, "vitamin_k": 45, "vitamin_a": 40, "vitamin_c": 8},
    "brussels sprout": {"vitamin_k": 150, "vitamin_c": 75, "folate": 50},
    "cabbage": {"vitamin_k": 60, "vitamin_c": 35, "folate": 40},
    "cauliflower": {"vitamin_c": 50, "vitamin_k": 15, "folate": 55},
    "celery": {"vitamin_k": 30, "potassium": 250, "folate": 35},
    "cucumber": {"vitamin_k": 15, "potassium": 150},
    "eggplant": {"fiber": 2.5, "potassium": 180, "folate": 15},
    "garlic": {"vitamin_c": 5, "vitamin_b6": 0.4, "manganese": 0.3, "selenium": 3},
    "green bean": {"vitamin_k": 15, "vitamin_c": 12, "folate": 30},
    "leek": {"vitamin_k": 45, "folate": 60, "vitamin_c": 10},
    "lettuce": {"vitamin_k": 50, "vitamin_a": 150, "folate": 40},
    "onion": {"vitamin_c": 8, "vitamin_b6": 0.1, "folate": 20},
    "pea": {"vitamin_k": 25, "vitamin_c": 40, "thiamin": 0.25, "folate": 65},
    "peas": {"vitamin_k": 25, "vitamin_c": 40, "thiamin": 0.25, "folate": 65},
    "radish": {"vitamin_c": 15, "folate": 25},
    "zucchini": {"vitamin_c": 15, "potassium": 260, "folate": 25},
    "squash": {"vitamin_a": 200, "vitamin_c": 15, "potassium": 350},
    "corn": {"thiamin": 0.15, "folate": 40, "vitamin_c": 7, "magnesium": 35},
    "beet": {"folate": 110, "potassium": 300, "manganese": 0.3},
}

# Fortification keywords that indicate added vitamins/minerals
FORTIFICATION_KEYWORDS = [
    "fortified",
    "enriched",
    "added vitamins",
    "vitamin d added",
    "calcium added",
    "with vitamins",
    "vitamin enriched",
]

# ==============================================================================
# INGREDIENT SYNONYMS AND ALIASES
# Maps variations to canonical ingredient names for better matching
# ==============================================================================
INGREDIENT_SYNONYMS: Dict[str, str] = {
    # Dairy variations
    "whole milk": "milk",
    "skim milk": "milk",
    "skimmed milk": "milk",
    "semi-skimmed milk": "milk",
    "low-fat milk": "milk",
    "fat-free milk": "milk",
    "pasteurized milk": "milk",
    "reconstituted milk": "milk",
    "milk powder": "milk",
    "dried milk": "milk",
    "powdered milk": "milk",
    "nonfat dry milk": "milk",
    "evaporated milk": "milk",
    "condensed milk": "milk",
    "buttermilk": "milk",
    "heavy cream": "cream",
    "heavy whipping cream": "cream",
    "double cream": "cream",
    "single cream": "cream",
    "light cream": "cream",
    "whipping cream": "cream",
    "sour cream": "cream",
    "creme fraiche": "cream",
    "crème fraîche": "cream",
    "clotted cream": "cream",
    "plain yogurt": "yogurt",
    "natural yogurt": "yogurt",
    "low-fat yogurt": "yogurt",
    "nonfat yogurt": "yogurt",
    "whole milk yogurt": "yogurt",
    "strained yogurt": "greek yogurt",
    "greek-style yogurt": "greek yogurt",
    "icelandic yogurt": "skyr",
    "quark cheese": "quark",
    "curd cheese": "quark",
    "farmer cheese": "cottage cheese",
    "pot cheese": "cottage cheese",
    "ricotta cheese": "cottage cheese",  # Similar profile
    "cream cheese": "cheese",
    "soft cheese": "cheese",
    "hard cheese": "cheese",
    "aged cheese": "cheese",
    # Protein variations
    "chicken breast": "chicken",
    "chicken thigh": "chicken",
    "chicken leg": "chicken",
    "chicken wing": "chicken",
    "ground chicken": "chicken",
    "turkey breast": "turkey",
    "ground turkey": "turkey",
    "beef steak": "beef",
    "ground beef": "beef",
    "minced beef": "beef",
    "beef mince": "beef",
    "lean beef": "beef",
    "pork loin": "pork",
    "ground pork": "pork",
    "pork shoulder": "pork",
    "lamb chop": "lamb",
    "ground lamb": "lamb",
    "lamb mince": "lamb",
    # Fish variations
    "wild salmon": "salmon",
    "atlantic salmon": "salmon",
    "pacific salmon": "salmon",
    "smoked salmon": "salmon",
    "canned salmon": "salmon",
    "salmon fillet": "salmon",
    "yellowfin tuna": "tuna",
    "albacore tuna": "tuna",
    "skipjack tuna": "tuna",
    "canned tuna": "tuna",
    "tuna steak": "tuna",
    "atlantic cod": "cod",
    "pacific cod": "cod",
    "fresh cod": "cod",
    # Egg variations
    "whole egg": "egg",
    "whole eggs": "eggs",
    "egg white": "egg",
    "egg whites": "eggs",
    "egg yolk": "egg",
    "egg yolks": "eggs",
    "pasteurized egg": "egg",
    "liquid egg": "egg",
    "dried egg": "egg",
    "egg powder": "egg",
    # Grain variations
    "rolled oats": "oats",
    "quick oats": "oats",
    "steel-cut oats": "oats",
    "instant oats": "oats",
    "oat flour": "oats",
    "oat bran": "oats",
    "whole wheat flour": "whole wheat",
    "whole grain wheat": "whole wheat",
    "wheat flour": "whole wheat",
    "bread flour": "whole wheat",
    "brown rice flour": "brown rice",
    "long grain rice": "brown rice",
    "basmati rice": "brown rice",
    "jasmine rice": "brown rice",
    "white rice": "brown rice",  # Less nutrients but similar category
    # Nut variations
    "raw almonds": "almond",
    "roasted almonds": "almond",
    "sliced almonds": "almond",
    "almond flour": "almond",
    "almond butter": "almond",
    "raw walnuts": "walnut",
    "walnut pieces": "walnut",
    "walnut halves": "walnut",
    "raw cashews": "cashew",
    "roasted cashews": "cashew",
    "cashew butter": "cashew",
    "raw peanuts": "peanut",
    "roasted peanuts": "peanut",
    "dry roasted peanuts": "peanut",
    "ground flaxseed": "flaxseed",
    "flax meal": "flaxseed",
    "milled flaxseed": "flaxseed",
    "chia seeds": "chia",
    "hemp seeds": "hemp",
    "hemp hearts": "hemp",
    # Legume variations
    "dried lentils": "lentils",
    "red lentils": "lentils",
    "green lentils": "lentils",
    "brown lentils": "lentils",
    "split lentils": "lentils",
    "canned chickpeas": "chickpeas",
    "garbanzo beans": "chickpeas",
    "canned black beans": "black bean",
    "dried black beans": "black bean",
    "canned kidney beans": "kidney bean",
    "red kidney beans": "kidney bean",
    "white kidney beans": "white bean",
    "cannellini beans": "white bean",
    "great northern beans": "white bean",
    "firm tofu": "tofu",
    "silken tofu": "tofu",
    "extra firm tofu": "tofu",
    "pressed tofu": "tofu",
    # Vegetable variations
    "fresh spinach": "spinach",
    "baby spinach": "spinach",
    "frozen spinach": "spinach",
    "fresh kale": "kale",
    "baby kale": "kale",
    "curly kale": "kale",
    "lacinato kale": "kale",
    "romaine lettuce": "lettuce",
    "iceberg lettuce": "lettuce",
    "butter lettuce": "lettuce",
    "red cabbage": "cabbage",
    "green cabbage": "cabbage",
    "napa cabbage": "cabbage",
    "savoy cabbage": "cabbage",
    "fresh broccoli": "broccoli",
    "broccoli florets": "broccoli",
    "frozen broccoli": "broccoli",
    "sweet potato": "sweet potato",
    "yam": "sweet potato",
    "red bell pepper": "bell pepper",
    "green bell pepper": "bell pepper",
    "yellow bell pepper": "bell pepper",
    # Fruit variations
    "fresh strawberries": "strawberry",
    "frozen strawberries": "strawberry",
    "fresh blueberries": "blueberry",
    "frozen blueberries": "blueberry",
    "wild blueberries": "blueberry",
    "fresh raspberries": "raspberry",
    "frozen raspberries": "raspberry",
    "navel orange": "orange",
    "blood orange": "orange",
    "mandarin orange": "orange",
    "valencia orange": "orange",
    "fresh mango": "mango",
    "mango puree": "mango",
    "mango pulp": "mango",
    "dried mango": "mango",
    "passion fruit": "mango",  # Similar tropical profile
    "papaya pulp": "papaya",
    "fresh papaya": "papaya",
    "fresh pineapple": "pineapple",
    "pineapple chunks": "pineapple",
    "crushed pineapple": "pineapple",
}

# Additional high-value ingredients to add to the main profiles
ADDITIONAL_INGREDIENT_PROFILES: Dict[str, Dict[str, float]] = {
    # More dairy products
    "mascarpone": {"calcium": 60, "vitamin_a": 150, "vitamin_b12": 0.4},
    "ricotta": {"calcium": 200, "vitamin_b12": 0.3, "phosphorus": 150},
    "paneer": {"calcium": 200, "vitamin_b12": 0.4, "phosphorus": 130},
    "ghee": {"vitamin_a": 400, "vitamin_e": 2.5, "vitamin_k": 8},
    "cream cheese": {"calcium": 80, "vitamin_a": 300, "vitamin_b12": 0.3},
    "sour cream": {
        "calcium": 100,
        "vitamin_a": 120,
        "vitamin_b12": 0.3,
        "riboflavin": 0.1,
    },
    "buttermilk": {
        "calcium": 280,
        "vitamin_b12": 0.5,
        "riboflavin": 0.15,
        "potassium": 150,
    },
    "evaporated milk": {"calcium": 260, "vitamin_d": 2.5, "vitamin_b12": 0.4},
    "condensed milk": {"calcium": 280, "vitamin_b12": 0.3, "riboflavin": 0.3},
    "whey protein": {"calcium": 150, "vitamin_b12": 0.5, "phosphorus": 100},
    "casein": {"calcium": 500, "phosphorus": 300},
    # More grains
    "spelt": {"magnesium": 45, "zinc": 2.5, "iron": 2.0, "thiamin": 0.25, "fiber": 5},
    "kamut": {"magnesium": 50, "zinc": 2.5, "iron": 2.5, "niacin": 3, "fiber": 4},
    "teff": {"calcium": 180, "iron": 7.5, "magnesium": 180, "zinc": 3.5, "fiber": 8},
    "amaranth": {
        "calcium": 160,
        "iron": 7.5,
        "magnesium": 250,
        "phosphorus": 550,
        "fiber": 7,
    },
    "farro": {"magnesium": 40, "zinc": 2.5, "iron": 2.5, "niacin": 3, "fiber": 5},
    "sorghum": {
        "magnesium": 35,
        "iron": 4.5,
        "phosphorus": 300,
        "thiamin": 0.25,
        "fiber": 6,
    },
    "freekeh": {"magnesium": 35, "zinc": 2.0, "iron": 2.0, "fiber": 8},
    # More proteins
    "seitan": {"iron": 5.0, "zinc": 1.5, "niacin": 3, "vitamin_b6": 0.2},
    "textured vegetable protein": {
        "iron": 5.0,
        "zinc": 2.0,
        "magnesium": 80,
        "folate": 150,
    },
    "tvp": {"iron": 5.0, "zinc": 2.0, "magnesium": 80, "folate": 150},
    "soy protein": {"iron": 5.0, "calcium": 100, "magnesium": 50, "zinc": 2.0},
    "pea protein": {"iron": 3.0, "zinc": 1.5, "magnesium": 40},
    "duck": {"vitamin_b12": 0.4, "iron": 2.5, "zinc": 2.0, "niacin": 5},
    "goose": {"vitamin_b12": 0.4, "iron": 2.5, "zinc": 2.5, "niacin": 4},
    "venison": {"vitamin_b12": 4.0, "iron": 3.5, "zinc": 3.5, "niacin": 7},
    "bison": {"vitamin_b12": 2.5, "iron": 3.0, "zinc": 4.5, "niacin": 6},
    "rabbit": {"vitamin_b12": 7.0, "iron": 2.0, "zinc": 2.0, "niacin": 6},
    # More seafood
    "scallop": {"vitamin_b12": 1.5, "zinc": 1.0, "phosphorus": 200, "magnesium": 25},
    "lobster": {"vitamin_b12": 1.5, "zinc": 3.5, "phosphorus": 200, "copper": 1.5},
    "squid": {"vitamin_b12": 1.5, "zinc": 1.5, "phosphorus": 200, "copper": 2.0},
    "octopus": {"vitamin_b12": 20, "iron": 5.0, "zinc": 1.5, "copper": 0.5},
    "mussels": {"vitamin_b12": 12, "iron": 4.0, "zinc": 1.5, "manganese": 3.5},
    "tilapia": {"vitamin_b12": 1.5, "niacin": 4, "vitamin_d": 3, "phosphorus": 200},
    "halibut": {"vitamin_b12": 1.2, "vitamin_d": 5, "niacin": 6, "magnesium": 25},
    "sole": {"vitamin_b12": 1.0, "vitamin_d": 3, "niacin": 2, "phosphorus": 200},
    "flounder": {"vitamin_b12": 1.5, "vitamin_d": 3, "niacin": 2, "magnesium": 20},
    "anchovy": {"vitamin_b12": 6, "calcium": 230, "iron": 3, "niacin": 14},
    "herring": {"vitamin_b12": 13, "vitamin_d": 20, "niacin": 3, "omega3": 2000},
    # More vegetables
    "artichoke": {
        "folate": 70,
        "vitamin_c": 12,
        "vitamin_k": 15,
        "magnesium": 60,
        "fiber": 5,
    },
    "fennel": {"vitamin_c": 12, "potassium": 400, "folate": 25, "fiber": 3},
    "kohlrabi": {"vitamin_c": 60, "potassium": 350, "vitamin_b6": 0.15, "fiber": 3.5},
    "turnip": {"vitamin_c": 20, "potassium": 230, "folate": 15, "fiber": 2},
    "rutabaga": {"vitamin_c": 25, "potassium": 300, "magnesium": 20, "fiber": 2.5},
    "parsnip": {"vitamin_c": 17, "potassium": 375, "folate": 65, "fiber": 4.5},
    "daikon": {"vitamin_c": 22, "potassium": 230, "folate": 25},
    "jicama": {"vitamin_c": 20, "potassium": 150, "fiber": 5},
    "water chestnut": {"potassium": 580, "vitamin_b6": 0.3, "copper": 0.3},
    "bamboo shoots": {"potassium": 530, "vitamin_b6": 0.2, "zinc": 1.0},
    # Regional/ethnic ingredients
    "tahini": {
        "calcium": 430,
        "iron": 9,
        "magnesium": 95,
        "zinc": 5,
        "phosphorus": 750,
    },
    "miso": {"vitamin_k": 30, "zinc": 2.5, "manganese": 0.9, "sodium": 3700},
    "natto": {"vitamin_k": 1000, "iron": 8, "zinc": 3, "manganese": 1.5},
    "kimchi": {"vitamin_k": 40, "vitamin_c": 18, "folate": 50, "vitamin_b6": 0.2},
    "sauerkraut": {"vitamin_c": 15, "vitamin_k": 13, "folate": 20, "iron": 1.5},
    "nutritional yeast": {
        "vitamin_b12": 8,
        "folate": 300,
        "thiamin": 10,
        "riboflavin": 10,
        "niacin": 50,
    },
    # Common processed ingredients (minimal but some nutrients)
    "cocoa powder": {
        "iron": 13,
        "magnesium": 500,
        "zinc": 7,
        "phosphorus": 730,
        "potassium": 1500,
    },
    "dark chocolate": {
        "iron": 8,
        "magnesium": 175,
        "zinc": 3,
        "copper": 1.8,
        "fiber": 10,
    },
    "honey": {"potassium": 50, "vitamin_c": 0.5},
    "maple syrup": {"manganese": 2.5, "zinc": 1.5, "potassium": 200, "calcium": 100},
    "molasses": {
        "iron": 4.7,
        "calcium": 200,
        "magnesium": 240,
        "potassium": 1460,
        "vitamin_b6": 0.7,
    },
    # More seeds
    "poppy seeds": {
        "calcium": 1440,
        "iron": 9.5,
        "magnesium": 330,
        "zinc": 8,
        "phosphorus": 870,
    },
    "caraway seeds": {"calcium": 690, "iron": 16, "magnesium": 260, "zinc": 5.5},
    "fennel seeds": {"calcium": 1200, "iron": 18, "magnesium": 385, "manganese": 6.5},
    "cumin seeds": {
        "iron": 66,
        "calcium": 930,
        "magnesium": 365,
        "zinc": 4.8,
        "potassium": 1790,
    },
    "coriander seeds": {
        "calcium": 710,
        "iron": 16,
        "magnesium": 330,
        "potassium": 1270,
    },
}

# Merge additional profiles into main dictionary
INGREDIENT_MICRONUTRIENT_PROFILES.update(ADDITIONAL_INGREDIENT_PROFILES)


@dataclass
class EstimationSignal:
    """A signal contributing to micronutrient estimation"""

    source: str  # "ingredient", "food_match", "macros", "category"
    confidence: float  # 0.0 to 1.0
    estimates: Dict[str, float]  # micronutrient -> value


@dataclass
class ParsedIngredient:
    """A parsed ingredient with weight information"""

    name: str
    position_weight: float
    percentage: Optional[float] = None  # If percentage was specified
    is_minor: bool = False  # True if in "contains less than X%" section


def parse_ingredients(ingredients_text: Optional[str]) -> List[Tuple[str, float]]:
    """
    Parse ingredients text and return list of (ingredient, weight) tuples.

    Ingredients are typically listed in descending order by weight.
    We assign position-based weights: first ingredient gets weight 1.0,
    subsequent ingredients get decreasing weights.

    Improvements over basic parsing:
    - Extracts and uses percentages when available
    - Handles "contains less than X%" sections
    - Preserves useful parenthetical content (like "milk" in "quark (milk)")
    - Better handling of compound ingredients

    Returns:
        List of (ingredient_name, position_weight) tuples
    """
    if not ingredients_text:
        return []

    text = ingredients_text.lower()

    # First, extract any explicit percentages and store them
    # e.g., "milk (30%)" -> we want to use 0.30 as weight
    percentage_map: Dict[str, float] = {}
    percentage_pattern = r"(\w[\w\s]*?)\s*\(?\s*(\d+(?:\.\d+)?)\s*%\s*\)?"
    for match in re.finditer(percentage_pattern, text):
        ingredient_name = match.group(1).strip()
        percentage = float(match.group(2)) / 100.0
        if ingredient_name and len(ingredient_name) > 1:
            percentage_map[ingredient_name] = percentage

    # Check for "contains less than X% of" or "contains 2% or less of"
    minor_ingredients_section = ""
    minor_pattern = r"contains?\s+(?:less\s+than\s+)?(\d+)%?\s*(?:or\s+less\s+)?(?:of)?[:\s]+(.+?)(?:\.|$)"
    minor_match = re.search(minor_pattern, text)
    if minor_match:
        minor_percentage = float(minor_match.group(1)) / 100.0
        minor_ingredients_section = minor_match.group(2)

    # Clean the text but preserve useful parenthetical content
    # Extract parenthetical content that might be useful (like "milk" in "quark (milk)")
    parenthetical_additions = []
    paren_pattern = r"\(([^)]+)\)"
    for match in re.finditer(paren_pattern, text):
        content = match.group(1).lower()
        # Keep if it looks like an ingredient (not a percentage or note)
        if not re.match(r"^[\d.]+%?$", content) and len(content) < 30:
            # Check if it's a known ingredient
            if any(
                ing in content for ing in ["milk", "wheat", "soy", "egg", "fish", "nut"]
            ):
                parenthetical_additions.append(content)

    # Remove parenthetical notes but track what we removed
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)

    # Remove common non-ingredient phrases
    text = re.sub(r"\b(contains|may contain|ingredients:?|and|or)\b", " ", text)

    # Split by common delimiters
    ingredients = re.split(r"[,;•·\n]", text)

    # Clean each ingredient
    parsed: List[ParsedIngredient] = []
    for i, ing in enumerate(ingredients):
        ing = ing.strip()
        ing = re.sub(r"\s+", " ", ing)  # Normalize whitespace
        ing = re.sub(r"^\d+%?\s*", "", ing)  # Remove leading percentages
        ing = re.sub(r"\s*\d+%?$", "", ing)  # Remove trailing percentages

        if not ing or len(ing) < 2 or len(ing) > 60:
            continue

        # Check if this ingredient is in the minor section
        is_minor = minor_ingredients_section and ing in minor_ingredients_section

        # Get percentage if available
        percentage = percentage_map.get(ing)

        # Calculate weight
        if percentage is not None:
            # Use the actual percentage
            weight = percentage
        elif is_minor:
            # Minor ingredients get low weight
            weight = 0.02
        else:
            # Position-based weight: 1.0, 0.85, 0.72, 0.61, 0.52, ...
            weight = 0.85**i

        parsed.append(
            ParsedIngredient(
                name=ing,
                position_weight=weight,
                percentage=percentage,
                is_minor=is_minor,
            )
        )

    # Add parenthetical ingredients with moderate weight
    for paren_ing in parenthetical_additions:
        if not any(p.name == paren_ing for p in parsed):
            parsed.append(
                ParsedIngredient(
                    name=paren_ing,
                    position_weight=0.5,  # Moderate weight for parenthetical
                )
            )

    # Convert to simple tuple format for backward compatibility
    return [(p.name, p.position_weight) for p in parsed]


def normalize_ingredient(ingredient: str) -> str:
    """
    Normalize an ingredient name by removing modifiers and standardizing.

    Examples:
    - "organic fresh spinach" -> "spinach"
    - "low-fat vanilla yogurt" -> "yogurt"
    """
    # Common modifiers to remove
    modifiers = [
        "organic",
        "fresh",
        "frozen",
        "dried",
        "raw",
        "cooked",
        "roasted",
        "toasted",
        "blanched",
        "steamed",
        "grilled",
        "baked",
        "low-fat",
        "lowfat",
        "low fat",
        "nonfat",
        "non-fat",
        "fat-free",
        "reduced-fat",
        "light",
        "lite",
        "diet",
        "unsweetened",
        "sweetened",
        "natural",
        "pure",
        "100%",
        "whole",
        "sliced",
        "diced",
        "chopped",
        "minced",
        "ground",
        "crushed",
        "canned",
        "jarred",
        "bottled",
        "packaged",
        "premium",
        "gourmet",
        "artisan",
        "homemade",
        "homestyle",
    ]

    result = ingredient.lower()
    for mod in modifiers:
        result = re.sub(rf"\b{mod}\b", "", result)

    # Clean up
    result = re.sub(r"\s+", " ", result).strip()
    return result


def match_ingredient_to_profiles(
    ingredient: str, profiles: Dict[str, Dict[str, float]]
) -> Optional[Tuple[str, Dict[str, float], float]]:
    """
    Match an ingredient string to known nutrient profiles.

    Uses multi-level matching:
    1. Direct match (highest confidence)
    2. Synonym lookup
    3. Normalized match (after removing modifiers)
    4. Partial/fuzzy match (lowest confidence)

    Returns:
        Tuple of (matched_key, profile, match_confidence) or None
    """
    ingredient_lower = ingredient.lower().strip()

    # 1. Direct match - highest confidence
    if ingredient_lower in profiles:
        return (ingredient_lower, profiles[ingredient_lower], 1.0)

    # 2. Check synonyms
    if ingredient_lower in INGREDIENT_SYNONYMS:
        canonical = INGREDIENT_SYNONYMS[ingredient_lower]
        if canonical in profiles:
            return (canonical, profiles[canonical], 0.95)

    # 3. Normalize and try again
    normalized = normalize_ingredient(ingredient_lower)
    if normalized != ingredient_lower:
        if normalized in profiles:
            return (normalized, profiles[normalized], 0.9)
        if normalized in INGREDIENT_SYNONYMS:
            canonical = INGREDIENT_SYNONYMS[normalized]
            if canonical in profiles:
                return (canonical, profiles[canonical], 0.85)

    # 4. Check if any profile key is contained in the ingredient (fuzzy match)
    matches = []
    for key, profile in profiles.items():
        if key in ingredient_lower:
            # Prefer longer matches (more specific)
            # Confidence based on how much of the ingredient is matched
            match_ratio = len(key) / max(len(ingredient_lower), 1)
            confidence = 0.5 + (0.3 * match_ratio)  # 0.5-0.8 range
            matches.append((len(key), key, profile, confidence))

    if matches:
        # Return the longest (most specific) match
        matches.sort(reverse=True)
        return (matches[0][1], matches[0][2], matches[0][3])

    # 5. Try compound ingredient splitting
    # e.g., "whole milk powder" -> try "whole milk", then "milk"
    words = ingredient_lower.split()
    if len(words) > 1:
        # Try progressively shorter suffixes
        for start in range(len(words)):
            partial = " ".join(words[start:])
            if partial in profiles:
                confidence = 0.6 - (start * 0.1)  # Lower confidence for later words
                return (partial, profiles[partial], max(confidence, 0.4))
            if partial in INGREDIENT_SYNONYMS:
                canonical = INGREDIENT_SYNONYMS[partial]
                if canonical in profiles:
                    confidence = 0.55 - (start * 0.1)
                    return (canonical, profiles[canonical], max(confidence, 0.35))

    return None


def estimate_from_ingredients(
    ingredients_text: Optional[str], portion_weight: float = 100.0
) -> Optional[EstimationSignal]:
    """
    Estimate micronutrients from ingredient analysis.

    Args:
        ingredients_text: Raw ingredients text from product
        portion_weight: Serving size in grams

    Returns:
        EstimationSignal with ingredient-based estimates
    """
    if not ingredients_text:
        return None

    parsed = parse_ingredients(ingredients_text)
    if not parsed:
        return None

    # Accumulate weighted micronutrient estimates
    estimates: Dict[str, float] = {}
    total_weight = 0.0
    matches_found = 0

    total_match_confidence = 0.0
    for ingredient, position_weight in parsed:
        match = match_ingredient_to_profiles(
            ingredient, INGREDIENT_MICRONUTRIENT_PROFILES
        )
        if match:
            matched_key, profile, match_confidence = match
            matches_found += 1
            total_match_confidence += match_confidence

            for nutrient, value in profile.items():
                # Scale by position weight (first ingredients contribute more)
                # Also scale by match confidence (lower for fuzzy matches)
                scaled_value = value * position_weight * match_confidence

                # Scale by portion weight (profiles assume ~100g reference)
                scaled_value *= portion_weight / 100.0

                if nutrient in estimates:
                    estimates[nutrient] += scaled_value
                else:
                    estimates[nutrient] = scaled_value

            total_weight += position_weight * match_confidence

    if not estimates:
        return None

    # Calculate confidence based on:
    # 1. How many ingredients we matched
    # 2. Their cumulative position weight
    # 3. Average match quality (direct vs fuzzy matches)
    coverage = min(1.0, total_weight / 2.0)  # Full confidence at weight sum >= 2.0
    match_ratio = matches_found / max(len(parsed), 1)
    avg_match_confidence = total_match_confidence / max(matches_found, 1)

    # Weight: 40% coverage, 30% match ratio, 30% match quality
    confidence = (coverage * 0.4) + (match_ratio * 0.3) + (avg_match_confidence * 0.3)

    # Check for fortification indicators
    for keyword in FORTIFICATION_KEYWORDS:
        if keyword in ingredients_text.lower():
            confidence = min(1.0, confidence + 0.1)
            break

    logger.debug(
        f"Ingredient estimation: {matches_found}/{len(parsed)} ingredients matched, "
        f"confidence={confidence:.2f}"
    )

    return EstimationSignal(
        source="ingredient", confidence=confidence, estimates=estimates
    )


def estimate_from_food_name(
    food_name: str, portion_weight: float = 100.0
) -> Optional[EstimationSignal]:
    """
    Try to match product name to our food database entries.

    Args:
        food_name: Product name
        portion_weight: Serving size in grams

    Returns:
        EstimationSignal if a match is found
    """
    name_lower = food_name.lower()

    # Try direct match and alias match
    for food_key, entry in FOOD_DATABASE.items():
        # Check main name
        if food_key in name_lower or entry.display_name.lower() in name_lower:
            # Get category estimates for this food's category
            category_estimates = MICRONUTRIENT_CATEGORY_ESTIMATES.get(
                entry.category, MICRONUTRIENT_CATEGORY_ESTIMATES[FoodCategory.UNKNOWN]
            )

            # Scale to portion weight
            scale = portion_weight / 100.0
            estimates = {
                key: round(value * scale, 2) if value is not None else None
                for key, value in category_estimates.items()
                if value is not None
            }

            logger.debug(
                f"Food name match: '{food_name}' -> '{food_key}' ({entry.category.value})"
            )

            return EstimationSignal(
                source="food_match",
                confidence=0.8,  # High confidence for direct food match
                estimates=estimates,
            )

        # Check aliases
        if entry.aliases:
            for alias in entry.aliases:
                if alias.lower() in name_lower:
                    category_estimates = MICRONUTRIENT_CATEGORY_ESTIMATES.get(
                        entry.category,
                        MICRONUTRIENT_CATEGORY_ESTIMATES[FoodCategory.UNKNOWN],
                    )
                    scale = portion_weight / 100.0
                    estimates = {
                        key: round(value * scale, 2) if value is not None else None
                        for key, value in category_estimates.items()
                        if value is not None
                    }

                    return EstimationSignal(
                        source="food_match",
                        confidence=0.7,  # Slightly lower for alias match
                        estimates=estimates,
                    )

    return None


def estimate_from_macros(
    protein: Optional[float],
    fiber: Optional[float],
    fat: Optional[float],
    portion_weight: float = 100.0,
) -> Optional[EstimationSignal]:
    """
    Infer likely micronutrient levels from macronutrient profile.

    High protein foods tend to be good B12, zinc, iron sources.
    High fiber foods tend to be good magnesium, folate sources.

    Args:
        protein: Protein per 100g
        fiber: Fiber per 100g
        fat: Fat per 100g
        portion_weight: Serving size in grams

    Returns:
        EstimationSignal with macro-informed estimates
    """
    estimates = {}
    confidence = 0.0
    scale = portion_weight / 100.0

    # High protein (>15g/100g) suggests animal or legume source
    if protein and protein > 15:
        estimates["vitamin_b12"] = round(1.0 * scale, 2)  # Conservative B12 estimate
        estimates["zinc"] = round(2.0 * scale, 2)
        estimates["niacin"] = round(4.0 * scale, 2)
        estimates["vitamin_b6"] = round(0.3 * scale, 2)
        confidence = max(confidence, 0.4)

        # Very high protein (>25g) - likely meat/fish
        if protein > 25:
            estimates["vitamin_b12"] = round(2.0 * scale, 2)
            estimates["zinc"] = round(3.5 * scale, 2)
            estimates["iron"] = round(1.5 * scale, 2)
            confidence = max(confidence, 0.5)

    # High fiber (>5g/100g) suggests whole grains/legumes/vegetables
    if fiber and fiber > 5:
        estimates["magnesium"] = round(40 * scale, 1)
        estimates["folate"] = round(50 * scale, 1)
        estimates["potassium"] = round(250 * scale, 0)
        confidence = max(confidence, 0.35)

        # Very high fiber (>10g) - likely legumes or bran
        if fiber > 10:
            estimates["magnesium"] = round(60 * scale, 1)
            estimates["iron"] = round(2.0 * scale, 2)
            estimates["folate"] = round(100 * scale, 0)
            confidence = max(confidence, 0.45)

    if not estimates:
        return None

    return EstimationSignal(source="macros", confidence=confidence, estimates=estimates)


def combine_estimation_signals(
    signals: List[EstimationSignal], category_baseline: Dict[str, Optional[float]]
) -> Dict[str, float]:
    """
    Combine multiple estimation signals using confidence-weighted averaging.

    The category baseline is used as a fallback for nutrients not covered
    by other signals.

    Args:
        signals: List of EstimationSignal objects
        category_baseline: Baseline estimates from food category

    Returns:
        Combined micronutrient estimates
    """
    # All micronutrient keys we want to estimate
    all_keys = [
        "potassium",
        "calcium",
        "iron",
        "magnesium",
        "zinc",
        "phosphorus",
        "vitamin_a",
        "vitamin_c",
        "vitamin_d",
        "vitamin_e",
        "vitamin_k",
        "vitamin_b6",
        "vitamin_b12",
        "folate",
        "thiamin",
        "riboflavin",
        "niacin",
    ]

    combined = {}

    for key in all_keys:
        # Collect all estimates for this nutrient with their confidence
        estimates_for_key = []

        for signal in signals:
            if key in signal.estimates and signal.estimates[key] is not None:
                estimates_for_key.append((signal.estimates[key], signal.confidence))

        if estimates_for_key:
            # Confidence-weighted average
            total_confidence = sum(conf for _, conf in estimates_for_key)
            if total_confidence > 0:
                weighted_sum = sum(val * conf for val, conf in estimates_for_key)
                combined[key] = round(weighted_sum / total_confidence, 2)
        elif key in category_baseline and category_baseline[key] is not None:
            # Fall back to category baseline
            combined[key] = category_baseline[key]

    return combined


def estimate_micronutrients_sophisticated(
    food_name: str,
    categories: Optional[List[str]],
    portion_weight: float,
    ingredients_text: Optional[str] = None,
    protein: Optional[float] = None,
    fiber: Optional[float] = None,
    fat: Optional[float] = None,
    existing: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], str, str, str]:
    """
    Sophisticated micronutrient estimation using multiple signals.

    This function combines:
    1. Ingredient-based estimation (highest priority when available)
    2. Food name matching to database
    3. Macronutrient-informed inference
    4. Category-based baseline (fallback)

    Args:
        food_name: Product name
        categories: Open Food Facts category tags
        portion_weight: Serving size in grams
        ingredients_text: Raw ingredients text
        protein: Protein per 100g (for macro-informed estimation)
        fiber: Fiber per 100g
        fat: Fat per 100g
        existing: Already known micronutrients (won't be overwritten)

    Returns:
        Tuple of (estimates_dict, category_used, confidence, source_description)
    """
    from app.api.food_analysis import map_off_categories_to_food_category

    signals: List[EstimationSignal] = []
    sources_used = []

    # Signal 1: Ingredient-based estimation (highest quality)
    ingredient_signal = estimate_from_ingredients(ingredients_text, portion_weight)
    if ingredient_signal:
        signals.append(ingredient_signal)
        sources_used.append("ingredients")
        logger.info(f"Ingredient signal: confidence={ingredient_signal.confidence:.2f}")

    # Signal 2: Food name matching
    name_signal = estimate_from_food_name(food_name, portion_weight)
    if name_signal:
        signals.append(name_signal)
        sources_used.append("food_match")
        logger.info(f"Food name signal: confidence={name_signal.confidence:.2f}")

    # Signal 3: Macronutrient-informed estimation
    macro_signal = estimate_from_macros(protein, fiber, fat, portion_weight)
    if macro_signal:
        signals.append(macro_signal)
        sources_used.append("macros")
        logger.info(f"Macro signal: confidence={macro_signal.confidence:.2f}")

    # Get category baseline
    food_category, category_confidence = map_off_categories_to_food_category(categories)
    category_estimates = MICRONUTRIENT_CATEGORY_ESTIMATES.get(
        food_category, MICRONUTRIENT_CATEGORY_ESTIMATES[FoodCategory.UNKNOWN]
    )

    # Scale category baseline to portion weight
    scale = portion_weight / 100.0
    scaled_category_baseline = {
        key: round(value * scale, 2) if value is not None else None
        for key, value in category_estimates.items()
    }

    # Combine all signals
    combined = combine_estimation_signals(signals, scaled_category_baseline)

    # Remove nutrients that already have values
    if existing:
        for key in list(combined.keys()):
            if key in existing and existing[key] is not None:
                del combined[key]

    # Determine overall confidence and source description
    if signals:
        max_confidence_signal = max(signals, key=lambda s: s.confidence)
        overall_confidence = (
            "high"
            if max_confidence_signal.confidence > 0.6
            else "medium"
            if max_confidence_signal.confidence > 0.35
            else "low"
        )
        source_desc = "+".join(sources_used) if sources_used else "category"
    else:
        overall_confidence = category_confidence
        source_desc = "category_only"
        sources_used.append("category")

    logger.info(
        f"Sophisticated estimation for '{food_name}': "
        f"{len(combined)} nutrients estimated, "
        f"sources={source_desc}, confidence={overall_confidence}"
    )

    return combined, food_category.value, overall_confidence, source_desc
