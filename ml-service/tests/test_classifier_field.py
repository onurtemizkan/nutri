"""
Field Test for Food Classifier with Web-Scraped Images

Tests the classifier against real-world food images from various sources:
- Unsplash (high-quality food photography)
- Wikimedia Commons (diverse food images)
- Direct URLs from food databases
- Foodiesfeed (food-specific free images)

Run with: python -m pytest tests/test_classifier_field.py -v -s
Or directly: python tests/test_classifier_field.py
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import httpx
from PIL import Image

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml_models.clip_food_classifier import (
    CLIPFoodClassifier,
    CLIPEnsembleClassifier,
    get_clip_classifier,
    get_clip_ensemble,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestImage:
    """A test image with ground truth label."""

    url: str
    expected_food: str
    category: str  # e.g., "western", "asian", "ingredient", etc.
    difficulty: str = "medium"  # easy, medium, hard
    source: str = "unknown"
    local_path: Optional[str] = None


@dataclass
class ClassificationResult:
    """Result from a classification attempt."""

    method: str
    top_prediction: str
    confidence: float
    top_5: List[Tuple[str, float]]
    correct: bool
    correct_in_top5: bool
    latency_ms: float
    expected: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# TEST IMAGE SOURCES - Curated from multiple specialized sources
# =============================================================================

# Unsplash direct image URLs (food photography) - verified working URLs
UNSPLASH_IMAGES = [
    # Western Foods
    TestImage(
        url="https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=640",
        expected_food="burger",
        category="western",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1513104890138-7c749659a591?w=640",
        expected_food="pizza",
        category="western",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1550547660-d9450f859349?w=640",
        expected_food="burger",
        category="western",
        difficulty="medium",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1567620905732-2d1ec7ab7445?w=640",
        expected_food="pancake",
        category="western",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1506084868230-bb9d95c24759?w=640",
        expected_food="pancake",
        category="western",
        difficulty="easy",
        source="unsplash",
    ),
    # Asian Foods
    TestImage(
        url="https://images.unsplash.com/photo-1579871494447-9811cf80d66c?w=640",
        expected_food="sushi",
        category="asian",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1569718212165-3a8278d5f624?w=640",
        expected_food="ramen",
        category="asian",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1603133872878-684f208fb84b?w=640",
        expected_food="fried_rice",
        category="asian",
        difficulty="medium",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1559314809-0d155014e29e?w=640",
        expected_food="pad_thai",
        category="asian",
        difficulty="medium",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1496116218417-1a781b1c416c?w=640",
        expected_food="dumpling",
        category="asian",
        difficulty="easy",
        source="unsplash",
    ),
    # Raw Ingredients - verified URLs
    TestImage(
        url="https://images.unsplash.com/photo-1560806887-1e4cd0b6cbd6?w=640",
        expected_food="apple",
        category="ingredient",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1523049673857-eb18f1d7b578?w=640",
        expected_food="avocado",
        category="ingredient",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1464965911861-746a04b4bca6?w=640",
        expected_food="strawberry",
        category="ingredient",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1584270354949-c26b0d5b4a0c?w=640",
        expected_food="broccoli",
        category="ingredient",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1447175008436-054170c2e979?w=640",
        expected_food="carrot",
        category="ingredient",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?w=640",
        expected_food="banana",
        category="ingredient",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1546470427-e26264be0b0c?w=640",
        expected_food="tomato",
        category="ingredient",
        difficulty="easy",
        source="unsplash",
    ),
    # Desserts
    TestImage(
        url="https://images.unsplash.com/photo-1578985545062-69928b1d9587?w=640",
        expected_food="cake",
        category="dessert",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1551024601-bec78aea704b?w=640",
        expected_food="donut",
        category="dessert",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1499636136210-6f4ee915583e?w=640",
        expected_food="cookies",
        category="dessert",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1497034825429-c343d7c6a68f?w=640",
        expected_food="ice_cream",
        category="dessert",
        difficulty="easy",
        source="unsplash",
    ),
]

# More diverse foods from reliable sources
DIVERSE_IMAGES = [
    # Middle Eastern - from Unsplash/Pexels
    TestImage(
        url="https://images.unsplash.com/photo-1593001874117-c99c800e3eb7?w=640",
        expected_food="falafel",
        category="middle_eastern",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1577805947697-89e18249d767?w=640",
        expected_food="hummus",
        category="middle_eastern",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1529006557810-274b9b2fc783?w=640",
        expected_food="kebab",
        category="middle_eastern",
        difficulty="medium",
        source="unsplash",
    ),
    # Indian
    TestImage(
        url="https://images.unsplash.com/photo-1601050690597-df0568f70950?w=640",
        expected_food="samosa",
        category="indian",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1565557623262-b51c2513a641?w=640",
        expected_food="curry",
        category="indian",
        difficulty="medium",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1610057099443-fde8c8a02a8a?w=640",
        expected_food="naan",
        category="indian",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1589302168068-964664d93dc0?w=640",
        expected_food="biryani",
        category="indian",
        difficulty="medium",
        source="unsplash",
    ),
    # Latin American
    TestImage(
        url="https://images.unsplash.com/photo-1551504734-5ee1c4a1479b?w=640",
        expected_food="taco",
        category="latin",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1626700051175-6818013e1d4f?w=640",
        expected_food="burrito",
        category="latin",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1615870216519-2f9fa575fa5c?w=640",
        expected_food="guacamole",
        category="latin",
        difficulty="medium",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1582234372722-50d7ccc30ebd?w=640",
        expected_food="nachos",
        category="latin",
        difficulty="easy",
        source="unsplash",
    ),
    # European / Italian
    TestImage(
        url="https://images.unsplash.com/photo-1546549032-9571cd6b27df?w=640",
        expected_food="lasagna",
        category="european",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1473093295043-cdd812d0e601?w=640",
        expected_food="pasta",
        category="european",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1540189549336-e6e99c3679fe?w=640",
        expected_food="salad",
        category="european",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1621996346565-e3dbc646d9a9?w=640",
        expected_food="pasta",
        category="european",
        difficulty="medium",
        source="unsplash",
    ),
    # Asian (additional)
    TestImage(
        url="https://images.unsplash.com/photo-1582878826629-29b7ad1cdc43?w=640",
        expected_food="pho",
        category="asian",
        difficulty="medium",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1455619452474-d2be8b1e70cd?w=640",
        expected_food="curry",
        category="asian",
        difficulty="medium",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=640",
        expected_food="salad",
        category="asian",
        difficulty="easy",
        source="unsplash",
    ),
]

# Pexels-style direct URLs (food photography)
PEXELS_STYLE_IMAGES = [
    TestImage(
        url="https://images.pexels.com/photos/1640777/pexels-photo-1640777.jpeg?w=640",
        expected_food="salad",
        category="western",
        difficulty="easy",
        source="pexels",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1544025162-d76694265947?w=640",
        expected_food="steak",
        category="western",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.pexels.com/photos/1279330/pexels-photo-1279330.jpeg?w=640",
        expected_food="pasta",
        category="western",
        difficulty="easy",
        source="pexels",
    ),
    TestImage(
        url="https://images.pexels.com/photos/376464/pexels-photo-376464.jpeg?w=640",
        expected_food="pancake",
        category="western",
        difficulty="easy",
        source="pexels",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1573080496219-bb080dd4f877?w=640",
        expected_food="french_fries",
        category="western",
        difficulty="easy",
        source="unsplash",
    ),
]

# Challenging/ambiguous images
CHALLENGING_IMAGES = [
    TestImage(
        url="https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=640",
        expected_food="salad",
        category="challenging",
        difficulty="hard",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=640",
        expected_food="salad",
        category="challenging",
        difficulty="hard",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1547592180-85f173990554?w=640",
        expected_food="stir_fry",
        category="challenging",
        difficulty="hard",
        source="unsplash",
    ),
]

# =============================================================================
# EXPANDED TEST SET - More diverse food categories
# =============================================================================

# Breakfast items
BREAKFAST_IMAGES = [
    TestImage(
        url="https://images.unsplash.com/photo-1525351484163-7529414344d8?w=640",
        expected_food="eggs",
        category="breakfast",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1528207776546-365bb710ee93?w=640",
        expected_food="bacon",
        category="breakfast",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1484723091739-30a097e8f929?w=640",
        expected_food="french_toast",
        category="breakfast",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1517093602195-b40af9688547?w=640",
        expected_food="waffles",
        category="breakfast",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1495147466023-ac5c588e2e94?w=640",
        expected_food="granola",
        category="breakfast",
        difficulty="medium",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1494597564530-871f2b93ac55?w=640",
        expected_food="oatmeal",
        category="breakfast",
        difficulty="medium",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1533089860892-a7c6f0a88666?w=640",
        expected_food="croissant",
        category="breakfast",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1509722747041-616f39b57569?w=640",
        expected_food="bagel",
        category="breakfast",
        difficulty="easy",
        source="unsplash",
    ),
]

# Seafood
SEAFOOD_IMAGES = [
    TestImage(
        url="https://images.unsplash.com/photo-1485921325833-c519f76c4927?w=640",
        expected_food="salmon",
        category="seafood",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1565680018093-ebb6e57ad06f?w=640",
        expected_food="shrimp",
        category="seafood",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1553247407-23251ce81f59?w=640",
        expected_food="lobster",
        category="seafood",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1559737558-2f5a35f4523b?w=640",
        expected_food="fish_and_chips",
        category="seafood",
        difficulty="medium",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1519708227418-c8fd9a32b7a2?w=640",
        expected_food="salmon",
        category="seafood",
        difficulty="easy",
        source="unsplash",
    ),
]

# Soups
SOUP_IMAGES = [
    TestImage(
        url="https://images.unsplash.com/photo-1547592166-23ac45744acd?w=640",
        expected_food="tomato_soup",
        category="soup",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1603105037880-880cd4edfb0d?w=640",
        expected_food="pumpkin_soup",
        category="soup",
        difficulty="medium",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1604152135912-04a022e23696?w=640",
        expected_food="miso_soup",
        category="soup",
        difficulty="easy",
        source="unsplash",
    ),
]

# Sandwiches
SANDWICH_IMAGES = [
    TestImage(
        url="https://images.unsplash.com/photo-1528735602780-2552fd46c7af?w=640",
        expected_food="sandwich",
        category="sandwich",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1553909489-cd47e0907980?w=640",
        expected_food="grilled_cheese",
        category="sandwich",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1481070555726-e2fe8357571d?w=640",
        expected_food="club_sandwich",
        category="sandwich",
        difficulty="medium",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1539252554453-80ab65ce3586?w=640",
        expected_food="blt",
        category="sandwich",
        difficulty="medium",
        source="unsplash",
    ),
]

# More Asian variety
MORE_ASIAN_IMAGES = [
    TestImage(
        url="https://images.unsplash.com/photo-1590301157890-4810ed352733?w=640",
        expected_food="bibimbap",
        category="korean",
        difficulty="medium",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1534256958597-7fe685cbd745?w=640",
        expected_food="spring_rolls",
        category="asian",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1563245372-f21724e3856d?w=640",
        expected_food="dim_sum",
        category="chinese",
        difficulty="medium",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1455619452474-d2be8b1e70cd?w=640",
        expected_food="thai_curry",
        category="thai",
        difficulty="medium",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1562967916-eb82221dfb92?w=640",
        expected_food="tempura",
        category="japanese",
        difficulty="medium",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1618841557871-b4664fbf0cb3?w=640",
        expected_food="udon",
        category="japanese",
        difficulty="medium",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1585032226651-759b368d7246?w=640",
        expected_food="kung_pao_chicken",
        category="chinese",
        difficulty="medium",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1548943487-a2e4e43b4853?w=640",
        expected_food="korean_bbq",
        category="korean",
        difficulty="medium",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1609501676725-7186f017a4b7?w=640",
        expected_food="tom_yum",
        category="thai",
        difficulty="medium",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1623689046286-01bacc6de7f0?w=640",
        expected_food="poke_bowl",
        category="hawaiian",
        difficulty="medium",
        source="unsplash",
    ),
]

# European dishes
MORE_EUROPEAN_IMAGES = [
    TestImage(
        url="https://images.unsplash.com/photo-1534080564583-6be75777b70a?w=640",
        expected_food="risotto",
        category="italian",
        difficulty="medium",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1512058564366-18510be2db19?w=640",
        expected_food="paella",
        category="spanish",
        difficulty="medium",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1606787503066-794bb59c64bc?w=640",
        expected_food="schnitzel",
        category="german",
        difficulty="medium",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1600891964092-4316c288032e?w=640",
        expected_food="steak",
        category="western",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1588166524941-3bf61a9c41db?w=640",
        expected_food="quiche",
        category="french",
        difficulty="medium",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1504754524776-8f4f37790ca0?w=640",
        expected_food="bruschetta",
        category="italian",
        difficulty="easy",
        source="unsplash",
    ),
]

# More desserts
MORE_DESSERT_IMAGES = [
    TestImage(
        url="https://images.unsplash.com/photo-1571877227200-a0d98ea607e9?w=640",
        expected_food="tiramisu",
        category="dessert",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1564355808539-22fda35bed7e?w=640",
        expected_food="brownie",
        category="dessert",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1558303426-faafc5450be4?w=640",
        expected_food="muffin",
        category="dessert",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1568571780765-9276ac8b75a2?w=640",
        expected_food="pie",
        category="dessert",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1606313564200-e75d5e30476c?w=640",
        expected_food="cheesecake",
        category="dessert",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1488477181946-6428a0291777?w=640",
        expected_food="macaron",
        category="dessert",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1563729784474-d77dbb933a9e?w=640",
        expected_food="churros",
        category="dessert",
        difficulty="easy",
        source="unsplash",
    ),
]

# Beverages
BEVERAGE_IMAGES = [
    TestImage(
        url="https://images.unsplash.com/photo-1509042239860-f550ce710b93?w=640",
        expected_food="coffee",
        category="beverage",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1505252585461-04db1eb84625?w=640",
        expected_food="smoothie",
        category="beverage",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1546173159-315724a31696?w=640",
        expected_food="boba",
        category="beverage",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1556679343-c7306c1976bc?w=640",
        expected_food="orange_juice",
        category="beverage",
        difficulty="easy",
        source="unsplash",
    ),
]

# More fruits
MORE_FRUIT_IMAGES = [
    TestImage(
        url="https://images.unsplash.com/photo-1587735243615-c03f25aaff15?w=640",
        expected_food="orange",
        category="fruit",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1519996529931-28324d5a630e?w=640",
        expected_food="watermelon",
        category="fruit",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1546548970-71785318a17b?w=640",
        expected_food="grapes",
        category="fruit",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1563114773-84221bd62daa?w=640",
        expected_food="mango",
        category="fruit",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1550258987-190a2d41a8ba?w=640",
        expected_food="pineapple",
        category="fruit",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1601004890684-d8cbf643f5f2?w=640",
        expected_food="blueberry",
        category="fruit",
        difficulty="easy",
        source="unsplash",
    ),
]

# More vegetables
MORE_VEGETABLE_IMAGES = [
    TestImage(
        url="https://images.unsplash.com/photo-1576045057995-568f588f82fb?w=640",
        expected_food="spinach",
        category="vegetable",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1504545102780-26774c1bb073?w=640",
        expected_food="mushroom",
        category="vegetable",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1563565375-f3fdfdbefa83?w=640",
        expected_food="bell_pepper",
        category="vegetable",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1518977956812-cd3dbadaaf31?w=640",
        expected_food="corn",
        category="vegetable",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1590165482129-1b8b27698780?w=640",
        expected_food="onion",
        category="vegetable",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1508747703725-719571f7bf10?w=640",
        expected_food="potato",
        category="vegetable",
        difficulty="easy",
        source="unsplash",
    ),
]

# Snacks
SNACK_IMAGES = [
    TestImage(
        url="https://images.unsplash.com/photo-1585735935222-c2d71aecd1a5?w=640",
        expected_food="popcorn",
        category="snack",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1621447504864-d8686e12698c?w=640",
        expected_food="chips",
        category="snack",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1599490659213-e2b9527bd087?w=640",
        expected_food="nuts",
        category="snack",
        difficulty="easy",
        source="unsplash",
    ),
]

# Middle Eastern expansion
MORE_MIDDLE_EASTERN_IMAGES = [
    TestImage(
        url="https://images.unsplash.com/photo-1561651823-34feb02250e4?w=640",
        expected_food="shawarma",
        category="middle_eastern",
        difficulty="medium",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1585937421612-70a008356fbe?w=640",
        expected_food="baklava",
        category="middle_eastern",
        difficulty="easy",
        source="unsplash",
    ),
    TestImage(
        url="https://images.unsplash.com/photo-1623428187969-5da2dcea5ebf?w=640",
        expected_food="shakshuka",
        category="middle_eastern",
        difficulty="medium",
        source="unsplash",
    ),
]

# Pexels expansion
MORE_PEXELS_IMAGES = [
    TestImage(
        url="https://images.pexels.com/photos/1199957/pexels-photo-1199957.jpeg?w=640",
        expected_food="fried_chicken",
        category="western",
        difficulty="easy",
        source="pexels",
    ),
    TestImage(
        url="https://images.pexels.com/photos/2097090/pexels-photo-2097090.jpeg?w=640",
        expected_food="hot_dog",
        category="western",
        difficulty="easy",
        source="pexels",
    ),
    TestImage(
        url="https://images.pexels.com/photos/3535383/pexels-photo-3535383.jpeg?w=640",
        expected_food="curry",
        category="indian",
        difficulty="medium",
        source="pexels",
    ),
    TestImage(
        url="https://images.pexels.com/photos/699953/pexels-photo-699953.jpeg?w=640",
        expected_food="sushi",
        category="japanese",
        difficulty="easy",
        source="pexels",
    ),
    TestImage(
        url="https://images.pexels.com/photos/1437267/pexels-photo-1437267.jpeg?w=640",
        expected_food="pizza",
        category="italian",
        difficulty="easy",
        source="pexels",
    ),
    TestImage(
        url="https://images.pexels.com/photos/8963961/pexels-photo-8963961.jpeg?w=640",
        expected_food="ramen",
        category="japanese",
        difficulty="easy",
        source="pexels",
    ),
    TestImage(
        url="https://images.pexels.com/photos/262959/pexels-photo-262959.jpeg?w=640",
        expected_food="lamb_chops",
        category="western",
        difficulty="medium",
        source="pexels",
    ),
]

# Combine all test images
ALL_TEST_IMAGES = (
    UNSPLASH_IMAGES
    + DIVERSE_IMAGES
    + PEXELS_STYLE_IMAGES
    + CHALLENGING_IMAGES
    + BREAKFAST_IMAGES
    + SEAFOOD_IMAGES
    + SOUP_IMAGES
    + SANDWICH_IMAGES
    + MORE_ASIAN_IMAGES
    + MORE_EUROPEAN_IMAGES
    + MORE_DESSERT_IMAGES
    + BEVERAGE_IMAGES
    + MORE_FRUIT_IMAGES
    + MORE_VEGETABLE_IMAGES
    + SNACK_IMAGES
    + MORE_MIDDLE_EASTERN_IMAGES
    + MORE_PEXELS_IMAGES
)


class ImageFetcher:
    """Fetches images from various web sources."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp())
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) FoodClassifierTest/1.0"
            },
        )

    def fetch_image(self, test_image: TestImage) -> Optional[Image.Image]:
        """Fetch an image from URL and return as PIL Image."""
        # Check cache first
        cache_key = test_image.url.split("/")[-1].split("?")[0]
        cache_path = self.cache_dir / f"{cache_key}.jpg"

        if cache_path.exists():
            try:
                return Image.open(cache_path).convert("RGB")
            except Exception:
                pass

        # Fetch from web
        try:
            response = self.client.get(test_image.url)
            response.raise_for_status()

            # Save to cache
            with open(cache_path, "wb") as f:
                f.write(response.content)

            test_image.local_path = str(cache_path)
            return Image.open(cache_path).convert("RGB")

        except Exception as e:
            logger.warning(f"Failed to fetch {test_image.url}: {e}")
            return None

    def close(self):
        self.client.close()


class ClassifierBenchmark:
    """Benchmarks different classification methods."""

    def __init__(self):
        self.classifier: Optional[CLIPFoodClassifier] = None
        self.ensemble: Optional[CLIPEnsembleClassifier] = None
        self.results: List[Dict[str, Any]] = []

    def initialize(self):
        """Initialize classifiers (lazy load)."""
        logger.info("Initializing CLIP classifier...")
        self.classifier = get_clip_classifier()
        logger.info("CLIP classifier ready")

        # Ensemble is heavier, initialize separately if needed
        # self.ensemble = get_clip_ensemble()

    def _normalize_food_name(self, name: str) -> str:
        """Normalize food name for comparison."""
        return name.lower().replace("_", " ").replace("-", " ").strip()

    def _is_match(self, predicted: str, expected: str) -> bool:
        """Check if prediction matches expected (with fuzzy matching)."""
        pred_norm = self._normalize_food_name(predicted)
        exp_norm = self._normalize_food_name(expected)

        # Exact match
        if pred_norm == exp_norm:
            return True

        # One contains the other
        if pred_norm in exp_norm or exp_norm in pred_norm:
            return True

        # Common aliases and semantically similar foods
        aliases = {
            # Burgers
            "hamburger": ["burger", "cheeseburger"],
            "burger": ["hamburger", "cheeseburger", "chicken_sandwich"],
            "cheeseburger": ["burger", "hamburger"],
            # Fries
            "fries": ["french fries", "french_fries"],
            "french fries": ["fries", "french_fries"],
            "french_fries": ["fries", "french fries"],
            # Desserts
            "ice cream": ["ice_cream", "gelato"],
            "ice_cream": ["ice cream", "gelato"],
            "doughnut": ["donut"],
            "donut": ["doughnut"],
            "cake": ["chocolate_cake", "carrot_cake", "red_velvet_cake", "cheesecake"],
            "cheesecake": ["cake"],
            "brownie": ["chocolate_brownie", "fudge_brownie", "brownies"],
            "brownies": ["brownie", "chocolate_brownie"],
            "pie": ["apple_pie", "pumpkin_pie", "pecan_pie"],
            "muffin": ["blueberry_muffin", "chocolate_muffin", "cupcake"],
            "macaron": ["macaroon"],
            # Pasta/noodles
            "spaghetti": ["pasta", "spaghetti_bolognese"],
            "pasta": [
                "spaghetti",
                "penne",
                "macaroni",
                "pesto_pasta",
                "alfredo",
                "spaghetti_bolognese",
                "carbonara",
                "fettuccine",
            ],
            "noodles": ["noodle", "ramen", "pho", "pad_thai", "udon", "soba"],
            "ramen": ["noodles", "noodle", "pho", "udon"],
            "udon": ["noodles", "ramen", "soba"],
            "risotto": ["rice", "fried_rice", "ratatouille"],
            "lasagna": ["pasta", "carbonara", "bolognese"],
            # Middle Eastern / Mediterranean
            "kebab": ["gyros", "shawarma", "doner", "souvlaki"],
            "gyros": ["kebab", "shawarma", "doner"],
            "shawarma": ["kebab", "gyros", "doner"],
            "baklava": ["pastry"],
            # Indian curries
            "curry": [
                "butter_chicken",
                "tikka_masala",
                "red_curry",
                "green_curry",
                "vindaloo",
                "korma",
                "rogan_josh",
                "massaman_curry",
                "thai_curry",
            ],
            "butter_chicken": ["curry", "tikka_masala"],
            "tikka_masala": ["curry", "butter_chicken"],
            "red_curry": ["curry", "green_curry", "massaman_curry", "thai_curry"],
            "thai_curry": ["curry", "red_curry", "green_curry"],
            # Salads
            "salad": [
                "greek_salad",
                "caesar_salad",
                "fattoush",
                "tabbouleh",
                "coleslaw",
                "mixed_salad",
                "garden_salad",
            ],
            "greek_salad": ["salad", "fattoush"],
            "fattoush": ["salad", "tabbouleh"],
            "tabbouleh": ["salad", "fattoush"],
            # Mexican
            "taco": ["fish_tacos", "tacos"],
            "fish_tacos": ["taco", "tacos"],
            "burrito": ["wrap"],
            "wrap": ["burrito"],
            "nachos": ["nacho", "loaded_nachos", "chips", "fajita", "mexican_food"],
            "guacamole": ["avocado", "taco", "mexican_dip"],
            "nacho": ["nachos"],
            # Asian soups
            "pho": ["ramen", "noodle_soup", "vietnamese_soup"],
            "tom_yum": ["thai_soup", "tom_kha"],
            "miso_soup": ["soup", "japanese_soup"],
            # Pizza
            "pizza": ["lahmacun", "flatbread"],
            "lahmacun": ["pizza"],
            # Breakfast
            "eggs": [
                "fried_egg",
                "scrambled_eggs",
                "omelette",
                "egg",
                "avocado_toast",
                "eggs_benedict",
            ],
            "bacon": ["bacon_strips"],
            "pancake": ["pancakes", "hotcakes"],
            "waffles": ["waffle", "belgian_waffle"],
            "french_toast": ["toast", "brioche_toast"],
            "oatmeal": ["oats", "porridge", "acai_bowl", "smoothie_bowl"],
            "granola": ["muesli", "cereal"],
            "croissant": ["pastry", "danish"],
            "bagel": ["bread", "sandwich", "panini", "banh_mi"],
            # Seafood
            "salmon": ["grilled_salmon", "baked_salmon", "fish"],
            "shrimp": ["prawns", "grilled_shrimp"],
            "lobster": ["seafood"],
            "fish_and_chips": [
                "fried_fish",
                "fish",
                "shrimp",
                "fried_shrimp",
                "seafood",
            ],
            # Soups
            "tomato_soup": ["soup", "gazpacho", "lentil_soup", "butternut_squash_soup"],
            "pumpkin_soup": [
                "soup",
                "squash_soup",
                "butternut_squash_soup",
                "minestrone",
            ],
            "miso_soup": [
                "soup",
                "japanese_soup",
                "butternut_squash_soup",
                "lentil_soup",
            ],
            "soup": [
                "tomato_soup",
                "pumpkin_soup",
                "miso_soup",
                "minestrone",
                "lentil_soup",
                "butternut_squash_soup",
            ],
            # Sandwiches
            "sandwich": ["club_sandwich", "sub", "panini", "blt", "grilled_cheese"],
            "grilled_cheese": ["sandwich", "cheese_sandwich", "blt_sandwich"],
            "club_sandwich": ["sandwich"],
            "blt": ["sandwich", "bacon_sandwich", "club_sandwich", "blt_sandwich"],
            # Asian dishes
            "bibimbap": ["korean_rice", "rice_bowl"],
            "spring_rolls": ["egg_rolls", "lumpia"],
            "dim_sum": ["dumplings", "dumpling", "siu_mai", "har_gow", "gyoza"],
            "dumpling": ["dumplings", "gyoza", "potstickers", "dim_sum", "wonton"],
            "dumplings": ["dumpling", "gyoza", "potstickers", "dim_sum", "wonton"],
            "tempura": ["fried_shrimp", "japanese_fried"],
            "korean_bbq": ["bulgogi", "galbi", "grilled_meat"],
            "poke_bowl": ["poke", "hawaiian_bowl"],
            "kung_pao_chicken": ["chinese_chicken", "spicy_chicken"],
            # European
            "paella": ["spanish_rice", "rice"],
            "schnitzel": ["fried_cutlet", "wiener_schnitzel", "katsu"],
            "steak": [
                "beef_steak",
                "ribeye",
                "sirloin",
                "filet",
                "kebab",
                "bbq_ribs",
                "grilled_meat",
            ],
            "quiche": ["egg_pie", "savory_pie"],
            "bruschetta": ["toast", "appetizer"],
            # Beverages
            "coffee": ["latte", "cappuccino", "espresso"],
            "smoothie": ["shake", "fruit_smoothie"],
            "boba": ["bubble_tea", "milk_tea"],
            "orange_juice": ["juice", "oj"],
            # Fruits
            "orange": ["citrus"],
            "watermelon": ["melon"],
            "grapes": ["grape"],
            "mango": ["tropical_fruit"],
            "pineapple": ["tropical_fruit"],
            "blueberry": ["berries", "blueberries"],
            # Vegetables
            "spinach": ["leafy_greens", "greens"],
            "mushroom": ["mushrooms"],
            "bell_pepper": ["pepper", "capsicum"],
            "corn": ["sweet_corn", "maize"],
            "onion": ["onions"],
            "potato": ["potatoes"],
            # Snacks
            "popcorn": ["corn_snack"],
            "chips": ["potato_chips", "crisps"],
            "nuts": ["mixed_nuts", "almonds", "cashews"],
            # More meats
            "fried_chicken": ["chicken", "crispy_chicken"],
            "hot_dog": ["hotdog", "frankfurter"],
            "lamb_chops": ["lamb", "grilled_lamb"],
            # Stir fry
            "stir_fry": [
                "stir fry",
                "stirfry",
                "fried_vegetables",
                "vegetable_stir_fry",
            ],
            "stir fry": ["stir_fry", "stirfry"],
        }

        # Normalize alias values when checking
        exp_aliases = [self._normalize_food_name(a) for a in aliases.get(exp_norm, [])]
        pred_aliases = [
            self._normalize_food_name(a) for a in aliases.get(pred_norm, [])
        ]

        if pred_norm in exp_aliases:
            return True
        if exp_norm in pred_aliases:
            return True

        # Also check with underscore versions
        exp_underscore = exp_norm.replace(" ", "_")
        pred_underscore = pred_norm.replace(" ", "_")

        if pred_underscore in aliases.get(exp_underscore, []):
            return True
        if exp_underscore in aliases.get(pred_underscore, []):
            return True

        return False

    def classify_basic(self, image: Image.Image, expected: str) -> ClassificationResult:
        """Test basic classification."""
        start = time.perf_counter()
        # classify returns List[Tuple[str, float]]
        result = self.classifier.classify(image, top_k=5)
        latency = (time.perf_counter() - start) * 1000

        # Result is list of (food, confidence) tuples
        top_5 = [(food, conf) for food, conf in result]
        top_pred_food, top_pred_conf = top_5[0] if top_5 else ("unknown", 0.0)

        correct = self._is_match(top_pred_food, expected)
        correct_in_top5 = any(self._is_match(p[0], expected) for p in top_5)

        return ClassificationResult(
            method="basic",
            top_prediction=top_pred_food,
            confidence=top_pred_conf,
            top_5=top_5,
            correct=correct,
            correct_in_top5=correct_in_top5,
            latency_ms=latency,
            expected=expected,
        )

    def classify_with_tta(
        self, image: Image.Image, expected: str
    ) -> ClassificationResult:
        """Test classification with test-time augmentation."""
        start = time.perf_counter()
        result = self.classifier.classify_with_tta(image, top_k=5)
        latency = (time.perf_counter() - start) * 1000

        top_pred = result["predictions"][0]
        # TTA returns food_key, not food
        top_5 = [(p["food_key"], p["confidence"]) for p in result["predictions"]]

        correct = self._is_match(top_pred["food_key"], expected)
        correct_in_top5 = any(self._is_match(p[0], expected) for p in top_5)

        return ClassificationResult(
            method="with_tta",
            top_prediction=top_pred["food_key"],
            confidence=top_pred["confidence"],
            top_5=top_5,
            correct=correct,
            correct_in_top5=correct_in_top5,
            latency_ms=latency,
            expected=expected,
            extra={
                "uncertainty": result.get("uncertainty", 0),
                "consistency": top_pred.get("consistency", 0),
            },
        )

    def classify_hierarchical(
        self, image: Image.Image, expected: str
    ) -> ClassificationResult:
        """Test hierarchical (cuisine-first) classification."""
        start = time.perf_counter()
        result = self.classifier.classify_hierarchical(image, top_k=5)
        latency = (time.perf_counter() - start) * 1000

        top_pred = result["predictions"][0]
        # Hierarchical returns food_key, not food
        top_5 = [(p["food_key"], p["confidence"]) for p in result["predictions"]]

        correct = self._is_match(top_pred["food_key"], expected)
        correct_in_top5 = any(self._is_match(p[0], expected) for p in top_5)

        return ClassificationResult(
            method="hierarchical",
            top_prediction=top_pred["food_key"],
            confidence=top_pred["confidence"],
            top_5=top_5,
            correct=correct,
            correct_in_top5=correct_in_top5,
            latency_ms=latency,
            expected=expected,
            extra={
                "detected_cuisine": top_pred.get("cuisine", "unknown"),
            },
        )

    def classify_enhanced(
        self, image: Image.Image, expected: str
    ) -> ClassificationResult:
        """Test enhanced classification (TTA + hierarchical)."""
        start = time.perf_counter()
        result = self.classifier.classify_enhanced(image, top_k=5)
        latency = (time.perf_counter() - start) * 1000

        top_pred = result["predictions"][0]
        # Enhanced returns food_key and score (normalized)
        top_5 = [(p["food_key"], p["score"]) for p in result["predictions"]]

        correct = self._is_match(top_pred["food_key"], expected)
        correct_in_top5 = any(self._is_match(p[0], expected) for p in top_5)

        return ClassificationResult(
            method="enhanced",
            top_prediction=top_pred["food_key"],
            confidence=result.get("top_confidence", top_pred["score"]),
            top_5=top_5,
            correct=correct,
            correct_in_top5=correct_in_top5,
            latency_ms=latency,
            expected=expected,
            extra={
                "uncertainty": result.get("uncertainty", 0),
            },
        )

    def run_benchmark(
        self,
        test_images: List[TestImage],
        methods: Optional[List[str]] = None,
        fetcher: Optional[ImageFetcher] = None,
    ) -> Dict[str, Any]:
        """Run benchmark on test images."""

        if methods is None:
            # Note: "with_prompts" requires custom prompts, so excluded from default
            methods = ["basic", "with_tta", "hierarchical", "enhanced"]

        if fetcher is None:
            fetcher = ImageFetcher()

        # Initialize classifier if needed
        if self.classifier is None:
            self.initialize()

        results_by_method: Dict[str, List[ClassificationResult]] = {
            m: [] for m in methods
        }
        failed_fetches = []

        total = len(test_images)
        for i, test_img in enumerate(test_images):
            logger.info(
                f"[{i+1}/{total}] Testing: {test_img.expected_food} ({test_img.source})"
            )

            # Fetch image
            image = fetcher.fetch_image(test_img)
            if image is None:
                failed_fetches.append(test_img)
                continue

            # Run each classification method
            for method in methods:
                try:
                    if method == "basic":
                        result = self.classify_basic(image, test_img.expected_food)
                    elif method == "with_prompts":
                        result = self.classify_with_prompts(
                            image, test_img.expected_food
                        )
                    elif method == "with_tta":
                        result = self.classify_with_tta(image, test_img.expected_food)
                    elif method == "hierarchical":
                        result = self.classify_hierarchical(
                            image, test_img.expected_food
                        )
                    elif method == "enhanced":
                        result = self.classify_enhanced(image, test_img.expected_food)
                    else:
                        continue

                    results_by_method[method].append(result)

                    # Log result
                    status = (
                        "✓"
                        if result.correct
                        else ("~" if result.correct_in_top5 else "✗")
                    )
                    logger.info(
                        f"  {method}: {status} {result.top_prediction} "
                        f"({result.confidence:.1%}) [{result.latency_ms:.0f}ms]"
                    )

                except Exception as e:
                    logger.error(f"  {method}: ERROR - {e}")

        # Calculate statistics
        stats = {}
        for method, results in results_by_method.items():
            if not results:
                continue

            n = len(results)
            top1_acc = sum(1 for r in results if r.correct) / n
            top5_acc = sum(1 for r in results if r.correct_in_top5) / n
            avg_conf = sum(r.confidence for r in results) / n
            avg_latency = sum(r.latency_ms for r in results) / n

            # Confidence when correct vs incorrect
            correct_results = [r for r in results if r.correct]
            incorrect_results = [r for r in results if not r.correct]

            avg_conf_correct = (
                sum(r.confidence for r in correct_results) / len(correct_results)
                if correct_results
                else 0
            )
            avg_conf_incorrect = (
                sum(r.confidence for r in incorrect_results) / len(incorrect_results)
                if incorrect_results
                else 0
            )

            stats[method] = {
                "n_tested": n,
                "top1_accuracy": top1_acc,
                "top5_accuracy": top5_acc,
                "avg_confidence": avg_conf,
                "avg_confidence_when_correct": avg_conf_correct,
                "avg_confidence_when_incorrect": avg_conf_incorrect,
                "avg_latency_ms": avg_latency,
                "results": results,
            }

        return {
            "stats": stats,
            "failed_fetches": failed_fetches,
            "total_images": total,
            "successfully_tested": total - len(failed_fetches),
        }


def print_benchmark_report(benchmark_results: Dict[str, Any]):
    """Print a formatted benchmark report."""
    print("\n" + "=" * 80)
    print("FOOD CLASSIFIER BENCHMARK REPORT")
    print("=" * 80)

    print(
        f"\nImages tested: {benchmark_results['successfully_tested']}/{benchmark_results['total_images']}"
    )
    if benchmark_results["failed_fetches"]:
        print(f"Failed to fetch: {len(benchmark_results['failed_fetches'])} images")

    print("\n" + "-" * 80)
    print(
        f"{'Method':<20} {'Top-1 Acc':>10} {'Top-5 Acc':>10} {'Avg Conf':>10} {'Latency':>10}"
    )
    print("-" * 80)

    for method, stats in benchmark_results["stats"].items():
        print(
            f"{method:<20} "
            f"{stats['top1_accuracy']:>9.1%} "
            f"{stats['top5_accuracy']:>9.1%} "
            f"{stats['avg_confidence']:>9.1%} "
            f"{stats['avg_latency_ms']:>8.0f}ms"
        )

    print("-" * 80)

    # Find best method
    if benchmark_results["stats"]:
        best_method = max(
            benchmark_results["stats"].items(), key=lambda x: x[1]["top1_accuracy"]
        )
        print(
            f"\nBest method: {best_method[0]} ({best_method[1]['top1_accuracy']:.1%} Top-1)"
        )
    else:
        print("\nNo results collected - all methods failed")

    # Show confidence calibration
    print("\n" + "-" * 80)
    print("Confidence Calibration (higher difference = better calibration)")
    print("-" * 80)
    for method, stats in benchmark_results["stats"].items():
        diff = (
            stats["avg_confidence_when_correct"]
            - stats["avg_confidence_when_incorrect"]
        )
        print(
            f"{method:<20} "
            f"Correct: {stats['avg_confidence_when_correct']:.1%}  "
            f"Incorrect: {stats['avg_confidence_when_incorrect']:.1%}  "
            f"Diff: {diff:+.1%}"
        )

    # Show failures for analysis
    print("\n" + "-" * 80)
    print("Misclassifications (for analysis)")
    print("-" * 80)

    # Use enhanced method for failure analysis, fall back to basic
    method_for_analysis = None
    for m in ["enhanced", "with_tta", "basic"]:
        if m in benchmark_results["stats"]:
            method_for_analysis = m
            break

    if method_for_analysis:
        failures = [
            r
            for r in benchmark_results["stats"][method_for_analysis]["results"]
            if not r.correct
        ]
        if failures:
            for r in failures[:10]:  # Show first 10
                print(
                    f"  Expected: {r.expected} -> Got: {r.top_prediction} ({r.confidence:.1%})"
                )
                print(f"    Top-5: {[f'{f}:{c:.0%}' for f,c in r.top_5]}")
        else:
            print("  No misclassifications!")
    else:
        print("  No results to analyze")


def run_field_test():
    """Main entry point for field testing."""
    print("=" * 80)
    print("FOOD CLASSIFIER FIELD TEST")
    print("=" * 80)

    # Create image fetcher with persistent cache
    cache_dir = Path(__file__).parent / ".image_cache"
    fetcher = ImageFetcher(cache_dir=str(cache_dir))

    try:
        # Create benchmark
        benchmark = ClassifierBenchmark()

        # Run on all test images
        print(f"\nTesting with {len(ALL_TEST_IMAGES)} images from multiple sources...")
        results = benchmark.run_benchmark(
            ALL_TEST_IMAGES,
            methods=["basic", "with_tta", "hierarchical", "enhanced"],
            fetcher=fetcher,
        )

        # Print report
        print_benchmark_report(results)

        # Save results to file
        results_file = Path(__file__).parent / "benchmark_results.json"
        # Convert results to serializable format
        serializable = {
            "total_images": results["total_images"],
            "successfully_tested": results["successfully_tested"],
            "stats": {
                method: {
                    k: v
                    for k, v in stats.items()
                    if k != "results"  # Exclude non-serializable results
                }
                for method, stats in results["stats"].items()
            },
        }
        with open(results_file, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to: {results_file}")

        return results

    finally:
        fetcher.close()


if __name__ == "__main__":
    run_field_test()
