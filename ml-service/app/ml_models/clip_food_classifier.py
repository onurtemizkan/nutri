"""
CLIP-based Food Classifier

Uses OpenAI's CLIP model for zero-shot food classification.
Key advantages:
- No training required - uses text prompts
- Can classify ANY food category
- Better at raw produce, ingredients, and unusual foods
- Trained on 400M web image-text pairs

Prompt engineering is critical for accuracy.
"""
# mypy: ignore-errors

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy loading
_clip_model: Any = None
_clip_processor: Any = None
_clip_loaded = False
_text_features_cache: Dict[str, Any] = {}

# Food categories with descriptive prompts for better CLIP matching
# Format: database_key -> list of descriptive prompts
FOOD_PROMPTS: Dict[str, List[str]] = {
    # === FRUITS ===
    "apple": ["a red apple fruit", "a green apple", "a fresh apple"],
    "avocado": ["a ripe avocado", "a halved avocado showing the pit", "avocado fruit"],
    "banana": ["a yellow banana", "a bunch of bananas", "a ripe banana fruit"],
    "blueberry": ["fresh blueberries", "a bowl of blueberries", "blueberry fruit"],
    "cherry": ["red cherries", "fresh cherries with stems", "cherry fruit"],
    "grape": ["a bunch of grapes", "green grapes", "red grapes"],
    "kiwi": ["a kiwi fruit", "sliced kiwi showing green flesh", "kiwifruit"],
    "lemon": ["a yellow lemon", "fresh lemon citrus", "lemon fruit"],
    "lime": ["a green lime", "fresh lime citrus", "lime fruit"],
    "mango": ["a ripe mango", "sliced mango fruit", "mango"],
    "orange": ["an orange citrus fruit", "a fresh orange", "peeled orange"],
    "peach": ["a ripe peach", "fresh peach fruit", "peach"],
    "pear": ["a ripe pear", "fresh pear fruit", "green pear"],
    "pineapple": ["a pineapple fruit", "sliced pineapple", "fresh pineapple"],
    "raspberry": ["fresh raspberries", "red raspberries", "raspberry fruit"],
    "strawberry": ["fresh strawberries", "red strawberry", "ripe strawberries"],
    "watermelon": ["watermelon slices", "fresh watermelon", "watermelon fruit"],
    # === VEGETABLES ===
    "asparagus": ["fresh asparagus spears", "green asparagus", "asparagus vegetable"],
    "bell_pepper": [
        "a colorful bell pepper vegetable",
        "red bell pepper on cutting board",
        "green bell pepper whole",
        "yellow bell pepper sliced",
        "raw bell peppers in kitchen",
        "sweet pepper vegetable",
    ],
    "broccoli": ["fresh broccoli florets", "green broccoli", "broccoli vegetable"],
    "cabbage": ["a head of cabbage", "green cabbage", "cabbage vegetable"],
    "carrot": ["fresh carrots", "orange carrots", "carrot vegetable"],
    "cauliflower": ["fresh cauliflower", "white cauliflower florets", "cauliflower"],
    "celery": ["celery stalks", "fresh celery", "celery vegetable"],
    "corn": ["corn on the cob", "fresh yellow corn", "sweet corn"],
    "cucumber": ["a fresh cucumber", "sliced cucumber", "green cucumber"],
    "eggplant": ["a purple eggplant", "fresh eggplant", "aubergine"],
    "lettuce": ["fresh lettuce leaves", "green lettuce", "romaine lettuce"],
    "mushroom": ["fresh mushrooms", "white button mushrooms", "sliced mushrooms"],
    "onion": ["a fresh onion", "sliced onion", "yellow onion", "red onion"],
    "potato": ["a potato", "raw potatoes", "baked potato"],
    "spinach": ["fresh spinach leaves", "green spinach", "baby spinach"],
    "tomato": ["a fresh tomato", "red tomatoes", "ripe tomato"],
    "zucchini": ["fresh zucchini", "green zucchini", "sliced zucchini"],
    "beets": ["fresh beets", "red beets", "beetroot"],
    # === NUTS & SEEDS ===
    "almonds": ["whole almonds", "raw almonds", "almond nuts"],
    "cashews": ["cashew nuts", "raw cashews", "roasted cashews"],
    "peanuts": ["peanuts", "roasted peanuts", "peanut nuts"],
    "walnuts": ["walnut halves", "shelled walnuts", "walnut nuts"],
    "mixed_nuts": ["mixed nuts", "assorted nuts", "variety of nuts"],
    "chia_seeds": ["chia seeds", "black chia seeds"],
    "flax_seeds": ["flaxseeds", "ground flaxseed"],
    "sunflower_seeds": ["sunflower seeds", "shelled sunflower seeds"],
    # === PROTEINS ===
    "chicken": ["grilled chicken breast", "roasted chicken", "cooked chicken meat"],
    "beef": ["cooked beef", "grilled steak", "beef meat"],
    "pork": ["cooked pork", "pork chop", "pork meat"],
    "bacon": ["crispy bacon strips", "cooked bacon", "bacon slices"],
    "salmon": ["grilled salmon fillet", "cooked salmon", "salmon fish"],
    "tuna": ["tuna fish", "seared tuna", "canned tuna"],
    "fish": ["cooked fish fillet", "grilled fish", "fish dish"],
    "shrimp": [
        "cooked pink shrimp seafood",
        "grilled shrimp on plate",
        "shrimp prawns",
        "jumbo shrimp dish",
        "shrimp cocktail seafood",
        "sauteed shrimp",
    ],
    "eggs": [
        "sunny side up fried eggs in pan",
        "fried eggs with runny yolk",
        "scrambled eggs on plate",
        "eggs over easy",
        "breakfast eggs cooked",
        "pan fried eggs",
    ],
    "tofu": ["tofu cubes", "fried tofu", "silken tofu"],
    # === DAIRY ===
    "cheese": [
        "cheese wedge on plate",
        "cheese platter assortment",
        "sliced cheese variety",
        "cheese board with crackers",
        "block of cheese",
        "cheddar cheese slices",
        "brie cheese wheel",
    ],
    "milk": ["glass of milk", "white milk", "pouring milk"],
    "yogurt": ["yogurt in a bowl", "greek yogurt", "plain yogurt"],
    "butter": ["butter block", "sliced butter"],
    # === GRAINS & CARBS ===
    "rice": ["cooked white rice", "steamed rice", "rice bowl"],
    "pasta": ["cooked pasta", "spaghetti noodles", "pasta dish"],
    "bread": ["sliced bread", "loaf of bread", "fresh bread"],
    "oats": [
        "bowl of oatmeal porridge",
        "oatmeal with toppings breakfast",
        "hot oatmeal cereal",
        "creamy oatmeal in bowl",
        "rolled oats cooked",
        "porridge with fruit",
    ],
    "quinoa": ["cooked quinoa", "quinoa grain"],
    # === PREPARED DISHES ===
    "burger": ["a hamburger", "cheeseburger", "beef burger with bun"],
    "pizza": ["a slice of pizza", "whole pizza", "pepperoni pizza"],
    "hot_dog": ["a hot dog in bun", "frankfurter", "hot dog with mustard"],
    "taco": ["a taco", "beef taco", "taco with toppings"],
    "sushi": ["sushi rolls", "nigiri sushi", "sushi plate"],
    "sandwich": ["a sandwich", "deli sandwich", "sub sandwich"],
    "salad": ["a fresh salad", "green salad", "mixed salad"],
    "soup": ["a bowl of soup", "hot soup", "vegetable soup"],
    "stew": ["beef stew", "hearty stew", "meat stew"],
    "curry": ["curry dish", "chicken curry", "curry with rice"],
    # === CHICKEN DISHES ===
    "chicken_kiev": [
        "chicken kiev cutlet",
        "breaded chicken kiev with butter filling",
        "fried chicken kiev on plate",
        "golden breaded chicken stuffed with garlic butter",
        "crispy chicken kiev dinner",
        "chicken cordon bleu style cutlet",
    ],
    "chicken_nuggets": [
        "chicken nuggets",
        "breaded chicken nuggets",
        "fried chicken pieces",
        "chicken tenders",
        "crispy chicken bites",
    ],
    "chicken_wings": [
        "chicken wings",
        "buffalo wings",
        "fried chicken wings",
        "bbq chicken wings",
        "crispy chicken wings",
    ],
    "fried_chicken": [
        "fried chicken pieces",
        "crispy fried chicken",
        "southern fried chicken",
        "breaded fried chicken",
        "kfc style chicken",
    ],
    "chicken_schnitzel": [
        "chicken schnitzel cutlet",
        "breaded chicken schnitzel",
        "pan fried chicken schnitzel",
        "wiener schnitzel chicken",
        "crispy breaded chicken cutlet",
    ],
    "roast_chicken": [
        "roasted whole chicken",
        "roast chicken dinner",
        "rotisserie chicken",
        "baked chicken",
        "golden roasted chicken",
    ],
    # === OTHER PREPARED MEATS ===
    "meatballs": [
        "meatballs in sauce",
        "swedish meatballs",
        "italian meatballs",
        "beef meatballs",
        "spaghetti and meatballs",
    ],
    "meatloaf": [
        "meatloaf slice",
        "baked meatloaf",
        "homemade meatloaf dinner",
        "ground beef meatloaf",
    ],
    "lasagna": [
        "lasagna serving",
        "baked lasagna",
        "meat lasagna",
        "italian lasagna pasta",
        "cheesy lasagna slice",
    ],
    "casserole": [
        "baked casserole dish",
        "casserole in baking dish",
        "comfort food casserole",
        "homemade casserole",
    ],
    # === FISH DISHES ===
    "fish_and_chips": [
        "fish and chips",
        "battered fried fish with fries",
        "british fish and chips",
        "cod fish and chips",
    ],
    "fish_sticks": [
        "fish sticks",
        "fish fingers",
        "breaded fish sticks",
        "frozen fish sticks",
    ],
    # === POTATO DISHES ===
    "french_fries": [
        "french fries",
        "potato fries",
        "golden french fries",
        "crispy fries",
        "fried potato sticks",
    ],
    "mashed_potatoes": [
        "mashed potatoes",
        "creamy mashed potatoes",
        "whipped potatoes",
        "potato puree",
    ],
    "baked_potato": [
        "baked potato with toppings",
        "loaded baked potato",
        "jacket potato",
        "stuffed baked potato",
    ],
    "hash_browns": [
        "hash browns",
        "crispy hash browns",
        "shredded potato hash",
        "breakfast hash browns",
    ],
    # === BREAKFAST ===
    "pancake": ["stack of pancakes", "pancakes with syrup", "fluffy pancakes"],
    "waffle": [
        "golden waffle with grid pattern",
        "belgian waffle breakfast",
        "waffle with syrup and butter",
        "crispy waffle on plate",
        "square waffle breakfast food",
        "homemade waffles",
    ],
    "cereal": ["bowl of cereal", "breakfast cereal", "cereal with milk"],
    "granola": ["granola", "granola cereal", "crunchy granola"],
    "bagel": ["a bagel", "toasted bagel", "bagel with cream cheese"],
    "croissant": ["a croissant", "buttery croissant", "french croissant"],
    "donut": [
        "glazed donut pastry",
        "chocolate frosted donut",
        "ring shaped doughnut",
        "donut with sprinkles",
        "fried donut dessert",
        "bakery donut",
    ],
    "muffin": ["a muffin", "blueberry muffin", "breakfast muffin"],
    "toast": ["buttered toast", "toast slices", "toasted bread"],
    # === DESSERTS ===
    "chocolate": ["chocolate bar", "chocolate pieces", "dark chocolate"],
    "ice_cream": ["ice cream scoop", "ice cream cone", "vanilla ice cream"],
    "cake": ["slice of cake", "birthday cake", "chocolate cake"],
    "cookies": ["chocolate chip cookies", "baked cookies", "cookies"],
    "pie": ["a slice of pie", "fruit pie", "apple pie"],
    # === BEVERAGES ===
    "coffee": ["cup of coffee", "black coffee", "latte"],
    "tea": ["cup of tea", "hot tea", "green tea"],
    "juice": ["glass of juice", "orange juice", "fruit juice"],
    "smoothie": ["fruit smoothie", "berry smoothie", "smoothie drink"],
    "soda": ["glass of soda", "cola drink", "soft drink"],
    "water": ["glass of water", "bottled water", "drinking water"],
    "beer": ["glass of beer", "beer mug", "draft beer"],
    "wine": ["glass of wine", "red wine", "white wine"],
    # === CONDIMENTS & EXTRAS ===
    "honey": ["honey jar", "golden honey", "dripping honey"],
    "olive_oil": ["olive oil bottle", "extra virgin olive oil"],
    "chickpeas": ["cooked chickpeas", "garbanzo beans", "chickpea bowl"],
    "hummus": ["hummus dip", "bowl of hummus", "chickpea hummus"],
    "guacamole": ["guacamole dip", "fresh guacamole", "avocado dip"],
    "salsa": ["tomato salsa", "fresh salsa", "salsa dip"],
    # === ASIAN DISHES ===
    "ramen": [
        "bowl of ramen noodles with broth",
        "japanese ramen soup",
        "noodle soup with toppings",
        "ramen with egg and pork",
        "hot steaming ramen bowl",
        "asian noodle soup dish",
    ],
    "pho": ["pho noodle soup", "vietnamese pho", "beef pho"],
    "fried_rice": ["fried rice", "chinese fried rice", "vegetable fried rice"],
    "dumplings": ["steamed dumplings", "asian dumplings", "pot stickers"],
    "spring_rolls": ["spring rolls", "fried spring rolls", "vietnamese spring rolls"],
    "pad_thai": ["pad thai noodles", "thai pad thai", "stir fried noodles"],
    # === SNACKS ===
    "chips": ["potato chips", "tortilla chips", "crispy chips"],
    "popcorn": ["popcorn", "buttered popcorn", "movie popcorn"],
    "pretzels": ["pretzels", "soft pretzel", "pretzel snack"],
    "crackers": ["crackers", "saltine crackers", "cheese crackers"],
    # ==========================================================================
    # COMPREHENSIVE INTERNATIONAL CUISINE EXPANSION
    # ==========================================================================
    # === CHINESE CUISINE ===
    "kung_pao_chicken": [
        "kung pao chicken with peanuts",
        "spicy kung pao chicken",
        "chinese kung pao dish",
        "diced chicken with chili peppers",
        "sichuan kung pao chicken",
    ],
    "general_tso_chicken": [
        "general tso chicken",
        "crispy orange chicken",
        "sweet and spicy fried chicken",
        "chinese general tso",
        "battered chicken in sauce",
    ],
    "sweet_and_sour_pork": [
        "sweet and sour pork",
        "chinese sweet sour pork",
        "pork with pineapple sauce",
        "battered pork in red sauce",
    ],
    "beef_and_broccoli": [
        "beef and broccoli stir fry",
        "chinese beef broccoli",
        "sliced beef with broccoli",
        "wok fried beef and broccoli",
    ],
    "mongolian_beef": [
        "mongolian beef",
        "crispy beef with scallions",
        "sweet soy beef",
        "chinese mongolian beef dish",
    ],
    "orange_chicken": [
        "orange chicken",
        "crispy orange chicken",
        "chinese orange chicken",
        "sweet citrus chicken",
    ],
    "sesame_chicken": [
        "sesame chicken",
        "crispy sesame chicken",
        "chicken with sesame seeds",
        "chinese sesame chicken",
    ],
    "chow_mein": [
        "chow mein noodles",
        "stir fried noodles with vegetables",
        "chinese chow mein",
        "crispy noodle dish",
    ],
    "lo_mein": [
        "lo mein noodles",
        "soft stir fried noodles",
        "chinese lo mein",
        "noodles with vegetables",
    ],
    "egg_foo_young": [
        "egg foo young",
        "chinese omelette",
        "egg patty with gravy",
        "vegetable egg pancake",
    ],
    "mapo_tofu": [
        "mapo tofu",
        "spicy tofu in chili sauce",
        "sichuan mapo tofu",
        "tofu with ground pork",
    ],
    "peking_duck": [
        "peking duck",
        "crispy duck skin",
        "beijing roast duck",
        "sliced duck with pancakes",
    ],
    "dim_sum": [
        "dim sum platter",
        "chinese dim sum",
        "steamed dim sum baskets",
        "variety of dim sum",
    ],
    "char_siu": [
        "char siu bbq pork",
        "chinese bbq pork",
        "red glazed pork",
        "cantonese roast pork",
    ],
    "wonton": [
        "wonton dumplings",
        "wonton soup",
        "fried wontons",
        "chinese wontons",
    ],
    "congee": [
        "rice congee porridge",
        "chinese congee",
        "rice porridge with toppings",
        "jook rice soup",
    ],
    "hot_pot": [
        "chinese hot pot",
        "boiling hot pot",
        "steamboat fondue",
        "shabu shabu pot",
    ],
    "scallion_pancakes": [
        "scallion pancakes",
        "chinese green onion pancakes",
        "crispy scallion flatbread",
        "cong you bing",
    ],
    "dan_dan_noodles": [
        "dan dan noodles",
        "spicy sichuan noodles",
        "noodles with chili oil",
        "chinese dan dan mian",
    ],
    # === JAPANESE CUISINE ===
    "sashimi": [
        "sashimi platter",
        "raw fish slices",
        "japanese sashimi",
        "fresh fish sashimi",
    ],
    "tempura": [
        "tempura shrimp",
        "vegetable tempura",
        "japanese fried tempura",
        "crispy battered tempura",
    ],
    "teriyaki": [
        "teriyaki chicken",
        "teriyaki salmon",
        "japanese teriyaki",
        "glazed teriyaki meat",
    ],
    "tonkatsu": [
        "tonkatsu pork cutlet",
        "japanese breaded pork",
        "crispy pork katsu",
        "fried pork cutlet",
    ],
    "udon": [
        "udon noodle soup",
        "japanese udon",
        "thick wheat noodles",
        "udon bowl",
    ],
    "soba": [
        "soba noodles",
        "japanese buckwheat noodles",
        "cold soba",
        "zaru soba",
    ],
    "yakitori": [
        "yakitori skewers",
        "grilled chicken skewers",
        "japanese yakitori",
        "chicken on sticks",
    ],
    "gyoza": [
        "gyoza dumplings",
        "japanese pot stickers",
        "pan fried gyoza",
        "steamed gyoza",
    ],
    "onigiri": [
        "onigiri rice ball",
        "japanese rice triangle",
        "seaweed wrapped rice",
        "stuffed rice ball",
    ],
    "miso_soup": [
        "miso soup",
        "japanese miso soup",
        "tofu miso soup",
        "wakame miso soup",
    ],
    "okonomiyaki": [
        "okonomiyaki pancake",
        "japanese savory pancake",
        "cabbage pancake",
        "osaka style okonomiyaki",
    ],
    "takoyaki": [
        "takoyaki balls",
        "octopus balls",
        "japanese takoyaki",
        "fried octopus dumplings",
    ],
    "katsu_curry": [
        "katsu curry",
        "japanese curry with cutlet",
        "curry rice with katsu",
        "chicken katsu curry",
    ],
    "donburi": [
        "donburi rice bowl",
        "japanese rice bowl",
        "katsudon bowl",
        "gyudon beef bowl",
    ],
    "edamame": [
        "edamame beans",
        "steamed edamame",
        "salted soybeans",
        "japanese edamame",
    ],
    "matcha": [
        "matcha green tea",
        "matcha latte",
        "matcha dessert",
        "green tea matcha",
    ],
    "mochi": [
        "mochi rice cake",
        "japanese mochi",
        "filled mochi",
        "daifuku mochi",
    ],
    # === KOREAN CUISINE ===
    "bibimbap": [
        "bibimbap rice bowl",
        "korean mixed rice",
        "vegetables and rice bowl",
        "dolsot bibimbap",
    ],
    "bulgogi": [
        "bulgogi beef",
        "korean bbq beef",
        "marinated bulgogi",
        "grilled bulgogi",
    ],
    "korean_fried_chicken": [
        "korean fried chicken",
        "crispy korean chicken",
        "yangnyeom chicken",
        "double fried chicken",
    ],
    "kimchi": [
        "kimchi fermented cabbage",
        "korean kimchi",
        "spicy kimchi",
        "napa cabbage kimchi",
    ],
    "japchae": [
        "japchae glass noodles",
        "korean stir fried noodles",
        "sweet potato noodles",
        "colorful japchae",
    ],
    "tteokbokki": [
        "tteokbokki rice cakes",
        "korean spicy rice cakes",
        "red tteokbokki",
        "street food tteokbokki",
    ],
    "kimbap": [
        "kimbap rolls",
        "korean seaweed rice rolls",
        "sliced kimbap",
        "vegetable kimbap",
    ],
    "samgyeopsal": [
        "samgyeopsal pork belly",
        "korean grilled pork",
        "thick pork belly slices",
        "korean bbq meat",
    ],
    "sundubu_jjigae": [
        "sundubu jjigae stew",
        "korean soft tofu stew",
        "spicy tofu soup",
        "bubbling tofu stew",
    ],
    "kimchi_jjigae": [
        "kimchi jjigae stew",
        "korean kimchi stew",
        "kimchi soup",
        "pork kimchi stew",
    ],
    "korean_bbq": [
        "korean bbq grill",
        "korean barbecue meat",
        "tabletop korean bbq",
        "grilling korean meat",
    ],
    "army_stew": [
        "army stew budae jjigae",
        "korean army base stew",
        "ramen with spam stew",
        "budae jjigae",
    ],
    "pajeon": [
        "pajeon korean pancake",
        "scallion pancake",
        "korean savory pancake",
        "seafood pajeon",
    ],
    # === THAI CUISINE ===
    "green_curry": [
        "thai green curry",
        "green curry with chicken",
        "coconut green curry",
        "spicy green curry",
    ],
    "red_curry": [
        "thai red curry",
        "red curry with beef",
        "coconut red curry",
        "spicy red curry",
    ],
    "massaman_curry": [
        "massaman curry",
        "thai massaman",
        "peanut curry",
        "mild thai curry",
    ],
    "tom_yum": [
        "tom yum soup",
        "spicy thai soup",
        "tom yum goong",
        "sour and spicy soup",
    ],
    "tom_kha": [
        "tom kha gai soup",
        "coconut chicken soup",
        "thai coconut soup",
        "galangal soup",
    ],
    "thai_basil_chicken": [
        "thai basil chicken",
        "pad krapao",
        "holy basil stir fry",
        "chicken with basil",
    ],
    "larb": [
        "larb salad",
        "thai minced meat salad",
        "laab",
        "spicy meat salad",
    ],
    "satay": [
        "satay skewers",
        "chicken satay",
        "grilled satay with peanut sauce",
        "thai satay",
    ],
    "mango_sticky_rice": [
        "mango sticky rice",
        "thai mango dessert",
        "coconut sticky rice with mango",
        "khao niao mamuang",
    ],
    "papaya_salad": [
        "papaya salad som tam",
        "green papaya salad",
        "spicy thai salad",
        "som tum",
    ],
    "thai_iced_tea": [
        "thai iced tea",
        "orange thai tea",
        "cha yen",
        "sweet thai tea",
    ],
    # === VIETNAMESE CUISINE ===
    "banh_mi": [
        "banh mi sandwich",
        "vietnamese baguette sandwich",
        "pork banh mi",
        "crispy banh mi",
    ],
    "bun_cha": [
        "bun cha noodles",
        "vietnamese grilled pork",
        "noodles with meatballs",
        "hanoi bun cha",
    ],
    "cao_lau": [
        "cao lau noodles",
        "hoi an noodles",
        "vietnamese cao lau",
        "pork noodle bowl",
    ],
    "com_tam": [
        "com tam broken rice",
        "vietnamese broken rice",
        "grilled pork rice",
        "rice with pork chop",
    ],
    "goi_cuon": [
        "goi cuon fresh rolls",
        "vietnamese spring rolls",
        "rice paper rolls",
        "fresh summer rolls",
    ],
    "bun_bo_hue": [
        "bun bo hue",
        "spicy beef noodle soup",
        "hue style noodles",
        "vietnamese spicy soup",
    ],
    # === INDIAN CUISINE ===
    "butter_chicken": [
        "butter chicken",
        "murgh makhani",
        "creamy tomato chicken curry",
        "indian butter chicken",
    ],
    "tikka_masala": [
        "chicken tikka masala",
        "tikka masala curry",
        "creamy masala curry",
        "indian tikka masala",
    ],
    "tandoori_chicken": [
        "tandoori chicken",
        "red tandoori chicken",
        "grilled indian chicken",
        "spiced yogurt chicken",
    ],
    "biryani": [
        "biryani rice",
        "chicken biryani",
        "lamb biryani",
        "spiced rice dish",
    ],
    "naan": [
        "naan bread",
        "indian flatbread",
        "garlic naan",
        "butter naan",
    ],
    "samosa": [
        "samosas",
        "fried indian pastry",
        "vegetable samosa",
        "triangular samosa",
    ],
    "dal": [
        "dal lentils",
        "indian dal curry",
        "yellow dal",
        "lentil soup",
    ],
    "palak_paneer": [
        "palak paneer",
        "spinach and cheese curry",
        "saag paneer",
        "green spinach curry",
    ],
    "paneer_tikka": [
        "paneer tikka",
        "grilled paneer",
        "tandoori paneer",
        "indian cheese tikka",
    ],
    "chana_masala": [
        "chana masala",
        "chickpea curry",
        "chole",
        "spiced chickpeas",
    ],
    "korma": [
        "korma curry",
        "chicken korma",
        "creamy korma",
        "mild indian curry",
    ],
    "vindaloo": [
        "vindaloo curry",
        "spicy vindaloo",
        "pork vindaloo",
        "hot indian curry",
    ],
    "rogan_josh": [
        "rogan josh lamb",
        "kashmiri lamb curry",
        "red lamb curry",
        "aromatic meat curry",
    ],
    "aloo_gobi": [
        "aloo gobi",
        "potato cauliflower curry",
        "indian potato dish",
        "vegetable curry",
    ],
    "dosa": [
        "dosa crepe",
        "masala dosa",
        "south indian dosa",
        "crispy rice crepe",
    ],
    "idli": [
        "idli rice cakes",
        "steamed idli",
        "south indian idli",
        "white rice cakes",
    ],
    "pakora": [
        "pakora fritters",
        "vegetable pakora",
        "onion bhaji",
        "fried indian snack",
    ],
    "raita": [
        "raita yogurt",
        "cucumber raita",
        "indian yogurt dip",
        "cooling raita",
    ],
    "lassi": [
        "lassi drink",
        "mango lassi",
        "yogurt lassi",
        "indian smoothie",
    ],
    "gulab_jamun": [
        "gulab jamun",
        "indian sweet balls",
        "fried milk balls in syrup",
        "indian dessert",
    ],
    "jalebi": [
        "jalebi sweets",
        "orange spiral dessert",
        "fried syrup sweets",
        "indian jalebi",
    ],
    # === MEXICAN & LATIN CUISINE ===
    "burrito": [
        "burrito wrapped",
        "mexican burrito",
        "large flour tortilla wrap",
        "stuffed burrito",
    ],
    "enchiladas": [
        "enchiladas",
        "cheese enchiladas",
        "rolled tortillas in sauce",
        "mexican enchiladas",
    ],
    "quesadilla": [
        "quesadilla",
        "cheese quesadilla",
        "grilled tortilla with cheese",
        "mexican quesadilla",
    ],
    "nachos": [
        "nachos with toppings",
        "loaded nachos",
        "tortilla chips with cheese",
        "mexican nachos",
    ],
    "tamales": [
        "tamales",
        "corn husk tamales",
        "steamed tamales",
        "mexican tamales",
    ],
    "carnitas": [
        "carnitas pulled pork",
        "mexican carnitas",
        "braised pork",
        "crispy carnitas",
    ],
    "carne_asada": [
        "carne asada",
        "grilled steak mexican",
        "sliced grilled beef",
        "marinated beef",
    ],
    "chile_relleno": [
        "chile relleno",
        "stuffed pepper",
        "fried stuffed chili",
        "mexican stuffed pepper",
    ],
    "fajita": [
        "fajitas with peppers and onions",
        "sizzling fajitas",
        "chicken fajitas",
        "beef fajitas",
        "mexican fajita platter",
    ],
    "pozole": [
        "pozole soup",
        "mexican hominy soup",
        "pork pozole",
        "traditional pozole",
    ],
    "elote": [
        "elote mexican corn",
        "grilled corn with mayo",
        "street corn",
        "corn on cob with toppings",
    ],
    "churros": [
        "churros",
        "fried dough sticks",
        "cinnamon churros",
        "spanish churros",
    ],
    "tres_leches": [
        "tres leches cake",
        "three milk cake",
        "soaked sponge cake",
        "latin american cake",
    ],
    "empanadas": [
        "empanadas",
        "stuffed pastry",
        "fried empanadas",
        "meat empanadas",
    ],
    "ceviche": [
        "ceviche",
        "fish ceviche",
        "lime marinated seafood",
        "peruvian ceviche",
    ],
    "arepas": [
        "arepas",
        "corn cakes",
        "stuffed arepas",
        "venezuelan arepas",
    ],
    "pupusas": [
        "pupusas",
        "stuffed corn tortillas",
        "salvadoran pupusas",
        "thick filled tortillas",
    ],
    # === MIDDLE EASTERN CUISINE ===
    "falafel": [
        "falafel balls",
        "fried chickpea falafel",
        "falafel in pita",
        "middle eastern falafel",
    ],
    "shawarma": [
        "shawarma wrap",
        "meat shawarma",
        "sliced shawarma",
        "middle eastern shawarma",
    ],
    "kebab": [
        "kebab skewers",
        "grilled meat kebab",
        "shish kebab",
        "middle eastern kebab",
    ],
    "kofta": [
        "kofta meatballs",
        "grilled kofta",
        "spiced meat kofta",
        "middle eastern kofta",
    ],
    "baba_ganoush": [
        "baba ganoush",
        "eggplant dip",
        "smoky eggplant puree",
        "middle eastern dip",
    ],
    "tabbouleh": [
        "tabbouleh salad",
        "parsley bulgur salad",
        "lebanese tabbouleh",
        "herb salad",
    ],
    "fattoush": [
        "fattoush salad",
        "lebanese bread salad",
        "crispy pita salad",
        "middle eastern salad",
    ],
    "kibbeh": [
        "kibbeh",
        "fried bulgur meat balls",
        "stuffed kibbeh",
        "lebanese kibbeh",
    ],
    "mansaf": [
        "mansaf",
        "jordanian lamb rice",
        "lamb with yogurt sauce",
        "traditional mansaf",
    ],
    "shakshuka": [
        "shakshuka eggs",
        "eggs in tomato sauce",
        "middle eastern shakshuka",
        "poached eggs in sauce",
    ],
    "baklava": [
        "baklava pastry",
        "layered phyllo dessert",
        "honey nut baklava",
        "middle eastern sweet",
    ],
    "dolma": [
        "dolma stuffed grape leaves",
        "stuffed vine leaves",
        "grape leaves with rice",
        "turkish dolma",
    ],
    "lahmacun": [
        "lahmacun turkish pizza",
        "thin meat flatbread",
        "turkish lahmacun",
        "armenian pizza",
    ],
    "pide": [
        "turkish pide",
        "boat shaped flatbread",
        "stuffed pide",
        "turkish pizza",
    ],
    # === GREEK & MEDITERRANEAN ===
    "gyros": [
        "gyros wrap",
        "greek gyros",
        "meat in pita",
        "gyros with tzatziki",
    ],
    "souvlaki": [
        "souvlaki skewers",
        "greek meat skewers",
        "grilled souvlaki",
        "pork souvlaki",
    ],
    "moussaka": [
        "moussaka",
        "greek eggplant casserole",
        "layered moussaka",
        "baked moussaka",
    ],
    "spanakopita": [
        "spanakopita",
        "greek spinach pie",
        "phyllo spinach pastry",
        "feta spinach pie",
    ],
    "tzatziki": [
        "tzatziki sauce",
        "greek yogurt dip",
        "cucumber yogurt sauce",
        "white garlic dip",
    ],
    "greek_salad": [
        "greek salad",
        "tomato cucumber feta salad",
        "mediterranean salad",
        "horiatiki salad",
    ],
    "dolmades": [
        "dolmades grape leaves",
        "stuffed grape leaves greek",
        "rice stuffed leaves",
        "greek dolmades",
    ],
    "pastitsio": [
        "pastitsio",
        "greek pasta bake",
        "meat pasta casserole",
        "greek lasagna",
    ],
    # === ITALIAN CUISINE (EXPANDED) ===
    "spaghetti_bolognese": [
        "spaghetti bolognese",
        "pasta with meat sauce",
        "bolognese sauce pasta",
        "italian meat pasta",
    ],
    "carbonara": [
        "pasta carbonara",
        "spaghetti carbonara",
        "creamy egg pasta",
        "bacon carbonara",
    ],
    "alfredo": [
        "fettuccine alfredo",
        "creamy alfredo pasta",
        "white sauce pasta",
        "parmesan cream pasta",
    ],
    "pesto_pasta": [
        "pesto pasta",
        "basil pesto noodles",
        "green pesto spaghetti",
        "italian pesto",
    ],
    "risotto": [
        "risotto",
        "creamy italian rice",
        "mushroom risotto",
        "arborio rice dish",
    ],
    "gnocchi": [
        "gnocchi",
        "potato gnocchi",
        "italian dumplings",
        "gnocchi with sauce",
    ],
    "ravioli": [
        "ravioli pasta",
        "stuffed ravioli",
        "cheese ravioli",
        "meat ravioli",
    ],
    "minestrone": [
        "minestrone soup",
        "italian vegetable soup",
        "hearty minestrone",
        "bean vegetable soup",
    ],
    "caprese": [
        "caprese salad",
        "tomato mozzarella basil",
        "italian caprese",
        "fresh caprese",
    ],
    "bruschetta": [
        "bruschetta",
        "tomato bruschetta",
        "italian toast appetizer",
        "toasted bread with tomatoes",
    ],
    "osso_buco": [
        "osso buco",
        "braised veal shank",
        "italian osso buco",
        "slow cooked veal",
    ],
    "tiramisu": [
        "tiramisu dessert",
        "italian coffee dessert",
        "layered tiramisu",
        "mascarpone tiramisu",
    ],
    "panna_cotta": [
        "panna cotta",
        "italian cream dessert",
        "vanilla panna cotta",
        "creamy italian pudding",
    ],
    "cannoli": [
        "cannoli pastry",
        "italian cannoli",
        "cream filled pastry tube",
        "sicilian cannoli",
    ],
    "gelato": [
        "gelato ice cream",
        "italian gelato",
        "scoops of gelato",
        "creamy gelato",
    ],
    "focaccia": [
        "focaccia bread",
        "italian flatbread",
        "olive oil bread",
        "herb focaccia",
    ],
    "calzone": [
        "calzone",
        "folded pizza",
        "stuffed calzone",
        "baked calzone",
    ],
    "arancini": [
        "arancini rice balls",
        "fried rice balls",
        "italian arancini",
        "stuffed rice balls",
    ],
    "prosciutto": [
        "prosciutto ham",
        "italian cured ham",
        "sliced prosciutto",
        "prosciutto and melon",
    ],
    "antipasto": [
        "antipasto platter",
        "italian appetizer plate",
        "cured meats and cheese",
        "antipasto board",
    ],
    # === FRENCH CUISINE ===
    "croissant": [
        "croissant pastry",
        "buttery croissant",
        "french croissant",
        "flaky croissant",
    ],
    "quiche": [
        "quiche lorraine",
        "french quiche",
        "egg and cheese tart",
        "savory quiche",
    ],
    "ratatouille": [
        "ratatouille",
        "french vegetable stew",
        "provencal ratatouille",
        "layered vegetables",
    ],
    "coq_au_vin": [
        "coq au vin",
        "french chicken in wine",
        "braised chicken",
        "chicken wine stew",
    ],
    "beef_bourguignon": [
        "beef bourguignon",
        "french beef stew",
        "burgundy beef",
        "red wine beef stew",
    ],
    "french_onion_soup": [
        "french onion soup",
        "onion soup with cheese",
        "gratineed onion soup",
        "caramelized onion soup",
    ],
    "crepes": [
        "french crepes",
        "thin pancakes",
        "filled crepes",
        "sweet crepes",
    ],
    "souffle": [
        "souffle",
        "french souffle",
        "chocolate souffle",
        "fluffy baked souffle",
    ],
    "eclairs": [
        "eclairs",
        "chocolate eclairs",
        "french pastry",
        "cream filled pastry",
    ],
    "macarons": [
        "french macarons",
        "colorful macarons",
        "sandwich cookies",
        "almond meringue cookies",
    ],
    "creme_brulee": [
        "creme brulee",
        "burnt cream dessert",
        "caramelized custard",
        "french custard dessert",
    ],
    "baguette": [
        "french baguette",
        "long bread loaf",
        "crusty baguette",
        "french bread",
    ],
    # === AMERICAN & COMFORT FOOD ===
    "mac_and_cheese": [
        "mac and cheese",
        "macaroni and cheese",
        "cheesy pasta",
        "creamy mac cheese",
    ],
    "grilled_cheese": [
        "grilled cheese sandwich",
        "toasted cheese sandwich",
        "melted cheese sandwich",
        "crispy grilled cheese",
    ],
    "meatball_sub": [
        "meatball sub",
        "meatball sandwich",
        "italian sub",
        "marinara meatball sub",
    ],
    "philly_cheesesteak": [
        "philly cheesesteak",
        "cheese steak sandwich",
        "sliced beef sandwich",
        "philadelphia cheesesteak",
    ],
    "pulled_pork": [
        "pulled pork sandwich",
        "bbq pulled pork",
        "shredded pork",
        "smoked pulled pork",
    ],
    "bbq_ribs": [
        "bbq ribs",
        "barbecue spare ribs",
        "grilled ribs",
        "smoked pork ribs",
    ],
    "brisket": [
        "smoked brisket",
        "beef brisket",
        "sliced brisket",
        "bbq brisket",
    ],
    "corn_dog": [
        "corn dog",
        "battered hot dog",
        "fried corn dog",
        "corn battered frank",
    ],
    "clam_chowder": [
        "clam chowder",
        "new england chowder",
        "creamy clam soup",
        "white clam chowder",
    ],
    "chili": [
        "chili con carne",
        "beef chili",
        "bowl of chili",
        "texas chili",
    ],
    "pot_roast": [
        "pot roast",
        "braised beef roast",
        "slow cooked roast",
        "sunday pot roast",
    ],
    "chicken_pot_pie": [
        "chicken pot pie",
        "pot pie",
        "creamy chicken pie",
        "baked pot pie",
    ],
    "shepherd_pie": [
        "shepherds pie",
        "cottage pie",
        "meat pie with mashed potato",
        "ground meat pie",
    ],
    "biscuits_and_gravy": [
        "biscuits and gravy",
        "southern breakfast",
        "sausage gravy biscuits",
        "white gravy biscuits",
    ],
    "chicken_and_waffles": [
        "chicken and waffles",
        "fried chicken with waffle",
        "southern chicken waffles",
        "crispy chicken waffle",
    ],
    "jambalaya": [
        "jambalaya",
        "cajun rice dish",
        "louisiana jambalaya",
        "sausage shrimp rice",
    ],
    "gumbo": [
        "gumbo stew",
        "cajun gumbo",
        "louisiana gumbo",
        "okra seafood stew",
    ],
    "po_boy": [
        "po boy sandwich",
        "fried shrimp po boy",
        "louisiana sandwich",
        "new orleans po boy",
    ],
    "cobb_salad": [
        "cobb salad",
        "american cobb salad",
        "salad with bacon and egg",
        "chopped cobb salad",
    ],
    "caesar_salad": [
        "caesar salad",
        "romaine caesar salad",
        "parmesan caesar",
        "classic caesar salad",
    ],
    "buffalo_wings": [
        "buffalo wings",
        "spicy chicken wings",
        "buffalo style wings",
        "hot wings with sauce",
    ],
    "onion_rings": [
        "onion rings",
        "fried onion rings",
        "crispy onion rings",
        "battered onion rings",
    ],
    "mozzarella_sticks": [
        "mozzarella sticks",
        "fried cheese sticks",
        "breaded mozzarella",
        "cheese sticks appetizer",
    ],
    "jalapeno_poppers": [
        "jalapeno poppers",
        "stuffed jalapenos",
        "fried jalapeno peppers",
        "cream cheese jalapenos",
    ],
    "loaded_potato_skins": [
        "loaded potato skins",
        "stuffed potato skins",
        "bacon cheese potato skins",
        "appetizer potato skins",
    ],
    # === BREAKFAST FOODS (EXPANDED) ===
    "eggs_benedict": [
        "eggs benedict",
        "poached eggs on muffin",
        "hollandaise eggs",
        "benedict with ham",
    ],
    "french_toast": [
        "french toast",
        "brioche french toast",
        "eggy bread",
        "cinnamon french toast",
    ],
    "omelette": [
        "omelette",
        "folded egg omelette",
        "cheese omelette",
        "vegetable omelette",
    ],
    "breakfast_burrito": [
        "breakfast burrito",
        "egg burrito",
        "morning burrito wrap",
        "scrambled egg wrap",
    ],
    "breakfast_sandwich": [
        "breakfast sandwich",
        "egg muffin sandwich",
        "bacon egg cheese sandwich",
        "morning sandwich",
    ],
    "avocado_toast": [
        "avocado toast",
        "smashed avocado on bread",
        "avo toast",
        "avocado on sourdough",
    ],
    "acai_bowl": [
        "acai bowl",
        "purple smoothie bowl",
        "acai berry bowl",
        "fruit topped bowl",
    ],
    "smoothie_bowl": [
        "smoothie bowl",
        "thick smoothie bowl",
        "fruit smoothie bowl",
        "blended fruit bowl",
    ],
    "fruit_salad": [
        "fruit salad",
        "mixed fresh fruit",
        "fruit bowl",
        "chopped fruit medley",
    ],
    "eggs_over_easy": [
        "eggs over easy",
        "fried eggs",
        "runny yolk eggs",
        "sunny side up eggs",
    ],
    "scrambled_eggs": [
        "scrambled eggs",
        "fluffy scrambled eggs",
        "soft scrambled eggs",
        "breakfast eggs scrambled",
    ],
    "hard_boiled_eggs": [
        "hard boiled eggs",
        "boiled eggs sliced",
        "peeled boiled eggs",
        "egg halves",
    ],
    "sausage_links": [
        "breakfast sausages",
        "sausage links",
        "pork sausage",
        "breakfast links",
    ],
    # === SANDWICHES & WRAPS ===
    "club_sandwich": [
        "club sandwich",
        "triple decker sandwich",
        "turkey club",
        "stacked club sandwich",
    ],
    "blt_sandwich": [
        "blt sandwich",
        "bacon lettuce tomato",
        "classic blt",
        "toasted blt",
    ],
    "reuben_sandwich": [
        "reuben sandwich",
        "corned beef sandwich",
        "sauerkraut sandwich",
        "deli reuben",
    ],
    "cuban_sandwich": [
        "cuban sandwich",
        "cubano pressed sandwich",
        "pork cuban sandwich",
        "pressed cuban",
    ],
    "wrap": [
        "tortilla wrap",
        "chicken wrap",
        "vegetable wrap",
        "flour tortilla wrap",
    ],
    "pita_sandwich": [
        "pita sandwich",
        "stuffed pita pocket",
        "falafel pita",
        "pita bread sandwich",
    ],
    "panini": [
        "panini sandwich",
        "grilled panini",
        "pressed panini",
        "italian panini",
    ],
    "bagel_sandwich": [
        "bagel sandwich",
        "stuffed bagel",
        "breakfast bagel",
        "deli bagel sandwich",
    ],
    # === SEAFOOD (EXPANDED) ===
    "lobster": [
        "whole lobster",
        "steamed lobster",
        "lobster tail",
        "red lobster dinner",
    ],
    "lobster_roll": [
        "lobster roll",
        "maine lobster roll",
        "lobster sandwich",
        "buttered lobster roll",
    ],
    "crab": [
        "crab legs",
        "steamed crab",
        "crab meat",
        "dungeness crab",
    ],
    "crab_cakes": [
        "crab cakes",
        "maryland crab cakes",
        "pan fried crab cakes",
        "lump crab patties",
    ],
    "fish_tacos": [
        "fish tacos",
        "battered fish tacos",
        "grilled fish tacos",
        "baja fish tacos",
    ],
    "grilled_salmon": [
        "grilled salmon fillet",
        "pan seared salmon",
        "baked salmon",
        "salmon steak",
    ],
    "fried_shrimp": [
        "fried shrimp",
        "breaded shrimp",
        "popcorn shrimp",
        "crispy fried shrimp",
    ],
    "shrimp_scampi": [
        "shrimp scampi",
        "garlic butter shrimp",
        "shrimp pasta",
        "lemon shrimp scampi",
    ],
    "calamari": [
        "fried calamari",
        "calamari rings",
        "crispy squid",
        "battered calamari",
    ],
    "mussels": [
        "steamed mussels",
        "mussels in broth",
        "garlic mussels",
        "moules frites",
    ],
    "oysters": [
        "raw oysters",
        "oysters on half shell",
        "fresh oysters",
        "oyster platter",
    ],
    "scallops": [
        "seared scallops",
        "pan seared scallops",
        "grilled scallops",
        "scallop dish",
    ],
    "clams": [
        "steamed clams",
        "clams in shell",
        "littleneck clams",
        "clam dish",
    ],
    # === SOUPS (EXPANDED) ===
    "chicken_noodle_soup": [
        "chicken noodle soup",
        "chicken soup with noodles",
        "homemade chicken soup",
        "classic chicken noodle",
    ],
    "tomato_soup": [
        "tomato soup",
        "creamy tomato soup",
        "red tomato soup",
        "tomato bisque",
    ],
    "split_pea_soup": [
        "split pea soup",
        "green pea soup",
        "ham and pea soup",
        "thick pea soup",
    ],
    "butternut_squash_soup": [
        "butternut squash soup",
        "orange squash soup",
        "creamy squash soup",
        "roasted squash soup",
    ],
    "broccoli_cheddar_soup": [
        "broccoli cheddar soup",
        "cheesy broccoli soup",
        "creamy broccoli soup",
        "broccoli cheese soup",
    ],
    "mushroom_soup": [
        "cream of mushroom soup",
        "mushroom soup",
        "creamy mushroom soup",
        "wild mushroom soup",
    ],
    "potato_soup": [
        "potato soup",
        "creamy potato soup",
        "loaded potato soup",
        "baked potato soup",
    ],
    "corn_chowder": [
        "corn chowder",
        "creamy corn soup",
        "sweet corn chowder",
        "corn potato chowder",
    ],
    "lentil_soup": [
        "lentil soup",
        "red lentil soup",
        "hearty lentil soup",
        "vegetable lentil soup",
    ],
    "gazpacho": [
        "gazpacho",
        "cold tomato soup",
        "spanish gazpacho",
        "chilled vegetable soup",
    ],
    "wonton_soup": [
        "wonton soup",
        "dumpling soup",
        "chinese wonton soup",
        "pork wonton soup",
    ],
    "egg_drop_soup": [
        "egg drop soup",
        "chinese egg soup",
        "egg flower soup",
        "silky egg soup",
    ],
    "hot_and_sour_soup": [
        "hot and sour soup",
        "spicy sour chinese soup",
        "szechuan soup",
        "tofu mushroom soup",
    ],
    # === DESSERTS (EXPANDED) ===
    "cheesecake": [
        "cheesecake slice",
        "new york cheesecake",
        "creamy cheesecake",
        "strawberry cheesecake",
    ],
    "brownies": [
        "chocolate brownies",
        "fudge brownies",
        "walnut brownies",
        "brownie squares",
    ],
    "cupcakes": [
        "cupcakes",
        "frosted cupcakes",
        "decorated cupcakes",
        "vanilla cupcakes",
    ],
    "apple_pie": [
        "apple pie slice",
        "homemade apple pie",
        "lattice apple pie",
        "warm apple pie",
    ],
    "pumpkin_pie": [
        "pumpkin pie",
        "pumpkin pie slice",
        "thanksgiving pie",
        "spiced pumpkin pie",
    ],
    "key_lime_pie": [
        "key lime pie",
        "lime pie slice",
        "florida key lime pie",
        "tangy lime pie",
    ],
    "banana_split": [
        "banana split",
        "ice cream banana split",
        "sundae banana split",
        "classic banana split",
    ],
    "ice_cream_sundae": [
        "ice cream sundae",
        "hot fudge sundae",
        "sundae with toppings",
        "chocolate sundae",
    ],
    "milkshake": [
        "milkshake",
        "chocolate milkshake",
        "vanilla milkshake",
        "thick milkshake",
    ],
    "pudding": [
        "pudding dessert",
        "chocolate pudding",
        "vanilla pudding",
        "creamy pudding cup",
    ],
    "bread_pudding": [
        "bread pudding",
        "warm bread pudding",
        "raisin bread pudding",
        "dessert bread pudding",
    ],
    "rice_pudding": [
        "rice pudding",
        "creamy rice pudding",
        "cinnamon rice pudding",
        "sweet rice pudding",
    ],
    "flan": [
        "flan dessert",
        "caramel flan",
        "spanish flan",
        "cream caramel custard",
    ],
    "fruit_tart": [
        "fruit tart",
        "fresh fruit tart",
        "berry tart",
        "pastry fruit tart",
    ],
    "chocolate_cake": [
        "chocolate cake slice",
        "layer chocolate cake",
        "rich chocolate cake",
        "dark chocolate cake",
    ],
    "carrot_cake": [
        "carrot cake",
        "cream cheese frosted cake",
        "spiced carrot cake",
        "layered carrot cake",
    ],
    "red_velvet_cake": [
        "red velvet cake",
        "red velvet slice",
        "cream cheese red velvet",
        "crimson red cake",
    ],
    "pound_cake": [
        "pound cake",
        "vanilla pound cake",
        "loaf pound cake",
        "buttery pound cake",
    ],
    "angel_food_cake": [
        "angel food cake",
        "white sponge cake",
        "light angel cake",
        "fluffy angel food",
    ],
    "cinnamon_roll": [
        "cinnamon roll",
        "iced cinnamon roll",
        "glazed cinnamon bun",
        "swirled cinnamon roll",
    ],
    "danish_pastry": [
        "danish pastry",
        "fruit danish",
        "cheese danish",
        "flaky danish",
    ],
    "scone": [
        "scone",
        "british scone",
        "blueberry scone",
        "cream scone",
    ],
    # === FAST FOOD ===
    "cheeseburger": [
        "cheeseburger",
        "double cheeseburger",
        "fast food burger",
        "cheese hamburger",
    ],
    "chicken_sandwich": [
        "chicken sandwich",
        "crispy chicken sandwich",
        "fried chicken sandwich",
        "grilled chicken sandwich",
    ],
    "fish_sandwich": [
        "fish sandwich",
        "filet o fish",
        "fried fish sandwich",
        "tartar sauce fish",
    ],
    "chicken_tenders": [
        "chicken tenders",
        "chicken strips",
        "breaded chicken tenders",
        "crispy tenders",
    ],
    "onion_petals": [
        "blooming onion",
        "onion blossom",
        "fried onion petals",
        "outback onion",
    ],
    "soft_pretzel": [
        "soft pretzel",
        "warm soft pretzel",
        "twisted pretzel",
        "mall pretzel",
    ],
    # === BEVERAGES (EXPANDED) ===
    "espresso": [
        "espresso shot",
        "espresso coffee",
        "italian espresso",
        "double espresso",
    ],
    "cappuccino": [
        "cappuccino",
        "foamy cappuccino",
        "coffee cappuccino",
        "italian cappuccino",
    ],
    "latte": [
        "latte coffee",
        "cafe latte",
        "milk coffee latte",
        "creamy latte",
    ],
    "mocha": [
        "mocha coffee",
        "chocolate mocha",
        "cafe mocha",
        "mocha latte",
    ],
    "iced_coffee": [
        "iced coffee",
        "cold brew coffee",
        "chilled coffee",
        "coffee over ice",
    ],
    "bubble_tea": [
        "bubble tea",
        "boba tea",
        "tapioca milk tea",
        "pearl milk tea",
    ],
    "hot_chocolate": [
        "hot chocolate",
        "cocoa drink",
        "hot cocoa",
        "chocolate drink",
    ],
    "lemonade": [
        "lemonade",
        "fresh lemonade",
        "yellow lemonade",
        "icy lemonade",
    ],
    "margarita": [
        "margarita cocktail",
        "frozen margarita",
        "lime margarita",
        "salted rim margarita",
    ],
    "mojito": [
        "mojito cocktail",
        "mint mojito",
        "cuban mojito",
        "lime mint drink",
    ],
    "sangria": [
        "sangria wine",
        "red sangria",
        "fruit sangria",
        "spanish sangria",
    ],
    "pina_colada": [
        "pina colada",
        "coconut cocktail",
        "tropical pina colada",
        "frozen pina colada",
    ],
    # === ADDITIONAL PROTEINS ===
    "chicken breast": [
        "grilled chicken breast",
        "cooked chicken breast",
        "boneless chicken breast",
        "sliced chicken breast",
    ],
    "chicken thigh": [
        "chicken thigh meat",
        "cooked chicken thigh",
        "grilled chicken thigh",
        "bone-in chicken thigh",
    ],
    "turkey": [
        "roasted turkey meat",
        "sliced turkey",
        "turkey breast",
        "thanksgiving turkey",
    ],
    "ham": [
        "sliced ham",
        "deli ham",
        "cooked ham",
        "honey glazed ham",
    ],
    "steak": [
        "grilled steak",
        "beef steak",
        "ribeye steak",
        "sirloin steak",
    ],
    "cod": [
        "cod fish fillet",
        "baked cod",
        "pan seared cod",
        "atlantic cod",
    ],
    "tilapia": [
        "tilapia fillet",
        "baked tilapia",
        "grilled tilapia",
        "pan fried tilapia",
    ],
    "oyster": [
        "fresh oysters",
        "raw oysters on half shell",
        "oyster seafood",
    ],
    "tempeh": [
        "sliced tempeh",
        "fried tempeh",
        "tempeh protein",
        "indonesian tempeh",
    ],
    "egg": [
        "a whole egg",
        "raw egg",
        "cooked egg",
        "chicken egg",
    ],
    "meatball": [
        "a meatball",
        "beef meatball",
        "italian meatball",
        "meatball in sauce",
    ],
    # === ADDITIONAL BEANS & LEGUMES ===
    "black beans": [
        "black beans",
        "cooked black beans",
        "black bean bowl",
    ],
    "kidney beans": [
        "kidney beans",
        "red kidney beans",
        "cooked kidney beans",
    ],
    "pinto beans": [
        "pinto beans",
        "cooked pinto beans",
        "refried pinto beans",
    ],
    "green beans": [
        "green beans",
        "steamed green beans",
        "fresh green beans",
        "string beans",
    ],
    "lentils": [
        "cooked lentils",
        "lentil soup",
        "brown lentils",
        "red lentils",
    ],
    # === ADDITIONAL NUTS & SEEDS ===
    "peanut": [
        "peanuts",
        "roasted peanut",
        "shelled peanuts",
    ],
    "walnut": [
        "walnut halves",
        "shelled walnuts",
        "walnut pieces",
    ],
    "pecans": [
        "pecan nuts",
        "shelled pecans",
        "roasted pecans",
    ],
    "hazelnuts": [
        "hazelnuts",
        "filberts",
        "shelled hazelnuts",
    ],
    "brazil nuts": [
        "brazil nuts",
        "shelled brazil nuts",
    ],
    "macadamia nuts": [
        "macadamia nuts",
        "roasted macadamias",
    ],
    "pine nuts": [
        "pine nuts",
        "pignoli nuts",
        "toasted pine nuts",
    ],
    "chestnuts": [
        "chestnuts",
        "roasted chestnuts",
        "chestnut nuts",
    ],
    "pistachios": [
        "pistachio nuts",
        "shelled pistachios",
        "green pistachios",
    ],
    "pumpkin seeds": [
        "pumpkin seeds",
        "pepitas",
        "roasted pumpkin seeds",
    ],
    "sesame seeds": [
        "sesame seeds",
        "white sesame seeds",
        "black sesame seeds",
    ],
    "hemp seeds": [
        "hemp seeds",
        "hemp hearts",
        "shelled hemp seeds",
    ],
    # === ADDITIONAL DAIRY ===
    "cottage cheese": [
        "cottage cheese",
        "creamy cottage cheese",
        "cottage cheese in bowl",
    ],
    "cream cheese": [
        "cream cheese",
        "cream cheese spread",
        "philadelphia cream cheese",
    ],
    # === ADDITIONAL VEGETABLES ===
    "kale": [
        "fresh kale leaves",
        "green kale",
        "curly kale",
        "kale salad",
    ],
    "sweet potato": [
        "sweet potato",
        "baked sweet potato",
        "orange sweet potato",
        "yam",
    ],
    # === ADDITIONAL GRAINS & PASTAS ===
    "oatmeal": [
        "bowl of oatmeal",
        "hot oatmeal",
        "oatmeal porridge",
        "breakfast oatmeal",
    ],
    "macaroni": [
        "macaroni pasta",
        "elbow macaroni",
        "cooked macaroni",
    ],
    "noodle": [
        "noodles",
        "asian noodles",
        "egg noodles",
        "ramen noodles",
    ],
    "tortilla": [
        "flour tortilla",
        "corn tortilla",
        "flat tortilla bread",
    ],
    "pretzel": [
        "soft pretzel",
        "pretzel snack",
        "twisted pretzel",
    ],
    "granola bar": [
        "granola bar",
        "oat granola bar",
        "chewy granola bar",
    ],
    # === ADDITIONAL PREPARED DISHES ===
    "dumpling": [
        "dumplings",
        "asian dumplings",
        "steamed dumplings",
        "pot stickers",
    ],
    "spring_roll": [
        "spring rolls",
        "vietnamese spring roll",
        "fried spring roll",
        "fresh spring roll",
    ],
    "nacho": [
        "nachos",
        "loaded nachos",
        "nachos with cheese",
        "tortilla chips with toppings",
    ],
    "stir fry": [
        "stir fry dish",
        "vegetable stir fry",
        "asian stir fry",
        "wok fried",
    ],
    "cookie": [
        "a cookie",
        "chocolate chip cookie",
        "baked cookie",
        "round cookie",
    ],
    # === CONDIMENTS & SPREADS ===
    "peanut butter": [
        "peanut butter",
        "creamy peanut butter",
        "peanut butter jar",
    ],
    "ketchup": [
        "ketchup",
        "tomato ketchup",
        "red ketchup sauce",
    ],
    "mustard": [
        "yellow mustard",
        "mustard sauce",
        "dijon mustard",
    ],
    "mayonnaise": [
        "mayonnaise",
        "white mayo",
        "creamy mayonnaise",
    ],
    "soy sauce": [
        "soy sauce",
        "dark soy sauce",
        "soy sauce bottle",
    ],
    "maple syrup": [
        "maple syrup",
        "pancake syrup",
        "golden maple syrup",
    ],
    # === SPACE-VERSION ALIASES (for database compatibility) ===
    # These reference foods that have underscore versions in prompts
    "bell pepper": [
        "a colorful bell pepper vegetable",
        "red bell pepper on cutting board",
        "green bell pepper whole",
        "yellow bell pepper sliced",
    ],
    "chia seeds": ["chia seeds", "black chia seeds"],
    "flax seeds": ["flaxseeds", "ground flaxseed", "flax seeds"],
    "ice cream": ["ice cream scoop", "ice cream cone", "vanilla ice cream"],
    "mixed nuts": ["mixed nuts", "assorted nuts", "variety of nuts"],
    "olive oil": ["olive oil bottle", "extra virgin olive oil"],
    "sunflower seeds": ["sunflower seeds", "shelled sunflower seeds"],
    # === UNKNOWN/CATCH-ALL ===
    "unknown": [
        "unidentified food",
        "unclear food item",
        "ambiguous dish",
        "unrecognized meal",
    ],
}

# Simplified prompts for faster matching
SIMPLE_FOOD_PROMPTS: Dict[str, str] = {
    key: f"a photo of {key.replace('_', ' ')}" for key in FOOD_PROMPTS.keys()
}

# =============================================================================
# MODEL CONFIGURATION PRESETS
# =============================================================================
# Available CLIP models with trade-offs between speed and accuracy

CLIP_MODELS = {
    # Fast model - good for real-time classification on mobile
    "fast": {
        "name": "openai/clip-vit-base-patch32",
        "description": "Fast model (400MB) - Good balance of speed and accuracy",
        "memory_mb": 400,
        "relative_speed": 1.0,
        "relative_accuracy": 0.85,
    },
    # Standard model - better accuracy, still reasonable speed
    "standard": {
        "name": "openai/clip-vit-base-patch16",
        "description": "Standard model (600MB) - Better accuracy, moderate speed",
        "memory_mb": 600,
        "relative_speed": 0.7,
        "relative_accuracy": 0.92,
    },
    # High accuracy model - best accuracy but slower and uses more memory
    "accurate": {
        "name": "openai/clip-vit-large-patch14",
        "description": "High accuracy model (1.7GB) - Best accuracy, slower",
        "memory_mb": 1700,
        "relative_speed": 0.4,
        "relative_accuracy": 1.0,
    },
}

# Default model preset
DEFAULT_MODEL_PRESET = "fast"

# ==============================================================================
# TEXT FEATURES CACHE CONFIGURATION
# ==============================================================================
# Cache pre-computed text features to disk to avoid recomputing on every restart.
# This reduces startup time from ~7 minutes to <1 second.

# Cache directory (uses HF_HOME if set, otherwise ~/.cache/nutri)
TEXT_FEATURES_CACHE_DIR = (
    Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/nutri")))
    / "clip_text_features"
)

# Batch size for text encoding (32 is efficient for most GPUs/CPUs)
TEXT_ENCODING_BATCH_SIZE = 32


def _compute_prompts_hash(prompts_dict: Dict[str, List[str]], model_name: str) -> str:
    """Compute a hash of the prompts dictionary and model name for cache invalidation."""
    # Sort for deterministic ordering
    content = json.dumps(prompts_dict, sort_keys=True) + model_name
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _get_cache_path(model_name: str, use_detailed: bool) -> Path:
    """Get the cache file path for text features."""
    # Clean model name for filename
    safe_model_name = model_name.replace("/", "_").replace("-", "_")
    prompts_dict = (
        FOOD_PROMPTS
        if use_detailed
        else {k: [v] for k, v in SIMPLE_FOOD_PROMPTS.items()}
    )
    prompts_hash = _compute_prompts_hash(prompts_dict, model_name)
    mode = "detailed" if use_detailed else "simple"
    return TEXT_FEATURES_CACHE_DIR / f"{safe_model_name}_{mode}_{prompts_hash}.pt"


# ==============================================================================
# HIERARCHICAL FOOD CLASSIFICATION
# ==============================================================================
# Maps broad cuisine/food categories to their specific dishes.
# Used for two-stage classification: first detect cuisine, then narrow to dish.

CUISINE_CATEGORIES: Dict[str, Dict[str, List[str]]] = {
    "japanese": {
        "prompts": [
            "Japanese food",
            "Japanese cuisine dish",
            "traditional Japanese meal",
        ],
        "dishes": [
            "sushi",
            "sashimi",
            "ramen",
            "tempura",
            "tonkatsu",
            "gyoza",
            "udon",
            "soba",
            "yakitori",
            "onigiri",
            "miso_soup",
            "okonomiyaki",
            "takoyaki",
            "katsu_curry",
            "donburi",
            "edamame",
            "mochi",
            "matcha",
            "teriyaki",
        ],
    },
    "chinese": {
        "prompts": [
            "Chinese food",
            "Chinese cuisine dish",
            "traditional Chinese meal",
        ],
        "dishes": [
            "fried_rice",
            "chow_mein",
            "kung_pao_chicken",
            "orange_chicken",
            "general_tso_chicken",
            "sweet_and_sour_pork",
            "beef_and_broccoli",
            "mongolian_beef",
            "sesame_chicken",
            "lo_mein",
            "egg_foo_young",
            "mapo_tofu",
            "peking_duck",
            "dim_sum",
            "char_siu",
            "wonton",
            "congee",
            "hot_pot",
            "scallion_pancakes",
            "dan_dan_noodles",
            "dumplings",
            "spring_rolls",
        ],
    },
    "korean": {
        "prompts": [
            "Korean food",
            "Korean cuisine dish",
            "traditional Korean meal",
        ],
        "dishes": [
            "bibimbap",
            "bulgogi",
            "korean_fried_chicken",
            "kimchi",
            "japchae",
            "tteokbokki",
            "kimbap",
            "samgyeopsal",
            "sundubu_jjigae",
            "kimchi_jjigae",
            "korean_bbq",
            "army_stew",
            "pajeon",
        ],
    },
    "thai": {
        "prompts": [
            "Thai food",
            "Thai cuisine dish",
            "traditional Thai meal",
        ],
        "dishes": [
            "pad_thai",
            "green_curry",
            "red_curry",
            "massaman_curry",
            "tom_yum",
            "tom_kha",
            "thai_basil_chicken",
            "larb",
            "satay",
            "mango_sticky_rice",
            "papaya_salad",
            "thai_iced_tea",
        ],
    },
    "vietnamese": {
        "prompts": [
            "Vietnamese food",
            "Vietnamese cuisine dish",
            "traditional Vietnamese meal",
        ],
        "dishes": [
            "pho",
            "banh_mi",
            "bun_cha",
            "cao_lau",
            "com_tam",
            "goi_cuon",
            "bun_bo_hue",
        ],
    },
    "indian": {
        "prompts": [
            "Indian food",
            "Indian cuisine dish",
            "traditional Indian meal",
        ],
        "dishes": [
            "butter_chicken",
            "tikka_masala",
            "biryani",
            "samosa",
            "naan",
            "tandoori_chicken",
            "dal",
            "palak_paneer",
            "paneer_tikka",
            "chana_masala",
            "korma",
            "vindaloo",
            "rogan_josh",
            "aloo_gobi",
            "dosa",
            "idli",
            "pakora",
            "raita",
            "lassi",
            "gulab_jamun",
            "jalebi",
        ],
    },
    "mexican": {
        "prompts": [
            "Mexican food",
            "Mexican cuisine dish",
            "traditional Mexican meal",
        ],
        "dishes": [
            "taco",
            "burrito",
            "quesadilla",
            "enchiladas",
            "nachos",
            "tamales",
            "carnitas",
            "carne_asada",
            "chile_relleno",
            "churros",
            "fajita",
            "guacamole",
        ],
    },
    "middle_eastern": {
        "prompts": [
            "Middle Eastern food",
            "Mediterranean cuisine dish",
            "traditional Middle Eastern meal",
        ],
        "dishes": [
            "falafel",
            "shawarma",
            "hummus",
            "kebab",
            "kofta",
            "baba_ganoush",
            "tabbouleh",
            "fattoush",
            "shakshuka",
            "baklava",
            "dolma",
            "gyros",
        ],
    },
    "italian": {
        "prompts": [
            "Italian food",
            "Italian cuisine dish",
            "traditional Italian meal",
        ],
        "dishes": [
            "pizza",
            "pasta",
            "lasagna",
            "carbonara",
            "risotto",
            "gnocchi",
            "spaghetti_bolognese",
            "alfredo",
            "ravioli",
            "minestrone",
            "tiramisu",
            "gelato",
            "bruschetta",
            "caprese",
        ],
    },
    "american": {
        "prompts": [
            "American food",
            "American cuisine dish",
            "comfort food",
        ],
        "dishes": [
            "burger",
            "hot_dog",
            "pizza",
            "mac_and_cheese",
            "grilled_cheese",
            "bbq_ribs",
            "pulled_pork",
            "cheeseburger",
            "chicken_sandwich",
            "chicken_tenders",
            "buffalo_wings",
            "onion_rings",
            "mozzarella_sticks",
            "clam_chowder",
            "chili",
            "chicken_pot_pie",
            "shepherd_pie",
            "pot_roast",
            "brisket",
            "jambalaya",
            "gumbo",
            "corn_dog",
        ],
    },
    "breakfast": {
        "prompts": [
            "breakfast food",
            "morning meal",
            "breakfast dish",
        ],
        "dishes": [
            "pancake",
            "waffle",
            "eggs",
            "omelette",
            "scrambled_eggs",
            "eggs_benedict",
            "french_toast",
            "bacon",
            "sausage_links",
            "avocado_toast",
            "acai_bowl",
            "smoothie_bowl",
            "fruit_salad",
            "biscuits_and_gravy",
            "chicken_and_waffles",
            "granola",
            "oats",
            "toast",
        ],
    },
    "dessert": {
        "prompts": [
            "dessert",
            "sweet treat",
            "cake or pastry",
        ],
        "dishes": [
            "cake",
            "ice_cream",
            "cheesecake",
            "brownies",
            "cookies",
            "chocolate_cake",
            "apple_pie",
            "cinnamon_roll",
            "crepes",
            "milkshake",
            "pumpkin_pie",
            "chocolate",
            "tiramisu",
            "gelato",
            "baklava",
            "mochi",
            "churros",
            "gulab_jamun",
            "jalebi",
        ],
    },
    "sandwich": {
        "prompts": [
            "sandwich",
            "sub sandwich",
            "deli sandwich",
        ],
        "dishes": [
            "blt_sandwich",
            "club_sandwich",
            "cuban_sandwich",
            "reuben_sandwich",
            "philly_cheesesteak",
            "meatball_sub",
            "wrap",
            "panini",
            "grilled_cheese",
            "chicken_sandwich",
            "burger",
            "cheeseburger",
        ],
    },
    "seafood": {
        "prompts": [
            "seafood dish",
            "fish meal",
            "shellfish plate",
        ],
        "dishes": [
            "salmon",
            "tuna",
            "shrimp",
            "fish",
            "sushi",
            "sashimi",
            "grilled_salmon",
            "fish_tacos",
            "fried_shrimp",
            "shrimp_scampi",
            "lobster_roll",
            "crab_cakes",
            "calamari",
            "mussels",
            "oysters",
            "clams",
            "ceviche",
        ],
    },
    "soup": {
        "prompts": [
            "soup",
            "bowl of soup",
            "warm soup dish",
        ],
        "dishes": [
            "chicken_noodle_soup",
            "tomato_soup",
            "lentil_soup",
            "wonton_soup",
            "egg_drop_soup",
            "hot_and_sour_soup",
            "miso_soup",
            "ramen",
            "pho",
            "tom_yum",
            "minestrone",
            "clam_chowder",
            "gumbo",
        ],
    },
}

# Build reverse lookup: dish -> cuisine
DISH_TO_CUISINE: Dict[str, str] = {}
for cuisine, data in CUISINE_CATEGORIES.items():
    for dish in data["dishes"]:
        if dish not in DISH_TO_CUISINE:  # First assignment wins
            DISH_TO_CUISINE[dish] = cuisine


# =============================================================================
# PROMPT TEMPLATES - Proven technique from CLIP research
# Using diverse templates significantly improves zero-shot accuracy
# OpenAI's original CLIP paper used 80 templates for ImageNet
# =============================================================================

FOOD_PROMPT_TEMPLATES: List[str] = [
    # Basic photo descriptions
    "a photo of {food}",
    "a picture of {food}",
    "an image of {food}",
    # Presentation context
    "a photo of {food} on a plate",
    "a photo of {food} in a bowl",
    "a photo of {food} on a table",
    "{food} served on a dish",
    "a serving of {food}",
    # Quality descriptors
    "a close-up photo of {food}",
    "a bright photo of {food}",
    "a good photo of {food}",
    # Food-specific contexts
    "a restaurant photo of {food}",
    "homemade {food}",
    "freshly prepared {food}",
    "delicious {food}",
    # Cooking states (for applicable foods)
    "cooked {food}",
    "{food} dish",
    "{food} meal",
]

# Templates specifically for raw ingredients
RAW_INGREDIENT_TEMPLATES: List[str] = [
    "fresh {food}",
    "raw {food}",
    "{food} on cutting board",
    "whole {food}",
    "organic {food}",
]

# Templates for prepared/cooked dishes
PREPARED_DISH_TEMPLATES: List[str] = [
    "a plate of {food}",
    "{food} ready to eat",
    "{food} from restaurant",
    "takeout {food}",
    "a portion of {food}",
]


def get_clip_model_name(preset: str = DEFAULT_MODEL_PRESET) -> str:
    """Get CLIP model name from preset.

    Args:
        preset: One of 'fast', 'standard', or 'accurate'

    Returns:
        HuggingFace model name
    """
    if preset not in CLIP_MODELS:
        logger.warning(f"Unknown preset '{preset}', using '{DEFAULT_MODEL_PRESET}'")
        preset = DEFAULT_MODEL_PRESET
    return CLIP_MODELS[preset]["name"]


class CLIPFoodClassifier:
    """
    Zero-shot food classifier using OpenAI's CLIP model.

    CLIP compares images to text descriptions, allowing classification
    of any food category without specific training.

    Supports 395+ food categories including:
    - International cuisines (Chinese, Japanese, Korean, Thai, Vietnamese,
      Indian, Mexican, Middle Eastern, Greek, Italian, French)
    - American comfort food and fast food
    - Breakfast items, sandwiches, soups
    - Seafood, proteins, vegetables, fruits
    - Desserts and beverages
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_preset: str = DEFAULT_MODEL_PRESET,
        device: Optional[str] = None,
        use_detailed_prompts: bool = True,
    ):
        """
        Initialize CLIP food classifier.

        Args:
            model_name: Explicit CLIP model name (overrides preset if provided).
                - "openai/clip-vit-base-patch32" (fast, 400MB)
                - "openai/clip-vit-base-patch16" (standard, 600MB)
                - "openai/clip-vit-large-patch14" (accurate, 1.7GB)
            model_preset: Model preset if model_name not specified.
                - "fast": Quick classification, good for real-time use
                - "standard": Better accuracy, moderate speed
                - "accurate": Best accuracy, requires more memory/time
            device: Device for inference ('cuda', 'mps', 'cpu', or None for auto)
            use_detailed_prompts: Use multiple descriptive prompts per food
                (more accurate but slower initial loading)
        """
        # Use explicit model name if provided, otherwise use preset
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = get_clip_model_name(model_preset)

        self.model_preset = model_preset
        self.device = device or self._detect_device()
        self.use_detailed_prompts = use_detailed_prompts

        self._model = None
        self._processor = None
        self._loaded = False
        self._text_features_cache: Dict[str, torch.Tensor] = {}

        logger.info(
            f"CLIPFoodClassifier initialized (model: {model_name}, device: {self.device})"
        )

    def _detect_device(self) -> str:
        """Detect best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self) -> None:
        """Load CLIP model (lazy loading)."""
        if self._loaded:
            return

        try:
            from transformers import CLIPProcessor, CLIPModel  # type: ignore[import-untyped]

            logger.info(f"Loading CLIP model: {self.model_name}...")

            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name)
            self._model = self._model.to(self.device)  # type: ignore[attr-defined]
            self._model.eval()  # type: ignore[attr-defined]

            self._loaded = True
            logger.info(f"CLIP model loaded successfully on {self.device}")

            # Pre-compute text features for all food categories
            self._precompute_text_features()

        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise

    def _precompute_text_features(self) -> None:
        """
        Pre-compute text embeddings for all food categories.

        Optimizations:
        1. Disk caching - Load from cache if available (instant startup)
        2. Batch encoding - Process multiple prompts at once (faster computation)
        """
        import time

        start_time = time.time()

        # Try to load from cache first
        cache_path = _get_cache_path(self.model_name, self.use_detailed_prompts)
        if cache_path.exists():
            try:
                logger.info(f"Loading text features from cache: {cache_path}")
                cached_data = torch.load(
                    cache_path, map_location=self.device, weights_only=True
                )
                self._text_features_cache = cached_data
                elapsed = time.time() - start_time
                logger.info(
                    f"Loaded {len(self._text_features_cache)} cached text features in {elapsed:.2f}s"
                )
                return
            except Exception as e:
                logger.warning(f"Failed to load cache, recomputing: {e}")

        # Compute text features with batching
        logger.info(
            "Computing text features (this may take a few minutes on first run)..."
        )

        if self.use_detailed_prompts:
            self._compute_detailed_features_batched()
        else:
            self._compute_simple_features_batched()

        elapsed = time.time() - start_time
        logger.info(
            f"Computed features for {len(self._text_features_cache)} food categories in {elapsed:.2f}s"
        )

        # Save to cache for next startup
        self._save_text_features_cache(cache_path)

    def _compute_detailed_features_batched(self) -> None:
        """Compute detailed text features using batched encoding."""
        # Collect all prompts with their food keys
        all_prompts: List[str] = []
        prompt_to_food: List[Tuple[str, int]] = []  # (food_key, prompt_index_in_food)
        food_prompt_counts: Dict[str, int] = {}

        for food_key, prompts in FOOD_PROMPTS.items():
            food_prompt_counts[food_key] = len(prompts)
            for idx, prompt in enumerate(prompts):
                all_prompts.append(prompt)
                prompt_to_food.append((food_key, idx))

        logger.info(
            f"Encoding {len(all_prompts)} prompts for {len(FOOD_PROMPTS)} foods..."
        )

        # Process in batches
        all_features: List[torch.Tensor] = []
        batch_size = TEXT_ENCODING_BATCH_SIZE

        for i in range(0, len(all_prompts), batch_size):
            batch_prompts = all_prompts[i : i + batch_size]
            inputs = self._processor(  # type: ignore[misc]
                text=batch_prompts, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                batch_features = self._model.get_text_features(**inputs)  # type: ignore[attr-defined]
                batch_features = batch_features / batch_features.norm(
                    dim=-1, keepdim=True
                )
                all_features.append(batch_features.cpu())

            # Log progress every 10 batches
            if (i // batch_size) % 10 == 0:
                progress = min(100, (i + batch_size) * 100 // len(all_prompts))
                logger.info(f"Text encoding progress: {progress}%")

        # Concatenate all features
        all_features_tensor = torch.cat(all_features, dim=0)

        # Group by food and average
        feature_idx = 0
        for food_key, num_prompts in food_prompt_counts.items():
            food_features = all_features_tensor[feature_idx : feature_idx + num_prompts]
            avg_features = torch.mean(food_features, dim=0, keepdim=True)
            avg_features = avg_features / avg_features.norm(dim=-1, keepdim=True)
            self._text_features_cache[food_key] = avg_features.to(self.device)
            feature_idx += num_prompts

    def _compute_simple_features_batched(self) -> None:
        """Compute simple text features using batched encoding."""
        food_keys = list(SIMPLE_FOOD_PROMPTS.keys())
        all_prompts = list(SIMPLE_FOOD_PROMPTS.values())

        logger.info(f"Encoding {len(all_prompts)} simple prompts...")

        # Process in batches
        all_features: List[torch.Tensor] = []
        batch_size = TEXT_ENCODING_BATCH_SIZE

        for i in range(0, len(all_prompts), batch_size):
            batch_prompts = all_prompts[i : i + batch_size]
            inputs = self._processor(  # type: ignore[misc]
                text=batch_prompts, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                batch_features = self._model.get_text_features(**inputs)  # type: ignore[attr-defined]
                batch_features = batch_features / batch_features.norm(
                    dim=-1, keepdim=True
                )
                all_features.append(batch_features.cpu())

        # Concatenate and assign to cache
        all_features_tensor = torch.cat(all_features, dim=0)
        for idx, food_key in enumerate(food_keys):
            self._text_features_cache[food_key] = all_features_tensor[idx : idx + 1].to(
                self.device
            )

    def _save_text_features_cache(self, cache_path: Path) -> None:
        """Save computed text features to disk cache."""
        try:
            # Ensure cache directory exists
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Move tensors to CPU for storage
            cache_data = {k: v.cpu() for k, v in self._text_features_cache.items()}
            torch.save(cache_data, cache_path)
            logger.info(f"Saved text features cache to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save text features cache: {e}")

    @torch.no_grad()
    def classify(
        self,
        image: Image.Image,
        candidate_foods: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Classify a food image using CLIP.

        Args:
            image: PIL Image to classify
            candidate_foods: Optional list of food keys to consider (defaults to all)
            top_k: Number of top predictions to return

        Returns:
            List of (food_key, confidence) tuples sorted by confidence
        """
        if not self._loaded:
            self.load_model()

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Get image features
        inputs = self._processor(images=image, return_tensors="pt")  # type: ignore[misc]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        image_features = self._model.get_image_features(**inputs)  # type: ignore[attr-defined]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Determine which foods to classify against
        if candidate_foods:
            foods_to_check = [
                f for f in candidate_foods if f in self._text_features_cache
            ]
        else:
            foods_to_check = list(self._text_features_cache.keys())

        if not foods_to_check:
            logger.warning("No valid food candidates for CLIP classification")
            return []

        # Compute similarities
        similarities = {}
        for food_key in foods_to_check:
            text_features = self._text_features_cache[food_key]
            similarity = (image_features @ text_features.T).squeeze().item()
            similarities[food_key] = similarity

        # Convert to probabilities using softmax
        sim_values = torch.tensor(list(similarities.values()))
        probs = torch.softmax(sim_values * 100, dim=0)  # Temperature scaling

        results = []
        for i, (food_key, sim) in enumerate(similarities.items()):
            results.append((food_key, probs[i].item()))

        # Sort by probability
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    @torch.no_grad()
    def classify_with_prompts(
        self, image: Image.Image, custom_prompts: List[str], top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Classify using custom text prompts (for dynamic classification).

        Args:
            image: PIL Image to classify
            custom_prompts: List of text prompts to compare against
            top_k: Number of top results

        Returns:
            List of (prompt, confidence) tuples
        """
        if not self._loaded:
            self.load_model()

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Get image features
        img_inputs = self._processor(images=image, return_tensors="pt")  # type: ignore[misc]
        img_inputs = {k: v.to(self.device) for k, v in img_inputs.items()}
        image_features = self._model.get_image_features(**img_inputs)  # type: ignore[attr-defined]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Get text features for all prompts
        text_inputs = self._processor(  # type: ignore[misc]
            text=custom_prompts, return_tensors="pt", padding=True
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        text_features = self._model.get_text_features(**text_inputs)  # type: ignore[attr-defined]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute similarities
        similarities = (image_features @ text_features.T).squeeze()
        probs = torch.softmax(similarities * 100, dim=0)

        results = [(prompt, prob.item()) for prompt, prob in zip(custom_prompts, probs)]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    @torch.no_grad()
    def classify_hierarchical(
        self,
        image: Image.Image,
        top_k: int = 5,
        cuisine_threshold: float = 0.15,
    ) -> Dict[str, Any]:
        """
        Two-stage hierarchical classification: first identify cuisine, then narrow to dishes.

        This approach improves accuracy by:
        1. First classifying into broad cuisine categories (Japanese, Italian, etc.)
        2. Then narrowing down to specific dishes within top cuisines
        3. Combining scores for more confident predictions

        Args:
            image: PIL Image to classify
            top_k: Number of top predictions to return
            cuisine_threshold: Minimum confidence to consider a cuisine (default 0.15)

        Returns:
            Dict with 'predictions', 'cuisine_scores', and 'method' keys
        """
        if not self._loaded:
            self.load_model()

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Stage 1: Classify into cuisine categories
        cuisine_prompts = []
        cuisine_names = []
        for cuisine, data in CUISINE_CATEGORIES.items():
            cuisine_prompts.extend(data["prompts"])
            cuisine_names.extend([cuisine] * len(data["prompts"]))

        cuisine_results = self.classify_with_prompts(
            image, cuisine_prompts, top_k=len(cuisine_prompts)
        )

        # Aggregate scores by cuisine (average of prompt scores)
        cuisine_scores: Dict[str, List[float]] = {c: [] for c in CUISINE_CATEGORIES}
        for prompt, score in cuisine_results:
            idx = cuisine_prompts.index(prompt)
            cuisine = cuisine_names[idx]
            cuisine_scores[cuisine].append(score)

        cuisine_avg_scores = {
            c: sum(scores) / len(scores) if scores else 0.0
            for c, scores in cuisine_scores.items()
        }

        # Sort cuisines by score
        sorted_cuisines = sorted(
            cuisine_avg_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Stage 2: Get dishes from top cuisines above threshold
        candidate_dishes: List[str] = []
        for cuisine, score in sorted_cuisines:
            if score >= cuisine_threshold:
                candidate_dishes.extend(CUISINE_CATEGORIES[cuisine]["dishes"])

        # If no cuisines above threshold, use top 3
        if not candidate_dishes:
            for cuisine, _ in sorted_cuisines[:3]:
                candidate_dishes.extend(CUISINE_CATEGORIES[cuisine]["dishes"])

        # Remove duplicates while preserving order
        seen = set()
        unique_dishes = []
        for dish in candidate_dishes:
            if dish not in seen and dish in self._text_features_cache:
                seen.add(dish)
                unique_dishes.append(dish)

        # Classify against candidate dishes
        if unique_dishes:
            dish_results = self.classify(
                image, candidate_foods=unique_dishes, top_k=top_k
            )
        else:
            # Fallback to full classification
            dish_results = self.classify(image, top_k=top_k)

        # Enhance results with cuisine context
        enhanced_predictions = []
        for food_key, confidence in dish_results:
            cuisine = DISH_TO_CUISINE.get(food_key, "unknown")
            cuisine_boost = cuisine_avg_scores.get(cuisine, 0.0)
            # Combine dish confidence with cuisine confidence (weighted)
            combined_score = confidence * 0.7 + cuisine_boost * 0.3
            enhanced_predictions.append(
                {
                    "food_key": food_key,
                    "confidence": confidence,
                    "cuisine": cuisine,
                    "cuisine_confidence": cuisine_boost,
                    "combined_score": combined_score,
                }
            )

        # Sort by combined score
        enhanced_predictions.sort(key=lambda x: x["combined_score"], reverse=True)

        return {
            "predictions": enhanced_predictions[:top_k],
            "cuisine_scores": dict(sorted_cuisines[:5]),
            "method": "hierarchical",
            "candidate_cuisines": [
                c for c, s in sorted_cuisines if s >= cuisine_threshold
            ],
        }

    def _get_multi_crop_images(self, image: Image.Image) -> List[Image.Image]:
        """
        Generate multiple crops of the image for test-time augmentation (TTA).

        Crops:
        - Center crop (main subject)
        - Four corner crops (capture different regions)
        - Horizontal flip of center crop

        This improves robustness by averaging predictions across views.
        """
        crops = []
        w, h = image.size
        crop_size = min(w, h)

        # Calculate crop positions
        center_x, center_y = w // 2, h // 2
        half_crop = crop_size // 2

        # Center crop
        center_crop = image.crop(
            (
                center_x - half_crop,
                center_y - half_crop,
                center_x + half_crop,
                center_y + half_crop,
            )
        )
        crops.append(center_crop)

        # Corner crops (slightly smaller to capture different regions)
        corner_size = int(crop_size * 0.8)
        corner_positions = [
            (0, 0),  # Top-left
            (w - corner_size, 0),  # Top-right
            (0, h - corner_size),  # Bottom-left
            (w - corner_size, h - corner_size),  # Bottom-right
        ]

        for x, y in corner_positions:
            # Ensure we don't go out of bounds
            x = max(0, min(x, w - corner_size))
            y = max(0, min(y, h - corner_size))
            corner_crop = image.crop((x, y, x + corner_size, y + corner_size))
            crops.append(corner_crop)

        # Horizontal flip of center crop
        flipped = center_crop.transpose(Image.FLIP_LEFT_RIGHT)
        crops.append(flipped)

        return crops

    @torch.no_grad()
    def classify_with_tta(
        self,
        image: Image.Image,
        candidate_foods: Optional[List[str]] = None,
        top_k: int = 5,
        use_multi_crop: bool = True,
        fast_mode: bool = True,
    ) -> Dict[str, Any]:
        """
        Classify with Test-Time Augmentation (TTA) for improved accuracy.

        TTA averages predictions across multiple image crops and augmentations,
        which improves robustness and accuracy, especially for:
        - Off-center food items
        - Partially visible foods
        - Different presentation angles

        Args:
            image: PIL Image to classify
            candidate_foods: Optional list of food keys to consider
            top_k: Number of top predictions to return
            use_multi_crop: Whether to use multi-crop augmentation
            fast_mode: Use batched processing for better performance (default True)

        Returns:
            Dict with predictions, confidence, uncertainty metrics
        """
        if not self._loaded:
            self.load_model()

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Determine foods to check
        if candidate_foods:
            foods_to_check = [
                f for f in candidate_foods if f in self._text_features_cache
            ]
        else:
            foods_to_check = list(self._text_features_cache.keys())

        if not foods_to_check:
            return {
                "predictions": [],
                "method": "tta",
                "num_crops": 0,
                "uncertainty": 1.0,
            }

        # Get image crops for TTA
        if use_multi_crop:
            crops = self._get_multi_crop_images(image)
        else:
            crops = [image]

        # Use fast batched processing
        if fast_mode and len(crops) > 1:
            return self._classify_with_tta_batched(crops, foods_to_check, top_k)

        # Fallback to sequential processing
        all_similarities: Dict[str, List[float]] = {food: [] for food in foods_to_check}

        for crop in crops:
            crop_resized = crop.resize((224, 224), Image.LANCZOS)
            inputs = self._processor(images=crop_resized, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            image_features = self._model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            for food_key in foods_to_check:
                text_features = self._text_features_cache[food_key]
                similarity = (image_features @ text_features.T).squeeze().item()
                all_similarities[food_key].append(similarity)

        return self._compute_tta_results(
            all_similarities, foods_to_check, len(crops), top_k
        )

    @torch.no_grad()
    def _classify_with_tta_batched(
        self,
        crops: List[Image.Image],
        foods_to_check: List[str],
        top_k: int,
    ) -> Dict[str, Any]:
        """
        Optimized batched TTA processing - all crops in single forward pass.

        This is ~4-5x faster than sequential processing.
        """
        # Resize all crops
        resized_crops = [crop.resize((224, 224), Image.LANCZOS) for crop in crops]

        # Batch process all crops at once
        inputs = self._processor(
            images=resized_crops, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Single forward pass for all crops
        batch_image_features = self._model.get_image_features(**inputs)
        batch_image_features = batch_image_features / batch_image_features.norm(
            dim=-1, keepdim=True
        )

        # Stack text features for efficient batch similarity computation
        text_feature_list = [self._text_features_cache[food] for food in foods_to_check]
        stacked_text_features = torch.stack(text_feature_list).squeeze(
            1
        )  # [num_foods, embed_dim]

        # Compute all similarities at once: [num_crops, num_foods]
        all_sims = batch_image_features @ stacked_text_features.T

        # Average across crops: [num_foods]
        avg_sims = all_sims.mean(dim=0)

        # Compute variance across crops for consistency
        variance = all_sims.var(dim=0)

        # Convert to similarities dict format for compatibility
        all_similarities = {
            food: all_sims[:, i].cpu().tolist() for i, food in enumerate(foods_to_check)
        }

        return self._compute_tta_results(
            all_similarities, foods_to_check, len(crops), top_k
        )

    def _compute_tta_results(
        self,
        all_similarities: Dict[str, List[float]],
        foods_to_check: List[str],
        num_crops: int,
        top_k: int,
    ) -> Dict[str, Any]:
        """Compute final TTA results from similarity scores."""
        # Average similarities across crops
        avg_similarities = {
            food: sum(sims) / len(sims) for food, sims in all_similarities.items()
        }

        # Convert to probabilities with temperature scaling
        sim_values = torch.tensor(list(avg_similarities.values()))

        # Adaptive temperature based on similarity spread
        sim_std = sim_values.std().item()
        temperature = max(50, min(150, 100 / (sim_std + 0.1)))

        probs = torch.softmax(sim_values * temperature, dim=0)

        # Calculate uncertainty (entropy-based)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        max_entropy = torch.log(torch.tensor(len(probs))).item()
        uncertainty = entropy / max_entropy if max_entropy > 0 else 1.0

        # Build results
        results = []
        for i, (food_key, avg_sim) in enumerate(avg_similarities.items()):
            sims = all_similarities[food_key]
            variance = (
                sum((s - avg_sim) ** 2 for s in sims) / len(sims)
                if len(sims) > 1
                else 0
            )

            results.append(
                {
                    "food_key": food_key,
                    "confidence": probs[i].item(),
                    "avg_similarity": avg_sim,
                    "consistency": 1.0 - min(variance * 10, 1.0),
                }
            )

        results.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "predictions": results[:top_k],
            "method": "tta_batched",
            "num_crops": num_crops,
            "uncertainty": uncertainty,
            "temperature": temperature,
        }

    @torch.no_grad()
    def classify_enhanced(
        self,
        image: Image.Image,
        top_k: int = 5,
        use_tta: bool = True,
        use_hierarchical: bool = True,
    ) -> Dict[str, Any]:
        """
        Enhanced classification combining multiple strategies.

        This is the most accurate classification method, combining:
        1. Test-time augmentation (TTA) for robustness
        2. Hierarchical cuisine-first classification
        3. Confidence calibration with uncertainty estimation

        Args:
            image: PIL Image to classify
            top_k: Number of top predictions to return
            use_tta: Whether to use test-time augmentation
            use_hierarchical: Whether to use hierarchical classification

        Returns:
            Dict with comprehensive prediction results
        """
        if not self._loaded:
            self.load_model()

        results: Dict[str, Any] = {
            "method": "enhanced",
            "strategies_used": [],
        }

        # Strategy 1: Hierarchical classification (cuisine-aware)
        if use_hierarchical:
            hierarchical_result = self.classify_hierarchical(image, top_k=top_k * 2)
            results["hierarchical"] = hierarchical_result
            results["strategies_used"].append("hierarchical")

        # Strategy 2: TTA classification
        if use_tta:
            tta_result = self.classify_with_tta(image, top_k=top_k * 2)
            results["tta"] = tta_result
            results["strategies_used"].append("tta")

        # Combine results using weighted voting
        vote_scores: Dict[str, float] = {}

        # Add hierarchical votes (weight by combined_score)
        if use_hierarchical and hierarchical_result.get("predictions"):
            for pred in hierarchical_result["predictions"]:
                food = pred["food_key"]
                score = pred["combined_score"]
                vote_scores[food] = (
                    vote_scores.get(food, 0) + score * 1.2
                )  # Boost hierarchical

        # Add TTA votes (weight by confidence and consistency)
        if use_tta and tta_result.get("predictions"):
            for pred in tta_result["predictions"]:
                food = pred["food_key"]
                score = pred["confidence"] * pred.get("consistency", 1.0)
                vote_scores[food] = vote_scores.get(food, 0) + score

        # Sort by raw vote score and convert to confidence
        if vote_scores:
            # Sort by raw vote score
            sorted_foods = sorted(
                vote_scores.items(), key=lambda x: x[1], reverse=True
            )[:top_k]

            # Get the best confidence from individual strategies for top prediction
            top_food = sorted_foods[0][0]
            best_confidence = 0.0

            # Find best individual confidence for top prediction
            if use_tta and tta_result.get("predictions"):
                for pred in tta_result["predictions"]:
                    if pred["food_key"] == top_food:
                        best_confidence = max(best_confidence, pred["confidence"])
                        break

            if use_hierarchical and hierarchical_result.get("predictions"):
                for pred in hierarchical_result["predictions"]:
                    if pred["food_key"] == top_food:
                        best_confidence = max(best_confidence, pred["confidence"])
                        break

            # Compute relative scores (for ranking display)
            max_vote = sorted_foods[0][1] if sorted_foods else 1.0
            final_predictions = [
                {
                    "food_key": food,
                    "score": vote / max_vote,  # Relative score for ranking
                    "vote_score": vote,
                }
                for food, vote in sorted_foods
            ]

            results["predictions"] = final_predictions
            results["top_prediction"] = final_predictions[0]["food_key"]
            results["top_confidence"] = best_confidence  # Use actual best confidence

            # Overall uncertainty (average of strategy uncertainties)
            uncertainties = []
            if use_tta and "uncertainty" in tta_result:
                uncertainties.append(tta_result["uncertainty"])
            results["uncertainty"] = (
                sum(uncertainties) / len(uncertainties) if uncertainties else 0.5
            )
        else:
            results["predictions"] = []
            results["top_prediction"] = "unknown"
            results["top_confidence"] = 0.0
            results["uncertainty"] = 1.0

        return results

    def get_model_info(self) -> dict:
        """Get information about the model."""
        return {
            "name": f"CLIP ({self.model_name})",
            "type": "Zero-shot Vision-Language Model",
            "dataset": "400M image-text pairs from web",
            "num_food_categories": len(FOOD_PROMPTS),
            "use_detailed_prompts": self.use_detailed_prompts,
            "device": self.device,
            "loaded": self._loaded,
            "description": (
                "CLIP (Contrastive Language-Image Pre-training) enables zero-shot "
                "classification by comparing images to text descriptions. "
                "Excellent for raw produce, ingredients, and unusual foods."
            ),
        }


# Singleton instance
_clip_classifier_instance: Optional[CLIPFoodClassifier] = None


def get_clip_classifier(
    model_name: str = "openai/clip-vit-base-patch32", use_detailed_prompts: bool = True
) -> CLIPFoodClassifier:
    """Get singleton CLIP classifier instance."""
    global _clip_classifier_instance
    if _clip_classifier_instance is None:
        _clip_classifier_instance = CLIPFoodClassifier(
            model_name=model_name, use_detailed_prompts=use_detailed_prompts
        )
    return _clip_classifier_instance


class CLIPEnsembleClassifier:
    """
    Ensemble classifier combining multiple CLIP models for improved accuracy.

    Combines:
    - ViT-B/32 (fast, good baseline) - weight: 0.4
    - ViT-L/14 (slower, more accurate) - weight: 0.6

    The ensemble leverages different model strengths:
    - ViT-B sees patterns at 32x32 patch resolution (good for textures)
    - ViT-L sees patterns at 14x14 patch resolution (good for shapes/details)
    """

    def __init__(
        self,
        use_detailed_prompts: bool = True,
        device: Optional[str] = None,
    ):
        """Initialize ensemble with multiple CLIP models."""
        self.use_detailed_prompts = use_detailed_prompts
        self.device = device

        # Lazy-loaded models
        self._fast_model: Optional[CLIPFoodClassifier] = None  # ViT-B/32
        self._accurate_model: Optional[CLIPFoodClassifier] = None  # ViT-L/14

        # Weights for combining predictions
        self.fast_weight = 0.4
        self.accurate_weight = 0.6

        self._loaded = False
        logger.info("CLIPEnsembleClassifier initialized")

    def _get_fast_model(self) -> CLIPFoodClassifier:
        """Get fast model (ViT-B/32)."""
        if self._fast_model is None:
            self._fast_model = CLIPFoodClassifier(
                model_preset="fast",
                device=self.device,
                use_detailed_prompts=self.use_detailed_prompts,
            )
        return self._fast_model

    def _get_accurate_model(self) -> CLIPFoodClassifier:
        """Get accurate model (ViT-L/14)."""
        if self._accurate_model is None:
            self._accurate_model = CLIPFoodClassifier(
                model_preset="accurate",
                device=self.device,
                use_detailed_prompts=self.use_detailed_prompts,
            )
        return self._accurate_model

    def load_models(self, load_accurate: bool = True) -> None:
        """
        Load ensemble models.

        Args:
            load_accurate: Whether to load the accurate (larger) model.
                          Set False for faster startup with only fast model.
        """
        logger.info("Loading CLIP ensemble models...")

        # Always load fast model
        fast = self._get_fast_model()
        fast.load_model()

        # Optionally load accurate model
        if load_accurate:
            accurate = self._get_accurate_model()
            accurate.load_model()

        self._loaded = True
        logger.info("CLIP ensemble models loaded")

    @torch.no_grad()
    def classify(
        self,
        image: Image.Image,
        candidate_foods: Optional[List[str]] = None,
        top_k: int = 5,
        use_both_models: bool = True,
    ) -> Dict[str, Any]:
        """
        Classify using ensemble of CLIP models.

        Args:
            image: PIL Image to classify
            candidate_foods: Optional list of food keys to consider
            top_k: Number of top predictions to return
            use_both_models: If True, use both models. If False, use only fast model.

        Returns:
            Dict with ensemble predictions and model-specific results
        """
        results: Dict[str, Any] = {
            "method": "ensemble",
            "models_used": [],
        }

        # Get predictions from fast model
        fast_model = self._get_fast_model()
        if not fast_model._loaded:
            fast_model.load_model()

        fast_predictions = fast_model.classify(image, candidate_foods, top_k=top_k * 2)
        results["fast_model"] = {
            "predictions": fast_predictions,
            "model": "ViT-B/32",
        }
        results["models_used"].append("ViT-B/32")

        # Accumulate weighted scores
        weighted_scores: Dict[str, float] = {}
        for food, conf in fast_predictions:
            weighted_scores[food] = conf * self.fast_weight

        # Get predictions from accurate model if requested
        if use_both_models:
            accurate_model = self._get_accurate_model()
            if not accurate_model._loaded:
                accurate_model.load_model()

            accurate_predictions = accurate_model.classify(
                image, candidate_foods, top_k=top_k * 2
            )
            results["accurate_model"] = {
                "predictions": accurate_predictions,
                "model": "ViT-L/14",
            }
            results["models_used"].append("ViT-L/14")

            # Add accurate model scores
            for food, conf in accurate_predictions:
                weighted_scores[food] = (
                    weighted_scores.get(food, 0) + conf * self.accurate_weight
                )

        # Normalize scores
        if weighted_scores:
            max_score = max(weighted_scores.values())
            normalized = {
                food: score / max_score for food, score in weighted_scores.items()
            }

            # Sort by score
            sorted_predictions = sorted(
                normalized.items(), key=lambda x: x[1], reverse=True
            )[:top_k]

            results["predictions"] = [
                {"food_key": food, "confidence": conf}
                for food, conf in sorted_predictions
            ]
            results["top_prediction"] = sorted_predictions[0][0]
            results["top_confidence"] = sorted_predictions[0][1]

            # Calculate agreement between models
            if use_both_models and fast_predictions and accurate_predictions:
                fast_top = fast_predictions[0][0]
                accurate_top = accurate_predictions[0][0]
                results["models_agree"] = fast_top == accurate_top
        else:
            results["predictions"] = []
            results["top_prediction"] = "unknown"
            results["top_confidence"] = 0.0

        return results

    @torch.no_grad()
    def classify_enhanced(
        self,
        image: Image.Image,
        top_k: int = 5,
        use_tta: bool = True,
    ) -> Dict[str, Any]:
        """
        Most accurate classification: ensemble + TTA + hierarchical.

        This combines:
        1. Multiple CLIP models (ViT-B + ViT-L)
        2. Test-time augmentation (6 crops per model)
        3. Hierarchical cuisine-first classification

        Note: This is slower but provides the best accuracy.

        Args:
            image: PIL Image to classify
            top_k: Number of top predictions to return
            use_tta: Whether to use test-time augmentation

        Returns:
            Dict with comprehensive ensemble predictions
        """
        results: Dict[str, Any] = {
            "method": "enhanced_ensemble",
            "strategies": [],
        }

        weighted_scores: Dict[str, float] = {}

        # Strategy 1: Fast model with enhanced classification
        fast_model = self._get_fast_model()
        if not fast_model._loaded:
            fast_model.load_model()

        if use_tta:
            fast_result = fast_model.classify_enhanced(image, top_k=top_k * 2)
        else:
            fast_result = fast_model.classify_hierarchical(image, top_k=top_k * 2)

        results["fast_model"] = fast_result
        results["strategies"].append("fast_model_enhanced")

        # Add fast model scores
        if fast_result.get("predictions"):
            for pred in fast_result["predictions"]:
                food = pred.get("food_key") or pred.get("food")
                score = (
                    pred.get("score")
                    or pred.get("combined_score")
                    or pred.get("confidence", 0)
                )
                if food:
                    weighted_scores[food] = score * self.fast_weight

        # Strategy 2: Accurate model with enhanced classification
        accurate_model = self._get_accurate_model()
        if not accurate_model._loaded:
            accurate_model.load_model()

        if use_tta:
            accurate_result = accurate_model.classify_enhanced(image, top_k=top_k * 2)
        else:
            accurate_result = accurate_model.classify_hierarchical(
                image, top_k=top_k * 2
            )

        results["accurate_model"] = accurate_result
        results["strategies"].append("accurate_model_enhanced")

        # Add accurate model scores (higher weight)
        if accurate_result.get("predictions"):
            for pred in accurate_result["predictions"]:
                food = pred.get("food_key") or pred.get("food")
                score = (
                    pred.get("score")
                    or pred.get("combined_score")
                    or pred.get("confidence", 0)
                )
                if food:
                    weighted_scores[food] = (
                        weighted_scores.get(food, 0) + score * self.accurate_weight
                    )

        # Combine and normalize
        if weighted_scores:
            max_score = max(weighted_scores.values())
            normalized = {
                food: score / max_score for food, score in weighted_scores.items()
            }

            sorted_predictions = sorted(
                normalized.items(), key=lambda x: x[1], reverse=True
            )[:top_k]

            results["predictions"] = [
                {"food_key": food, "confidence": conf}
                for food, conf in sorted_predictions
            ]
            results["top_prediction"] = sorted_predictions[0][0]
            results["top_confidence"] = sorted_predictions[0][1]

            # Calculate model agreement
            fast_top = fast_result.get("top_prediction")
            accurate_top = accurate_result.get("top_prediction")
            results["models_agree"] = fast_top == accurate_top

            # Combine uncertainties
            fast_unc = fast_result.get("uncertainty", 0.5)
            accurate_unc = accurate_result.get("uncertainty", 0.5)
            results["uncertainty"] = (
                fast_unc * self.fast_weight + accurate_unc * self.accurate_weight
            )
        else:
            results["predictions"] = []
            results["top_prediction"] = "unknown"
            results["top_confidence"] = 0.0
            results["uncertainty"] = 1.0

        return results


# Singleton ensemble instance
_clip_ensemble_instance: Optional[CLIPEnsembleClassifier] = None


def get_clip_ensemble() -> CLIPEnsembleClassifier:
    """Get singleton CLIP ensemble classifier instance."""
    global _clip_ensemble_instance
    if _clip_ensemble_instance is None:
        _clip_ensemble_instance = CLIPEnsembleClassifier()
    return _clip_ensemble_instance
