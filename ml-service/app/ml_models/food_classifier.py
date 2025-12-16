"""
Food Classification Model using Hugging Face Vision Transformer (ViT)

Uses nateraw/food model - a ViT fine-tuned on Food-101 dataset with 101 food classes.
Achieves ~90%+ accuracy on food classification.
"""

import logging
from typing import List, Tuple, Dict, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy imports to avoid loading heavy models at startup
_model = None
_processor = None
_model_loaded = False

# Food-101 class labels - 101 categories from the dataset
FOOD_101_CLASSES = [
    "apple_pie",
    "baby_back_ribs",
    "baklava",
    "beef_carpaccio",
    "beef_tartare",
    "beet_salad",
    "beignets",
    "bibimbap",
    "bread_pudding",
    "breakfast_burrito",
    "bruschetta",
    "caesar_salad",
    "cannoli",
    "caprese_salad",
    "carrot_cake",
    "ceviche",
    "cheesecake",
    "cheese_plate",
    "chicken_curry",
    "chicken_quesadilla",
    "chicken_wings",
    "chocolate_cake",
    "chocolate_mousse",
    "churros",
    "clam_chowder",
    "club_sandwich",
    "crab_cakes",
    "creme_brulee",
    "croque_madame",
    "cup_cakes",
    "deviled_eggs",
    "donuts",
    "dumplings",
    "edamame",
    "eggs_benedict",
    "escargots",
    "falafel",
    "filet_mignon",
    "fish_and_chips",
    "foie_gras",
    "french_fries",
    "french_onion_soup",
    "french_toast",
    "fried_calamari",
    "fried_rice",
    "frozen_yogurt",
    "garlic_bread",
    "gnocchi",
    "greek_salad",
    "grilled_cheese_sandwich",
    "grilled_salmon",
    "guacamole",
    "gyoza",
    "hamburger",
    "hot_and_sour_soup",
    "hot_dog",
    "huevos_rancheros",
    "hummus",
    "ice_cream",
    "lasagna",
    "lobster_bisque",
    "lobster_roll_sandwich",
    "macaroni_and_cheese",
    "macarons",
    "miso_soup",
    "mussels",
    "nachos",
    "omelette",
    "onion_rings",
    "oysters",
    "pad_thai",
    "paella",
    "pancakes",
    "panna_cotta",
    "peking_duck",
    "pho",
    "pizza",
    "pork_chop",
    "poutine",
    "prime_rib",
    "pulled_pork_sandwich",
    "ramen",
    "ravioli",
    "red_velvet_cake",
    "risotto",
    "samosa",
    "sashimi",
    "scallops",
    "seaweed_salad",
    "shrimp_and_grits",
    "spaghetti_bolognese",
    "spaghetti_carbonara",
    "spring_rolls",
    "steak",
    "strawberry_shortcake",
    "sushi",
    "tacos",
    "takoyaki",
    "tiramisu",
    "tuna_tartare",
    "waffles",
]

# Mapping from Food-101 classes to our food database keys
# This bridges the model output to our nutrition database
FOOD_101_TO_DATABASE: Dict[str, str] = {
    # Direct mappings (similar names)
    "apple_pie": "apple",  # Map to closest food
    "baklava": "mixed_nuts",  # Nut-based dessert
    "beef_carpaccio": "beef",
    "beef_tartare": "beef",
    "bibimbap": "rice",  # Rice-based dish
    "bread_pudding": "bread",
    "breakfast_burrito": "eggs",  # Egg-based
    "bruschetta": "bread",
    "caesar_salad": "chicken",  # Often has chicken
    "cannoli": "cheese",  # Ricotta filling
    "caprese_salad": "cheese",  # Mozzarella based
    "carrot_cake": "carrot",
    "cheesecake": "cheese",
    "cheese_plate": "cheese",
    "chicken_curry": "chicken",
    "chicken_quesadilla": "chicken",
    "chicken_wings": "chicken",
    "chocolate_cake": "chocolate",
    "chocolate_mousse": "chocolate",
    "churros": "bread",  # Fried dough
    "clam_chowder": "fish",  # Seafood soup
    "club_sandwich": "bread",
    "crab_cakes": "fish",  # Seafood
    "creme_brulee": "eggs",  # Custard
    "croque_madame": "bread",  # Sandwich
    "cup_cakes": "bread",  # Cake
    "deviled_eggs": "eggs",
    "donuts": "bread",  # Fried dough
    "dumplings": "pork",  # Often pork filled
    "edamame": "tofu",  # Soy beans
    "eggs_benedict": "eggs",
    "escargots": "fish",  # Closest protein
    "falafel": "chickpeas",
    "filet_mignon": "beef",
    "fish_and_chips": "fish",
    "foie_gras": "chicken",  # Poultry liver
    "french_fries": "potato",
    "french_onion_soup": "cheese",  # Cheese topped
    "french_toast": "bread",
    "fried_calamari": "fish",  # Seafood
    "fried_rice": "rice",
    "frozen_yogurt": "yogurt",
    "garlic_bread": "bread",
    "gnocchi": "potato",  # Potato based
    "greek_salad": "cheese",  # Feta based
    "grilled_cheese_sandwich": "cheese",
    "grilled_salmon": "salmon",
    "guacamole": "avocado",
    "gyoza": "pork",  # Often pork filled
    "hamburger": "burger",  # Database has "burger" with full nutrition
    "hot_and_sour_soup": "tofu",
    "hot_dog": "hot_dog",  # Keep as hot_dog if in database
    "huevos_rancheros": "eggs",
    "hummus": "chickpeas",
    "ice_cream": "milk",  # Dairy
    "lasagna": "beef",  # Meat lasagna
    "lobster_bisque": "fish",  # Seafood
    "lobster_roll_sandwich": "fish",  # Seafood
    "macaroni_and_cheese": "cheese",
    "macarons": "almonds",  # Almond flour
    "miso_soup": "tofu",
    "mussels": "fish",  # Seafood
    "nachos": "cheese",
    "omelette": "eggs",
    "onion_rings": "bread",  # Fried batter
    "oysters": "fish",  # Seafood
    "pad_thai": "pasta",  # Rice noodles (pasta has noodles alias)
    "paella": "rice",
    "pancakes": "pancake",  # Database has "pancake" with full nutrition
    "panna_cotta": "milk",  # Cream based
    "peking_duck": "chicken",  # Duck → poultry
    "pho": "pasta",  # Noodle-based soup (pasta has noodles alias)
    "pizza": "pizza",  # Database has "pizza" with full nutrition (285 cal/slice)
    "pork_chop": "pork",
    "poutine": "potato",
    "prime_rib": "beef",
    "pulled_pork_sandwich": "pork",
    "ramen": "pasta",  # Noodle dish (pasta has noodles alias)
    "ravioli": "cheese",  # Often cheese filled
    "red_velvet_cake": "bread",  # Cake
    "risotto": "rice",
    "samosa": "potato",  # Often potato filled
    "sashimi": "salmon",  # Raw fish
    "scallops": "fish",  # Seafood
    "seaweed_salad": "fish",  # Seafood adjacent
    "shrimp_and_grits": "fish",  # Seafood
    "spaghetti_bolognese": "beef",
    "spaghetti_carbonara": "pork",  # Bacon/guanciale
    "spring_rolls": "pork",
    "steak": "beef",
    "strawberry_shortcake": "strawberry",  # Strawberry dessert
    "sushi": "sushi",  # Database has "sushi" (200 cal per roll)
    "tacos": "taco",  # Database has "taco" with full nutrition
    "takoyaki": "fish",  # Octopus
    "tiramisu": "cheese",  # Mascarpone
    "tuna_tartare": "tuna",
    "waffles": "waffle",  # Database has "waffle" with full nutrition
    # Nuts that might be detected
    "baby_back_ribs": "pork",
    "beet_salad": "beets",
    "beignets": "bread",
    "ceviche": "fish",
}

# Additional common foods that might need custom handling
COMMON_FOOD_PATTERNS = {
    "nut": [
        "chestnuts",
        "almonds",
        "walnuts",
        "cashews",
        "peanuts",
        "hazelnuts",
        "pecans",
        "pistachios",
        "macadamia_nuts",
        "brazil_nuts",
        "pine_nuts",
        "mixed_nuts",
    ],
    "seed": [
        "chia_seeds",
        "flax_seeds",
        "pumpkin_seeds",
        "sesame_seeds",
        "hemp_seeds",
        "sunflower_seeds",
    ],
    "fruit": ["apple", "banana", "orange", "grape", "strawberry", "blueberry", "mango"],
    "vegetable": ["broccoli", "carrot", "spinach", "potato", "tomato", "cucumber"],
}


def get_model():
    """Lazy load the food classification model."""
    global _model, _processor, _model_loaded

    if _model_loaded:
        return _model, _processor

    try:
        from transformers import ViTImageProcessor, ViTForImageClassification  # type: ignore[import-untyped]

        logger.info("Loading food classification model (nateraw/food)...")

        # Use nateraw/food - ViT model fine-tuned on Food-101
        model_name = "nateraw/food"

        # Load processor and model
        _processor = ViTImageProcessor.from_pretrained(model_name)
        _model = ViTForImageClassification.from_pretrained(model_name)

        # Move to GPU if available
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        _model = _model.to(device)
        _model.eval()

        _model_loaded = True
        logger.info(f"Food classification model loaded successfully on {device}")

        return _model, _processor

    except Exception as e:
        logger.error(f"Failed to load food classification model: {e}")
        raise


class FoodClassifier:
    """
    State-of-the-art food classifier using Vision Transformer (ViT).

    Model: nateraw/food (ViT fine-tuned on Food-101)
    Accuracy: ~90% on Food-101 benchmark
    Classes: 101 food categories
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the food classifier.

        Args:
            device: Device to run inference on ('cuda', 'mps', 'cpu', or None for auto)
        """
        self.device = device or self._detect_device()
        self._model = None
        self._processor = None
        self._loaded = False

        logger.info(f"FoodClassifier initialized (device: {self.device})")

    def _detect_device(self) -> str:
        """Detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self) -> None:
        """Load the model (lazy loading on first inference)."""
        if self._loaded:
            return

        try:
            from transformers import ViTImageProcessor, ViTForImageClassification

            logger.info("Loading food classification model...")
            model_name = "nateraw/food"

            self._processor = ViTImageProcessor.from_pretrained(model_name)
            self._model = ViTForImageClassification.from_pretrained(model_name)
            self._model = self._model.to(self.device)  # type: ignore[attr-defined]
            self._model.eval()  # type: ignore[attr-defined]

            self._loaded = True
            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    @torch.no_grad()
    def classify(self, image: Image.Image, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Classify a food image.

        Args:
            image: PIL Image to classify
            top_k: Number of top predictions to return

        Returns:
            List of (class_name, confidence) tuples
        """
        if not self._loaded:
            self.load_model()

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess
        inputs = self._processor(images=image, return_tensors="pt")  # type: ignore[misc]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        outputs = self._model(**inputs)  # type: ignore[misc]
        logits = outputs.logits

        # Get top-k predictions
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(
            probs[0], k=min(top_k, len(FOOD_101_CLASSES))
        )

        results = []
        for prob, idx in zip(top_probs, top_indices):
            class_name = FOOD_101_CLASSES[idx.item()]
            confidence = prob.item()
            results.append((class_name, confidence))

        return results

    def classify_with_database_mapping(
        self, image: Image.Image, database_keys: List[str], top_k: int = 3
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Classify food and map to nutrition database keys.

        Args:
            image: PIL Image to classify
            database_keys: List of valid keys in the nutrition database
            top_k: Number of alternatives to return

        Returns:
            Tuple of (primary_class, confidence, alternatives)
        """
        # Get model predictions
        predictions = self.classify(image, top_k=10)

        if not predictions:
            # Fallback if no predictions
            return database_keys[0] if database_keys else "unknown", 0.5, []

        primary_food101, primary_conf = predictions[0]

        # Map Food-101 class to database key
        primary_db_key = self._map_to_database(primary_food101, database_keys)

        # Get alternatives
        alternatives = []
        seen_keys = {primary_db_key}

        for food101_class, conf in predictions[1:]:
            db_key = self._map_to_database(food101_class, database_keys)
            if db_key not in seen_keys:
                alternatives.append((db_key, conf))
                seen_keys.add(db_key)
                if len(alternatives) >= top_k:
                    break

        return primary_db_key, primary_conf, alternatives

    def _map_to_database(self, food101_class: str, database_keys: List[str]) -> str:
        """
        Map a Food-101 class to a nutrition database key.

        Args:
            food101_class: Food-101 class name
            database_keys: Valid database keys

        Returns:
            Best matching database key
        """
        # Check direct mapping
        if food101_class in FOOD_101_TO_DATABASE:
            mapped = FOOD_101_TO_DATABASE[food101_class]
            if mapped in database_keys:
                return mapped

        # Try fuzzy matching on class name
        food101_lower = food101_class.lower().replace("_", " ")

        # Check if any database key is contained in the Food-101 class name
        for db_key in database_keys:
            db_lower = db_key.lower().replace("_", " ")
            if db_lower in food101_lower or food101_lower in db_lower:
                return db_key

        # Check by food category patterns
        for category, foods in COMMON_FOOD_PATTERNS.items():
            if category in food101_lower:
                for food in foods:
                    if food in database_keys:
                        return food

        # Fallback: return the mapped value or first database key
        if food101_class in FOOD_101_TO_DATABASE:
            mapped = FOOD_101_TO_DATABASE[food101_class]
            # Find closest match in database
            for db_key in database_keys:
                if mapped in db_key or db_key in mapped:
                    return db_key

        # Last resort: return first available key
        return database_keys[0] if database_keys else "unknown"

    def get_model_info(self) -> dict:
        """Get information about the model."""
        return {
            "name": "nateraw/food (ViT-Food-101)",
            "type": "Vision Transformer (ViT)",
            "dataset": "Food-101",
            "num_classes": len(FOOD_101_CLASSES),
            "accuracy": 0.90,  # ~90% on Food-101 benchmark
            "device": self.device,
            "loaded": self._loaded,
            "description": (
                "State-of-the-art Vision Transformer model fine-tuned on Food-101 dataset. "
                "Achieves ~90% accuracy on 101 food categories."
            ),
        }


# Singleton instance for lazy loading
_classifier_instance: Optional[FoodClassifier] = None


def get_food_classifier() -> FoodClassifier:
    """Get the singleton food classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = FoodClassifier()
    return _classifier_instance
