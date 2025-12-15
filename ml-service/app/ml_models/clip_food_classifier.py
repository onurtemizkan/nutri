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
import logging
from typing import List, Tuple, Dict, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy loading
_clip_model = None
_clip_processor = None
_clip_loaded = False
_text_features_cache = {}

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
}

# Simplified prompts for faster matching
SIMPLE_FOOD_PROMPTS: Dict[str, str] = {
    key: f"a photo of {key.replace('_', ' ')}" for key in FOOD_PROMPTS.keys()
}


class CLIPFoodClassifier:
    """
    Zero-shot food classifier using OpenAI's CLIP model.

    CLIP compares images to text descriptions, allowing classification
    of any food category without specific training.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        use_detailed_prompts: bool = True,
    ):
        """
        Initialize CLIP food classifier.

        Args:
            model_name: CLIP model to use. Options:
                - "openai/clip-vit-base-patch32" (faster, 400MB)
                - "openai/clip-vit-large-patch14" (more accurate, 1.7GB)
            device: Device for inference ('cuda', 'mps', 'cpu', or None for auto)
            use_detailed_prompts: Use multiple descriptive prompts per food (more accurate but slower)
        """
        self.model_name = model_name
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
            from transformers import CLIPProcessor, CLIPModel

            logger.info(f"Loading CLIP model: {self.model_name}...")

            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name)
            self._model = self._model.to(self.device)
            self._model.eval()

            self._loaded = True
            logger.info(f"CLIP model loaded successfully on {self.device}")

            # Pre-compute text features for all food categories
            self._precompute_text_features()

        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise

    def _precompute_text_features(self) -> None:
        """Pre-compute text embeddings for all food categories."""
        logger.info("Pre-computing text features for food categories...")

        if self.use_detailed_prompts:
            # Use multiple prompts per food and average
            for food_key, prompts in FOOD_PROMPTS.items():
                features_list = []
                for prompt in prompts:
                    inputs = self._processor(
                        text=[prompt], return_tensors="pt", padding=True
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        text_features = self._model.get_text_features(**inputs)
                        text_features = text_features / text_features.norm(
                            dim=-1, keepdim=True
                        )
                        features_list.append(text_features)

                # Average the features from all prompts
                avg_features = torch.mean(torch.stack(features_list), dim=0)
                avg_features = avg_features / avg_features.norm(dim=-1, keepdim=True)
                self._text_features_cache[food_key] = avg_features
        else:
            # Use simple prompts (faster)
            for food_key, prompt in SIMPLE_FOOD_PROMPTS.items():
                inputs = self._processor(
                    text=[prompt], return_tensors="pt", padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    text_features = self._model.get_text_features(**inputs)
                    text_features = text_features / text_features.norm(
                        dim=-1, keepdim=True
                    )
                    self._text_features_cache[food_key] = text_features

        logger.info(
            f"Pre-computed features for {len(self._text_features_cache)} food categories"
        )

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
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        image_features = self._model.get_image_features(**inputs)
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
        img_inputs = self._processor(images=image, return_tensors="pt")
        img_inputs = {k: v.to(self.device) for k, v in img_inputs.items()}
        image_features = self._model.get_image_features(**img_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Get text features for all prompts
        text_inputs = self._processor(
            text=custom_prompts, return_tensors="pt", padding=True
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        text_features = self._model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute similarities
        similarities = (image_features @ text_features.T).squeeze()
        probs = torch.softmax(similarities * 100, dim=0)

        results = [(prompt, prob.item()) for prompt, prob in zip(custom_prompts, probs)]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

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
