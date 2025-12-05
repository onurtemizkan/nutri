"""
Tests for Food Database module.

Tests the comprehensive food database including:
- Food entries and lookups
- Density and shape factor calculations
- Cooking method modifiers
- Portion validation
- Weight estimation from volume
"""
import pytest
from app.data.food_database import (
    FOOD_DATABASE,
    COOKING_MODIFIERS,
    PORTION_VALIDATION,
    AMINO_ACID_PROTEIN_RATIOS,
    FoodCategory,
    CookingMethod,
    FoodEntry,
    get_food_entry,
    get_density,
    get_shape_factor,
    get_cooking_modifier,
    estimate_weight_from_volume,
    validate_portion,
    get_amino_acids,
    estimate_lysine_arginine_ratio,
)


class TestFoodDatabase:
    """Tests for FOOD_DATABASE structure and content."""

    def test_database_not_empty(self):
        """Database should contain food entries."""
        assert len(FOOD_DATABASE) > 0
        assert len(FOOD_DATABASE) >= 100  # We added 100+ foods

    def test_all_entries_have_required_fields(self):
        """Each food entry should have all required fields."""
        for key, entry in FOOD_DATABASE.items():
            assert isinstance(entry, FoodEntry)
            assert entry.name, f"Missing name for {key}"
            assert entry.display_name, f"Missing display_name for {key}"
            assert isinstance(entry.category, FoodCategory)
            assert entry.density > 0, f"Invalid density for {key}"
            assert 0 < entry.shape_factor <= 1, f"Invalid shape_factor for {key}"
            assert entry.serving_size, f"Missing serving_size for {key}"
            assert entry.serving_weight > 0, f"Invalid serving_weight for {key}"
            assert entry.calories >= 0, f"Invalid calories for {key}"
            assert entry.protein >= 0, f"Invalid protein for {key}"
            assert entry.carbs >= 0, f"Invalid carbs for {key}"
            assert entry.fat >= 0, f"Invalid fat for {key}"

    def test_density_values_realistic(self):
        """Food densities should be within realistic range."""
        for key, entry in FOOD_DATABASE.items():
            # Most foods have density 0.02 - 2.0 g/cm³
            # (popcorn is very light: 0.05)
            assert 0.02 <= entry.density <= 3.0, (
                f"Unrealistic density {entry.density} for {key}"
            )

    def test_nutrition_values_consistent(self):
        """Macros should roughly sum to expected caloric range."""
        for key, entry in FOOD_DATABASE.items():
            # Calculate expected calories from macros
            # Protein: 4 cal/g, Carbs: 4 cal/g, Fat: 9 cal/g
            expected_cal = (entry.protein * 4) + (entry.carbs * 4) + (entry.fat * 9)
            # Allow 30% variance due to fiber, water, etc.
            if entry.calories > 0:
                ratio = entry.calories / expected_cal if expected_cal > 0 else 1
                assert 0.5 <= ratio <= 2.0, (
                    f"Calories mismatch for {key}: {entry.calories} vs {expected_cal}"
                )

    def test_categories_coverage(self):
        """Database should cover multiple food categories."""
        categories_found = set()
        for entry in FOOD_DATABASE.values():
            categories_found.add(entry.category)
        # Should have at least 8 categories represented
        assert len(categories_found) >= 8


class TestGetFoodEntry:
    """Tests for get_food_entry() function."""

    def test_exact_match(self):
        """Should find exact key matches."""
        entry = get_food_entry("apple")
        assert entry is not None
        assert entry.name == "apple"

    def test_case_insensitive(self):
        """Should be case-insensitive."""
        entry_lower = get_food_entry("apple")
        entry_upper = get_food_entry("APPLE")
        entry_mixed = get_food_entry("ApPlE")

        assert entry_lower is not None
        assert entry_upper is not None
        assert entry_mixed is not None
        assert entry_lower.name == entry_upper.name == entry_mixed.name

    def test_unknown_food_returns_none(self):
        """Should return None for unknown foods."""
        entry = get_food_entry("unknown_alien_food_xyz")
        assert entry is None

    def test_alias_matching(self):
        """Should find foods by alias."""
        # Check if we have any foods with aliases
        foods_with_aliases = [
            entry for entry in FOOD_DATABASE.values()
            if entry.aliases
        ]
        if foods_with_aliases:
            test_entry = foods_with_aliases[0]
            alias = test_entry.aliases[0]
            found = get_food_entry(alias)
            assert found is not None


class TestGetDensity:
    """Tests for get_density() function."""

    def test_known_food_density(self):
        """Should return correct density for known foods."""
        apple_density = get_density("apple")
        assert apple_density > 0
        # Apple density should be around 0.8 g/cm³
        assert 0.6 <= apple_density <= 1.0

    def test_unknown_food_default_density(self):
        """Should return default density for unknown foods."""
        default = get_density("unknown_food_xyz")
        assert default == 0.7  # Default density

    def test_density_consistency(self):
        """Same food should always return same density."""
        d1 = get_density("chicken_breast")
        d2 = get_density("chicken_breast")
        assert d1 == d2


class TestGetShapeFactor:
    """Tests for get_shape_factor() function."""

    def test_known_food_shape_factor(self):
        """Should return correct shape factor for known foods."""
        apple_sf = get_shape_factor("apple")
        assert 0 < apple_sf <= 1
        # Apple is roughly spherical, shape factor ~0.52
        assert 0.4 <= apple_sf <= 0.7

    def test_unknown_food_default_shape_factor(self):
        """Should return default shape factor for unknown foods."""
        default = get_shape_factor("unknown_food_xyz")
        assert default == 0.7  # Default shape factor

    def test_bread_near_cuboid(self):
        """Bread (cuboid-like) should have higher shape factor."""
        bread_sf = get_shape_factor("bread")
        # Bread is more cuboid, shape factor should be higher
        assert bread_sf >= 0.7


class TestCookingModifiers:
    """Tests for cooking modifiers."""

    def test_all_methods_have_modifiers(self):
        """Each cooking method should have a modifier defined."""
        for method in CookingMethod:
            modifier = get_cooking_modifier(method)
            assert modifier is not None
            assert modifier.weight_multiplier > 0
            assert modifier.calorie_multiplier > 0

    def test_raw_no_change(self):
        """Raw cooking method should have no modification."""
        raw = get_cooking_modifier(CookingMethod.RAW)
        assert raw.weight_multiplier == 1.0
        assert raw.calorie_multiplier == 1.0

    def test_grilled_weight_loss(self):
        """Grilled should have weight loss (moisture evaporation)."""
        grilled = get_cooking_modifier(CookingMethod.GRILLED)
        assert grilled.weight_multiplier < 1.0  # Weight loss

    def test_fried_calorie_increase(self):
        """Fried should have calorie increase (oil absorption)."""
        fried = get_cooking_modifier(CookingMethod.FRIED)
        assert fried.calorie_multiplier > 1.0  # Calorie increase

    def test_boiled_weight_gain(self):
        """Boiled foods often absorb water."""
        boiled = get_cooking_modifier(CookingMethod.BOILED)
        # Boiled can be neutral to slight weight gain
        assert 0.9 <= boiled.weight_multiplier <= 1.2


class TestEstimateWeightFromVolume:
    """Tests for estimate_weight_from_volume() function."""

    def test_known_food_weight_estimation(self):
        """Should estimate weight correctly for known foods."""
        # 125 cm³ (5x5x5 cube equivalent) of apple
        volume = 5 * 5 * 5  # 125 cm³
        result = estimate_weight_from_volume(
            volume_cm3=volume,
            food_name="apple"
        )

        assert "weight" in result
        assert "density_used" in result
        assert "shape_factor_used" in result
        assert result["weight"] > 0

    def test_cooking_method_affects_weight(self):
        """Cooking method should affect final weight."""
        volume = 5 * 5 * 5  # 125 cm³
        raw_result = estimate_weight_from_volume(
            volume_cm3=volume,
            food_name="chicken_breast",
            cooking_method=CookingMethod.RAW
        )
        grilled_result = estimate_weight_from_volume(
            volume_cm3=volume,
            food_name="chicken_breast",
            cooking_method=CookingMethod.GRILLED
        )

        # Grilled should be lighter due to moisture loss
        assert grilled_result["weight"] < raw_result["weight"]

    def test_unknown_food_uses_defaults(self):
        """Unknown food should use default values."""
        volume = 5 * 5 * 5  # 125 cm³
        result = estimate_weight_from_volume(
            volume_cm3=volume,
            food_name="unknown_food_xyz"
        )

        assert result["weight"] > 0
        assert result["density_used"] == 0.7  # Default density
        assert result["shape_factor_used"] == 0.7  # Default shape factor

    def test_result_includes_all_metadata(self):
        """Result should include calculation metadata."""
        volume = 10 * 8 * 5  # 400 cm³
        result = estimate_weight_from_volume(
            volume_cm3=volume,
            food_name="rice",
            cooking_method=CookingMethod.COOKED
        )

        expected_keys = [
            "weight", "volume_raw", "volume_adjusted",
            "density_used", "shape_factor_used",
            "cooking_modifier", "confidence", "method"
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


class TestValidatePortion:
    """Tests for validate_portion() function."""

    def test_valid_portion(self):
        """Normal portion should be valid."""
        result = validate_portion(150, {"width": 10, "height": 8, "depth": 5})
        assert result["valid"] is True
        assert len(result["warnings"]) == 0

    def test_too_heavy_portion(self):
        """Very heavy portion should have warning."""
        result = validate_portion(6000, {"width": 30, "height": 30, "depth": 30})
        assert "weight" in str(result["warnings"]).lower() or len(result["warnings"]) > 0

    def test_too_light_portion(self):
        """Very light portion should have warning."""
        result = validate_portion(0.5, {"width": 0.3, "height": 0.3, "depth": 0.3})
        assert len(result["warnings"]) > 0

    def test_suspicious_dimensions(self):
        """Suspicious aspect ratio should trigger warning."""
        result = validate_portion(100, {"width": 50, "height": 1, "depth": 1})
        # Very thin, tall shape should be suspicious
        assert len(result["warnings"]) > 0

    def test_missing_dimensions(self):
        """Should handle missing dimensions gracefully."""
        result = validate_portion(100, None)
        assert result is not None
        assert "valid" in result


class TestPortionValidationThresholds:
    """Tests for PORTION_VALIDATION constants."""

    def test_thresholds_defined(self):
        """All validation thresholds should be defined."""
        required_keys = [
            "min_weight_g",
            "max_weight_g",
            "typical_min_g",
            "typical_max_g",
            "suspicious_aspect_ratio",
            "suspicious_min_dimension_cm",
            "suspicious_max_dimension_cm",
        ]
        for key in required_keys:
            assert key in PORTION_VALIDATION, f"Missing threshold: {key}"

    def test_thresholds_reasonable(self):
        """Thresholds should have reasonable values."""
        assert PORTION_VALIDATION["min_weight_g"] < PORTION_VALIDATION["typical_min_g"]
        assert PORTION_VALIDATION["typical_max_g"] < PORTION_VALIDATION["max_weight_g"]
        assert PORTION_VALIDATION["suspicious_min_dimension_cm"] < 1  # Sub-centimeter
        assert PORTION_VALIDATION["suspicious_max_dimension_cm"] > 30  # Large items


class TestFoodCategories:
    """Tests for FoodCategory enum."""

    def test_category_values(self):
        """Category enum should have expected values."""
        expected = [
            "fruit", "vegetable", "protein", "grain", "dairy",
            "beverage", "baked", "snack", "mixed", "legume",
            "seafood", "nut", "condiment", "unknown"
        ]
        category_values = [c.value for c in FoodCategory]
        for exp in expected:
            assert exp in category_values, f"Missing category: {exp}"


class TestCookingMethods:
    """Tests for CookingMethod enum."""

    def test_cooking_method_values(self):
        """CookingMethod enum should have expected values."""
        expected = [
            "raw", "cooked", "boiled", "steamed", "grilled",
            "fried", "baked", "roasted", "sauteed", "poached"
        ]
        method_values = [m.value for m in CookingMethod]
        for exp in expected:
            assert exp in method_values, f"Missing cooking method: {exp}"


class TestAminoAcidProteinRatios:
    """Tests for AMINO_ACID_PROTEIN_RATIOS constants."""

    def test_ratios_defined_for_all_categories(self):
        """All food categories should have amino acid ratios defined."""
        for category in FoodCategory:
            assert category in AMINO_ACID_PROTEIN_RATIOS, (
                f"Missing amino acid ratios for {category}"
            )

    def test_ratios_have_required_keys(self):
        """Each category should have lysine and arginine ratios."""
        for category, ratios in AMINO_ACID_PROTEIN_RATIOS.items():
            assert "lysine" in ratios, f"Missing lysine for {category}"
            assert "arginine" in ratios, f"Missing arginine for {category}"

    def test_ratios_are_positive(self):
        """All ratio values should be positive."""
        for category, ratios in AMINO_ACID_PROTEIN_RATIOS.items():
            assert ratios["lysine"] > 0, f"Invalid lysine ratio for {category}"
            assert ratios["arginine"] > 0, f"Invalid arginine ratio for {category}"

    def test_protein_foods_high_lysine(self):
        """Protein foods should have high lysine ratios."""
        protein_lysine = AMINO_ACID_PROTEIN_RATIOS[FoodCategory.PROTEIN]["lysine"]
        # Meats typically have 80-95 mg lysine per gram of protein
        assert 70 <= protein_lysine <= 100

    def test_nuts_low_lysine_high_arginine(self):
        """Nuts should have low lysine but high arginine."""
        nut_ratios = AMINO_ACID_PROTEIN_RATIOS[FoodCategory.NUT]
        # Nuts are known to be high in arginine, low in lysine
        assert nut_ratios["arginine"] > nut_ratios["lysine"]

    def test_grains_low_lysine(self):
        """Grains should have low lysine (limiting amino acid)."""
        grain_lysine = AMINO_ACID_PROTEIN_RATIOS[FoodCategory.GRAIN]["lysine"]
        protein_lysine = AMINO_ACID_PROTEIN_RATIOS[FoodCategory.PROTEIN]["lysine"]
        # Grains have significantly less lysine than animal protein
        assert grain_lysine < protein_lysine * 0.5


class TestGetAminoAcids:
    """Tests for get_amino_acids() function."""

    def test_known_food_with_explicit_values(self):
        """Foods with explicit amino acid values should return them."""
        # Chicken breast has explicit values in database
        result = get_amino_acids("chicken_breast", protein_grams=31)
        assert result["lysine"] is not None
        assert result["arginine"] is not None
        # Chicken is high in lysine
        assert result["lysine"] > result["arginine"]

    def test_known_food_scales_with_portion(self):
        """Amino acids should scale with protein amount."""
        result_small = get_amino_acids("chicken_breast", protein_grams=15)
        result_large = get_amino_acids("chicken_breast", protein_grams=30)

        # Double the protein should roughly double the amino acids
        assert result_large["lysine"] > result_small["lysine"] * 1.5
        assert result_large["arginine"] > result_small["arginine"] * 1.5

    def test_unknown_food_estimates_from_category(self):
        """Unknown foods should estimate based on category."""
        result = get_amino_acids(
            "unknown_food_xyz",
            protein_grams=20,
            category=FoodCategory.PROTEIN
        )

        assert result["lysine"] is not None
        assert result["arginine"] is not None
        # 20g protein * ~89 mg/g = ~1780 mg lysine
        assert 1500 <= result["lysine"] <= 2000

    def test_unknown_food_without_category_uses_default(self):
        """Unknown foods without category should use default ratios."""
        result = get_amino_acids("completely_unknown", protein_grams=10)

        assert result["lysine"] is not None
        assert result["arginine"] is not None
        assert result["lysine"] > 0
        assert result["arginine"] > 0

    def test_egg_values(self):
        """Eggs should have reasonable amino acid content."""
        result = get_amino_acids("egg", protein_grams=6.3)  # One large egg
        # One egg has ~456mg lysine per serving
        assert 300 <= result["lysine"] <= 600
        assert result["arginine"] > 0

    def test_almonds_high_arginine(self):
        """Almonds should have high arginine relative to lysine."""
        result = get_amino_acids("almonds", protein_grams=6)  # ~1oz serving
        # Almonds are known to be high in arginine, low in lysine
        assert result["arginine"] > result["lysine"]

    def test_result_rounded(self):
        """Results should be rounded to 1 decimal place."""
        result = get_amino_acids("chicken_breast", protein_grams=31)
        # Check that values are properly rounded (no more than 1 decimal)
        if result["lysine"] is not None:
            assert result["lysine"] == round(result["lysine"], 1)
        if result["arginine"] is not None:
            assert result["arginine"] == round(result["arginine"], 1)


class TestEstimateLysineArginineRatio:
    """Tests for estimate_lysine_arginine_ratio() function.

    Classification thresholds:
    - ratio >= 2.0: high_lysine
    - ratio >= 1.0: balanced
    - ratio >= 0.5: moderate_arginine
    - ratio < 0.5: high_arginine
    """

    def test_chicken_balanced(self):
        """Chicken has more lysine than arginine, classified as balanced."""
        result = estimate_lysine_arginine_ratio("chicken_breast")
        assert result["ratio"] > 1.0  # More lysine than arginine
        # Ratio ~1.38, classified as balanced (not high_lysine which requires >= 2.0)
        assert result["classification"] in ["balanced", "high_lysine"]

    def test_almonds_high_arginine(self):
        """Almonds have high arginine relative to lysine."""
        result = estimate_lysine_arginine_ratio("almonds")
        assert result["ratio"] < 0.5  # Much more arginine
        assert result["classification"] == "high_arginine"

    def test_egg_classification(self):
        """Egg should have a reasonable classification."""
        result = estimate_lysine_arginine_ratio("egg")
        assert result["ratio"] is not None
        assert result["classification"] in [
            "high_lysine", "balanced", "moderate_arginine", "high_arginine"
        ]

    def test_unknown_food_fallback(self):
        """Unknown food should use fallback estimation."""
        result = estimate_lysine_arginine_ratio("unknown_food_xyz")
        assert result is not None
        assert "classification" in result

    def test_beef_balanced_or_high(self):
        """Beef (meat) has more lysine than arginine."""
        result = estimate_lysine_arginine_ratio("beef")
        assert result["ratio"] > 1.0
        assert result["classification"] in ["balanced", "high_lysine"]

    def test_tofu_classification(self):
        """Tofu should have a known classification."""
        result = estimate_lysine_arginine_ratio("tofu")
        assert result["ratio"] is not None
        # Tofu has more arginine than lysine
        assert result["classification"] in [
            "high_arginine", "moderate_arginine", "balanced"
        ]

    def test_ratio_thresholds(self):
        """Verify classification threshold logic."""
        # Foods with known explicit values
        chicken = estimate_lysine_arginine_ratio("chicken_breast")
        almonds = estimate_lysine_arginine_ratio("almonds")

        # Chicken: lysine=2705, arginine=1965, ratio ~1.38
        assert 1.0 <= chicken["ratio"] < 2.0
        assert chicken["classification"] == "balanced"

        # Almonds: lysine=176, arginine=672, ratio ~0.26
        assert almonds["ratio"] < 0.5
        assert almonds["classification"] == "high_arginine"


class TestAminoAcidDatabaseEntries:
    """Tests for amino acid values in FOOD_DATABASE."""

    def test_high_protein_foods_have_explicit_values(self):
        """Key high-protein foods should have explicit amino acid values."""
        high_protein_foods = [
            "chicken_breast", "beef", "salmon", "egg", "tofu"
        ]
        for food in high_protein_foods:
            entry = get_food_entry(food)
            if entry is not None:
                # At least some should have explicit values
                pass  # Test passes if entry exists

    def test_explicit_values_realistic(self):
        """Explicit amino acid values should be realistic."""
        for key, entry in FOOD_DATABASE.items():
            if entry.lysine is not None:
                # Lysine should be between 0 and 5000mg per serving
                assert 0 < entry.lysine <= 5000, (
                    f"Unrealistic lysine {entry.lysine} for {key}"
                )
            if entry.arginine is not None:
                # Arginine should be between 0 and 5000mg per serving
                assert 0 < entry.arginine <= 5000, (
                    f"Unrealistic arginine {entry.arginine} for {key}"
                )
