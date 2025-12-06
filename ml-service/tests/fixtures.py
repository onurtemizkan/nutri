"""
Test Fixtures for Integration Testing

Realistic test data representing:
- User: John (athlete tracking RHR and HRV)
- 90 days of data (sufficient for training)
- Meals, activities, health metrics
- Varied patterns (high protein days, rest days, intense workouts)
"""

from datetime import date, datetime, timedelta
from typing import Dict, List
import random

import numpy as np


class TestDataGenerator:
    """Generate realistic test data for ML Engine testing."""

    def __init__(self, seed: int = 42):
        """Initialize with seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)

        self.user_id = "test_user_john"
        self.start_date = date.today() - timedelta(days=90)
        self.end_date = date.today()

    # ========================================================================
    # User Data
    # ========================================================================

    def generate_user(self) -> Dict:
        """Generate test user (John - athlete)."""
        return {
            "id": self.user_id,
            "email": "john@test.com",
            "password": "test_password_hash",  # In real app, this would be hashed
            "name": "John Doe",
            "created_at": datetime.now() - timedelta(days=180),
            # User profile for personalization
            "height": 180.0,  # cm
            "current_weight": 75.0,  # kg
            "goal_weight": 73.0,
            "goal_calories": 2500,
            "goal_protein": 150.0,
            "goal_carbs": 300.0,
            "goal_fat": 80.0,
            "activity_level": "very_active",  # Athlete
        }

    # ========================================================================
    # Meal Data (Nutrition Tracking)
    # ========================================================================

    def generate_meals(self) -> List[Dict]:
        """
        Generate 90 days of realistic meals.

        Patterns:
        - Baseline: ~2500 cal, 150g protein, 300g carbs, 80g fat
        - High protein days (2x/week): 200g protein
        - Rest days (1x/week): Lower calories
        - Late night eating varies (0-100g carbs)
        """
        meals = []
        current_date = self.start_date

        day_count = 0
        while current_date <= self.end_date:
            day_type = self._get_day_type(day_count)

            # Generate 3-5 meals per day
            num_meals = random.randint(3, 5)

            for meal_num in range(num_meals):
                meal = self._generate_meal(current_date, meal_num, num_meals, day_type)
                meals.append(meal)

            current_date += timedelta(days=1)
            day_count += 1

        return meals

    def _get_day_type(self, day_count: int) -> str:
        """Determine day type based on patterns."""
        if day_count % 7 == 6:  # Every Sunday
            return "rest"
        elif day_count % 3 == 0:  # Every 3rd day
            return "high_protein"
        elif day_count % 7 == 5:  # Saturday
            return "cheat"  # Higher carbs/calories
        else:
            return "normal"

    def _generate_meal(
        self, date: date, meal_num: int, total_meals: int, day_type: str
    ) -> Dict:
        """Generate single meal with realistic macros."""
        # Meal timing
        if meal_num == 0:  # Breakfast
            hour = random.randint(7, 9)
            name = "Breakfast"
        elif meal_num == total_meals - 1:  # Dinner
            hour = random.randint(18, 21)
            name = "Dinner"
        elif meal_num == 1:  # Lunch
            hour = random.randint(12, 14)
            name = "Lunch"
        else:  # Snack
            hour = random.randint(15, 17)
            name = f"Snack {meal_num - 1}"

        # Base macros (divided across meals)
        if day_type == "normal":
            daily_protein = 150
            daily_carbs = 300
            daily_fat = 80
        elif day_type == "high_protein":
            daily_protein = 200
            daily_carbs = 250
            daily_fat = 70
        elif day_type == "rest":
            daily_protein = 120
            daily_carbs = 200
            daily_fat = 60
        else:  # cheat
            daily_protein = 130
            daily_carbs = 400
            daily_fat = 100

        # Distribute macros across meals
        protein = (daily_protein / total_meals) + random.uniform(-10, 10)
        carbs = (daily_carbs / total_meals) + random.uniform(-20, 20)
        fat = (daily_fat / total_meals) + random.uniform(-5, 5)

        # Late night eating (after 8pm)
        if hour >= 20:
            carbs += random.uniform(0, 50)  # Extra late night carbs

        # Calculate calories (4 cal/g protein & carbs, 9 cal/g fat)
        calories = (protein * 4) + (carbs * 4) + (fat * 9)

        # Add fiber
        fiber = max(0, carbs * 0.1 + random.uniform(-2, 2))

        # Determine meal type
        if meal_num == 0:
            meal_type = "breakfast"
        elif meal_num == 1:
            meal_type = "lunch"
        elif meal_num == total_meals - 1:
            meal_type = "dinner"
        else:
            meal_type = "snack"

        return {
            "id": f"meal_{date.isoformat()}_{meal_num}",
            "user_id": self.user_id,
            "name": name,
            "meal_type": meal_type,
            "consumed_at": datetime.combine(
                date, datetime.min.time().replace(hour=hour)
            ),
            "calories": round(calories, 1),
            "protein": round(protein, 1),
            "carbs": round(carbs, 1),
            "fat": round(fat, 1),
            "fiber": round(fiber, 1),
        }

    # ========================================================================
    # Activity Data (Workouts)
    # ========================================================================

    def generate_activities(self) -> List[Dict]:
        """
        Generate 90 days of realistic activities.

        Patterns:
        - Workout 5-6 days/week
        - Intensity varies (0.4-0.9)
        - Rest days: minimal activity
        - Duration: 45-90 minutes
        """
        activities = []
        current_date = self.start_date

        day_count = 0
        while current_date <= self.end_date:
            day_type = self._get_day_type(day_count)

            # Rest day: no workout
            if day_type == "rest":
                # Light walk only
                activity = self._generate_light_activity(current_date)
                if activity:
                    activities.append(activity)

            # Normal training day
            else:
                # 85% chance of workout
                if random.random() < 0.85:
                    activity = self._generate_workout(current_date, day_type)
                    activities.append(activity)

            current_date += timedelta(days=1)
            day_count += 1

        return activities

    def _generate_workout(self, date: date, day_type: str) -> Dict:
        """Generate realistic workout."""
        # Workout time (morning or evening)
        if random.random() < 0.6:  # 60% morning
            hour = random.randint(6, 8)
        else:  # 40% evening
            hour = random.randint(17, 19)

        # Intensity varies by day (store numeric for correlations, convert to string for model)
        if day_type == "high_protein":
            # High protein days = hard training
            intensity_numeric = random.uniform(0.75, 0.95)
            intensity_str = "high"
            duration_min = random.randint(60, 90)
        elif day_type == "cheat":
            # Weekend = moderate
            intensity_numeric = random.uniform(0.5, 0.7)
            intensity_str = "medium"
            duration_min = random.randint(45, 60)
        else:
            # Normal training
            intensity_numeric = random.uniform(0.6, 0.8)
            intensity_str = "medium"
            duration_min = random.randint(50, 75)

        # Calories burned (rough estimate)
        calories_burned = duration_min * intensity_numeric * 10

        started_at = datetime.combine(date, datetime.min.time().replace(hour=hour))
        ended_at = started_at + timedelta(minutes=duration_min)

        return {
            "id": f"activity_{date.isoformat()}",
            "user_id": self.user_id,
            "activity_type": "strength_training",
            "started_at": started_at,
            "ended_at": ended_at,
            "duration": duration_min,
            "intensity": intensity_str,
            "intensity_numeric": intensity_numeric,  # For correlation analysis
            "calories_burned": round(calories_burned, 1),
            "source": "manual",
        }

    def _generate_light_activity(self, date: date) -> Dict:
        """Generate light activity for rest day."""
        if random.random() < 0.7:  # 70% chance of light walk
            duration_min = random.randint(20, 40)
            started_at = datetime.combine(date, datetime.min.time().replace(hour=10))
            ended_at = started_at + timedelta(minutes=duration_min)

            return {
                "id": f"activity_{date.isoformat()}_walk",
                "user_id": self.user_id,
                "activity_type": "walking",
                "started_at": started_at,
                "ended_at": ended_at,
                "duration": duration_min,
                "intensity": "low",
                "intensity_numeric": 0.3,  # For correlation analysis
                "calories_burned": random.randint(80, 150),
                "source": "manual",
            }
        return None

    # ========================================================================
    # Health Metrics (RHR, HRV - Ground Truth)
    # ========================================================================

    def generate_health_metrics(
        self, meals: List[Dict], activities: List[Dict]
    ) -> List[Dict]:
        """
        Generate realistic health metrics (RHR, HRV).

        Simulates realistic relationships:
        - High protein → Lower RHR (better recovery)
        - High workout intensity → Higher RHR next day (fatigue)
        - Late night carbs → Higher RHR
        - Good recovery → Higher HRV
        - Hard training → Lower HRV next day

        This creates realistic correlations for the ML model to learn!
        """
        metrics = []

        # Baseline values
        baseline_rhr = 55  # Athlete baseline
        baseline_hrv = 65  # SDNN

        # Group meals and activities by date
        meals_by_date = self._group_by_date(meals)
        activities_by_date = self._group_by_date(activities)

        current_date = self.start_date
        previous_rhr = baseline_rhr
        previous_hrv = baseline_hrv

        while current_date <= self.end_date:
            date_str = current_date.isoformat()

            # Get yesterday's data (affects today's metrics)
            yesterday = current_date - timedelta(days=1)
            yesterday_str = yesterday.isoformat()

            yesterday_meals = meals_by_date.get(yesterday_str, [])
            yesterday_activities = activities_by_date.get(yesterday_str, [])

            # Calculate influences on RHR
            rhr = baseline_rhr

            # 1. Protein influence (high protein → -2 BPM)
            if yesterday_meals:
                total_protein = sum(m["protein"] for m in yesterday_meals)
                if total_protein > 180:
                    rhr -= 2
                elif total_protein > 150:
                    rhr -= 1

            # 2. Late night carbs influence (+1-3 BPM)
            if yesterday_meals:
                late_night_carbs = sum(
                    m["carbs"] for m in yesterday_meals if m["consumed_at"].hour >= 20
                )
                if late_night_carbs > 50:
                    rhr += min(3, late_night_carbs / 30)

            # 3. Workout intensity influence (high intensity → +2-4 BPM next day)
            if yesterday_activities:
                max_intensity = max(
                    (a.get("intensity_numeric", 0) for a in yesterday_activities),
                    default=0,
                )
                if max_intensity > 0.8:
                    rhr += 3
                elif max_intensity > 0.6:
                    rhr += 1

            # 4. Recovery momentum (moving toward baseline)
            rhr = rhr * 0.7 + previous_rhr * 0.3

            # Add noise
            rhr += random.uniform(-1, 1)

            # Calculate influences on HRV
            hrv = baseline_hrv

            # 1. Hard training → Lower HRV next day
            if yesterday_activities:
                max_intensity = max(
                    (a.get("intensity_numeric", 0) for a in yesterday_activities),
                    default=0,
                )
                if max_intensity > 0.8:
                    hrv -= 8
                elif max_intensity > 0.6:
                    hrv -= 3

            # 2. Good recovery (high protein, low late carbs) → Higher HRV
            if yesterday_meals:
                total_protein = sum(m["protein"] for m in yesterday_meals)
                late_night_carbs = sum(
                    m["carbs"] for m in yesterday_meals if m["consumed_at"].hour >= 20
                )
                if total_protein > 180 and late_night_carbs < 30:
                    hrv += 5

            # 3. Rest day → HRV recovery
            if not yesterday_activities or all(
                a.get("intensity_numeric", 1) < 0.4 for a in yesterday_activities
            ):
                hrv += 3

            # 4. Recovery momentum
            hrv = hrv * 0.7 + previous_hrv * 0.3

            # Add noise
            hrv += random.uniform(-2, 2)

            # Bounds
            rhr = max(45, min(75, rhr))
            hrv = max(40, min(90, hrv))

            # Create metrics (recorded in morning)
            metrics.append(
                {
                    "id": f"metric_rhr_{date_str}",
                    "user_id": self.user_id,
                    "metric_type": "RESTING_HEART_RATE",
                    "value": round(rhr, 1),
                    "unit": "bpm",
                    "source": "manual",
                    "recorded_at": datetime.combine(
                        current_date, datetime.min.time().replace(hour=7)
                    ),
                }
            )

            metrics.append(
                {
                    "id": f"metric_hrv_{date_str}",
                    "user_id": self.user_id,
                    "metric_type": "HEART_RATE_VARIABILITY_SDNN",
                    "value": round(hrv, 1),
                    "unit": "ms",
                    "source": "manual",
                    "recorded_at": datetime.combine(
                        current_date, datetime.min.time().replace(hour=7)
                    ),
                }
            )

            # Update previous values
            previous_rhr = rhr
            previous_hrv = hrv

            current_date += timedelta(days=1)

        return metrics

    def _group_by_date(self, items: List[Dict]) -> Dict[str, List[Dict]]:
        """Group items by date."""
        grouped = {}
        for item in items:
            # Get date from consumed_at or started_at
            dt = item.get("consumed_at") or item.get("started_at")
            if dt:
                date_str = dt.date().isoformat()
                if date_str not in grouped:
                    grouped[date_str] = []
                grouped[date_str].append(item)
        return grouped

    # ========================================================================
    # Complete Test Dataset
    # ========================================================================

    def generate_complete_dataset(self) -> Dict:
        """
        Generate complete 90-day dataset.

        Returns:
            Dictionary with user, meals, activities, health_metrics
        """
        print("🧪 Generating test dataset...")
        print(f"   Period: {self.start_date} to {self.end_date} (90 days)")

        user = self.generate_user()
        print(f"✅ User: {user['name']}")

        meals = self.generate_meals()
        print(f"✅ Meals: {len(meals)} generated")

        activities = self.generate_activities()
        print(f"✅ Activities: {len(activities)} generated")

        health_metrics = self.generate_health_metrics(meals, activities)
        print(f"✅ Health Metrics: {len(health_metrics)} generated (RHR + HRV)")

        # Summary stats
        total_days = (self.end_date - self.start_date).days + 1
        avg_meals_per_day = len(meals) / total_days
        avg_protein = np.mean([m["protein"] for m in meals])
        workout_days = len(
            [a for a in activities if a.get("intensity_numeric", 0) > 0.5]
        )

        print("\n📊 Dataset Summary:")
        print(f"   Days: {total_days}")
        print(f"   Meals per day: {avg_meals_per_day:.1f}")
        print(f"   Avg protein per meal: {avg_protein:.1f}g")
        print(f"   Workout days: {workout_days}")

        rhr_values = [
            m["value"]
            for m in health_metrics
            if m["metric_type"] == "RESTING_HEART_RATE"
        ]
        hrv_values = [
            m["value"]
            for m in health_metrics
            if m["metric_type"] == "HEART_RATE_VARIABILITY_SDNN"
        ]

        print(f"   RHR range: {min(rhr_values):.1f} - {max(rhr_values):.1f} BPM")
        print(f"   HRV range: {min(hrv_values):.1f} - {max(hrv_values):.1f} ms")

        return {
            "user": user,
            "meals": meals,
            "activities": activities,
            "health_metrics": health_metrics,
        }


# Singleton instance for easy import
test_data_generator = TestDataGenerator()
