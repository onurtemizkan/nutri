/**
 * Food Classification Feedback API Client
 *
 * Allows users to submit corrections when food is misclassified.
 * This helps improve the ML model over time.
 */
import api from './client';

// === Types ===

export interface FeedbackSubmitRequest {
  originalPrediction: string;
  originalConfidence: number;
  correctedLabel: string;
  imageHash: string;
  alternatives?: { name: string; confidence: number }[];
  userDescription?: string;
}

export interface FeedbackSubmitResponse {
  success: boolean;
  feedbackId: number;
  message: string;
  promptSuggestions?: string[];
}

export interface FeedbackStats {
  totalFeedback: number;
  pendingFeedback: number;
  approvedFeedback: number;
  rejectedFeedback: number;
  correctionRate: number;
  topMisclassifications: {
    original: string;
    corrected: string;
    count: number;
  }[];
  problemFoods: {
    food: string;
    correctionCount: number;
    avgConfidence: number;
  }[];
  learnedPromptsCount: number;
  activePromptsCount: number;
}

export interface PromptSuggestion {
  prompt: string;
  source: string;
  confidence: number;
  feedbackCount: number;
}

export interface PromptSuggestionsResponse {
  foodKey: string;
  currentPrompts: string[];
  suggestedPrompts: PromptSuggestion[];
  feedbackCount: number;
  commonCorrections: { confusedWith: string; count: number }[];
}

// Common food categories for quick selection
export const COMMON_FOODS = [
  // Fruits
  'apple', 'banana', 'orange', 'strawberry', 'blueberry', 'grape', 'mango',
  'watermelon', 'pineapple', 'peach', 'cherry', 'kiwi', 'avocado', 'lemon',
  'pear', 'plum', 'raspberry', 'blackberry', 'pomegranate', 'fig', 'date',
  // Vegetables
  'broccoli', 'carrot', 'tomato', 'cucumber', 'lettuce', 'spinach', 'potato',
  'onion', 'pepper', 'corn', 'mushroom', 'asparagus', 'cauliflower', 'cabbage',
  'zucchini', 'eggplant', 'celery', 'kale', 'beet', 'sweet potato', 'squash',
  // Proteins
  'chicken', 'beef', 'pork', 'salmon', 'shrimp', 'eggs', 'tofu', 'bacon',
  'turkey', 'lamb', 'tuna', 'cod', 'tilapia', 'duck', 'sausage',
  // Grains & Carbs
  'rice', 'pasta', 'bread', 'oatmeal', 'cereal', 'quinoa', 'couscous', 'barley',
  // Nuts & Seeds
  'chestnut', 'walnut', 'almond', 'peanut', 'cashew', 'pistachio', 'hazelnut',
  'pecan', 'macadamia', 'sunflower seeds', 'pumpkin seeds', 'chia seeds',
  // Prepared Foods
  'pizza', 'burger', 'sandwich', 'salad', 'soup', 'sushi', 'taco',
  'burrito', 'pasta dish', 'stir fry', 'curry', 'stew', 'casserole',
  // Breakfast
  'pancake', 'waffle', 'donut', 'muffin', 'croissant', 'bagel', 'granola',
  'french toast', 'hash browns', 'omelet', 'scrambled eggs',
  // Desserts
  'cake', 'ice cream', 'cookie', 'chocolate', 'pie', 'brownie', 'pudding',
  'cheesecake', 'tiramisu', 'cupcake', 'macaron',
  // Beverages
  'coffee', 'tea', 'juice', 'smoothie', 'milk', 'water', 'soda', 'wine', 'beer',
  // Dairy
  'cheese', 'yogurt', 'butter', 'cream', 'cottage cheese',
  // Snacks
  'chips', 'popcorn', 'nuts', 'crackers', 'pretzels', 'trail mix',
  // Legumes
  'beans', 'lentils', 'chickpeas', 'hummus', 'edamame',
];

// === Transform Functions ===

function transformFeedbackStats(data: Record<string, unknown>): FeedbackStats {
  return {
    totalFeedback: data.total_feedback as number,
    pendingFeedback: data.pending_feedback as number,
    approvedFeedback: data.approved_feedback as number,
    rejectedFeedback: data.rejected_feedback as number,
    correctionRate: data.correction_rate as number,
    topMisclassifications: (data.top_misclassifications as Array<Record<string, unknown>>)?.map(m => ({
      original: m.original as string,
      corrected: m.corrected as string,
      count: m.count as number,
    })) || [],
    problemFoods: (data.problem_foods as Array<Record<string, unknown>>)?.map(f => ({
      food: f.food as string,
      correctionCount: f.correction_count as number,
      avgConfidence: f.avg_confidence as number,
    })) || [],
    learnedPromptsCount: data.learned_prompts_count as number,
    activePromptsCount: data.active_prompts_count as number,
  };
}

function transformSuggestionsResponse(data: Record<string, unknown>): PromptSuggestionsResponse {
  return {
    foodKey: data.food_key as string,
    currentPrompts: data.current_prompts as string[],
    suggestedPrompts: (data.suggested_prompts as Array<Record<string, unknown>>)?.map(s => ({
      prompt: s.prompt as string,
      source: s.source as string,
      confidence: s.confidence as number,
      feedbackCount: s.feedback_count as number,
    })) || [],
    feedbackCount: data.feedback_count as number,
    commonCorrections: (data.common_corrections as Array<Record<string, unknown>>)?.map(c => ({
      confusedWith: c.confused_with as string,
      count: c.count as number,
    })) || [],
  };
}

// === API Client ===

class FoodFeedbackAPI {
  /**
   * Submit feedback when food classification is incorrect
   */
  async submitFeedback(request: FeedbackSubmitRequest): Promise<FeedbackSubmitResponse> {
    try {
      const response = await api.post('/food/feedback', {
        original_prediction: request.originalPrediction,
        original_confidence: request.originalConfidence,
        corrected_label: request.correctedLabel,
        image_hash: request.imageHash,
        alternatives: request.alternatives,
        user_description: request.userDescription,
      });

      return {
        success: response.data.success,
        feedbackId: response.data.feedback_id,
        message: response.data.message,
        promptSuggestions: response.data.prompt_suggestions,
      };
    } catch (error) {
      console.error('Feedback submission error:', error);
      throw error;
    }
  }

  /**
   * Get feedback statistics (useful for debugging/admin)
   */
  async getStats(): Promise<FeedbackStats> {
    try {
      const response = await api.get('/food/feedback/stats');
      return transformFeedbackStats(response.data);
    } catch (error) {
      console.error('Get stats error:', error);
      throw error;
    }
  }

  /**
   * Get prompt suggestions for a specific food
   */
  async getPromptSuggestions(foodKey: string): Promise<PromptSuggestionsResponse> {
    try {
      const response = await api.get(`/food/feedback/suggestions/${encodeURIComponent(foodKey)}`);
      return transformSuggestionsResponse(response.data);
    } catch (error) {
      console.error('Get suggestions error:', error);
      throw error;
    }
  }

  /**
   * Search common foods by query
   */
  searchFoods(query: string): string[] {
    if (!query || query.length < 2) {
      return COMMON_FOODS.slice(0, 20);
    }

    const normalizedQuery = query.toLowerCase().trim();
    return COMMON_FOODS.filter(food =>
      food.toLowerCase().includes(normalizedQuery)
    ).slice(0, 20);
  }
}

// Export singleton instance
export const foodFeedbackApi = new FoodFeedbackAPI();
