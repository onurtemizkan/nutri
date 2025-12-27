/**
 * Application-wide constants
 * Centralizes magic numbers and strings for better maintainability
 */

// ============================================================================
// QUERY DEFAULTS
// ============================================================================

/**
 * Default pagination limits
 */
export const DEFAULT_PAGE_LIMIT = 100;
export const MAX_PAGE_LIMIT = 1000;

/**
 * Default time periods (in days)
 */
export const DEFAULT_TIME_PERIOD_DAYS = 30;
export const WEEK_IN_DAYS = 7;
export const MONTH_IN_DAYS = 30;

// ============================================================================
// VALIDATION
// ============================================================================

/**
 * Password requirements
 * OWASP recommends minimum 12 characters for modern password policies
 * @see https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html
 */
export const MIN_PASSWORD_LENGTH = 12;
export const MAX_PASSWORD_LENGTH = 128;

/**
 * Password validation error messages
 */
export const PASSWORD_ERRORS = {
  TOO_SHORT: `Password must be at least ${MIN_PASSWORD_LENGTH} characters`,
  TOO_LONG: `Password must be at most ${MAX_PASSWORD_LENGTH} characters`,
  MISSING_UPPERCASE: 'Password must contain at least one uppercase letter',
  MISSING_LOWERCASE: 'Password must contain at least one lowercase letter',
  MISSING_NUMBER: 'Password must contain at least one number',
} as const;

/**
 * String length limits
 */
export const MIN_NAME_LENGTH = 1;
export const MAX_NAME_LENGTH = 255;

/**
 * Numeric limits
 */
export const MIN_CALORIES = 0;
export const MAX_CALORIES = 10000;
export const MIN_DURATION_MINUTES = 1;
export const MAX_DURATION_MINUTES = 1440; // 24 hours

// ============================================================================
// TIME CONSTANTS
// ============================================================================

/**
 * Time in milliseconds
 */
export const MINUTE_MS = 60 * 1000;
export const HOUR_MS = 60 * MINUTE_MS;
export const DAY_MS = 24 * HOUR_MS;
export const WEEK_MS = 7 * DAY_MS;
export const MONTH_MS = 30 * DAY_MS;

/**
 * Token expiration
 */
export const JWT_EXPIRATION = '7d';
export const RESET_TOKEN_EXPIRATION_MS = 1 * HOUR_MS; // 1 hour

// ============================================================================
// HTTP STATUS CODES
// ============================================================================

export const HTTP_STATUS = {
  OK: 200,
  CREATED: 201,
  NO_CONTENT: 204,
  BAD_REQUEST: 400,
  UNAUTHORIZED: 401,
  FORBIDDEN: 403,
  NOT_FOUND: 404,
  CONFLICT: 409,
  INTERNAL_SERVER_ERROR: 500,
} as const;

// ============================================================================
// ERROR MESSAGES
// ============================================================================

export const ERROR_MESSAGES = {
  // Authentication
  INVALID_CREDENTIALS: 'Invalid email or password',
  UNAUTHORIZED: 'Unauthorized access',
  TOKEN_EXPIRED: 'Token has expired',
  TOKEN_INVALID: 'Invalid token',

  // Validation
  VALIDATION_FAILED: 'Validation failed',
  REQUIRED_FIELD: (field: string) => `${field} is required`,
  INVALID_FORMAT: (field: string) => `Invalid ${field} format`,

  // Resources
  NOT_FOUND: (resource: string) => `${resource} not found`,
  ALREADY_EXISTS: (resource: string) => `${resource} already exists`,

  // Generic
  INTERNAL_ERROR: 'Internal server error',
  BAD_REQUEST: 'Bad request',
} as const;

// ============================================================================
// SUCCESS MESSAGES
// ============================================================================

export const SUCCESS_MESSAGES = {
  CREATED: (resource: string) => `${resource} created successfully`,
  UPDATED: (resource: string) => `${resource} updated successfully`,
  DELETED: (resource: string) => `${resource} deleted successfully`,
  OPERATION_SUCCESS: 'Operation completed successfully',
} as const;

// ============================================================================
// RESOURCE NAMES
// ============================================================================

export const RESOURCES = {
  USER: 'User',
  MEAL: 'Meal',
  ACTIVITY: 'Activity',
  HEALTH_METRIC: 'Health Metric',
  WATER_INTAKE: 'Water Intake',
  WEIGHT_RECORD: 'Weight Record',
} as const;

// ============================================================================
// DATE FORMATS
// ============================================================================

export const DATE_FORMATS = {
  ISO_DATE: 'YYYY-MM-DD',
  ISO_DATETIME: 'YYYY-MM-DDTHH:mm:ss.SSSZ',
  DISPLAY_DATE: 'MMM DD, YYYY',
  DISPLAY_DATETIME: 'MMM DD, YYYY HH:mm',
} as const;

// ============================================================================
// DATABASE SELECT FIELDS
// ============================================================================

/**
 * Standard user fields to select (excludes password and sensitive data)
 * Use this constant to ensure consistent user data shape across the app
 */
export const USER_SELECT_FIELDS = {
  id: true,
  email: true,
  name: true,
  profilePicture: true,
  goalCalories: true,
  goalProtein: true,
  goalCarbs: true,
  goalFat: true,
  currentWeight: true,
  goalWeight: true,
  height: true,
  activityLevel: true,
  createdAt: true,
} as const;

/**
 * User fields for goal-only queries (lighter payload)
 */
export const USER_GOALS_SELECT_FIELDS = {
  goalCalories: true,
  goalProtein: true,
  goalCarbs: true,
  goalFat: true,
} as const;

/**
 * User fields for profile display (excludes createdAt)
 */
export const USER_PROFILE_SELECT_FIELDS = {
  id: true,
  email: true,
  name: true,
  profilePicture: true,
  goalCalories: true,
  goalProtein: true,
  goalCarbs: true,
  goalFat: true,
  currentWeight: true,
  goalWeight: true,
  height: true,
  activityLevel: true,
} as const;

// ============================================================================
// CGM (CONTINUOUS GLUCOSE MONITOR) CONSTANTS
// ============================================================================

/**
 * CGM OAuth configuration
 * Production URLs for Dexcom and LibreView APIs
 */
export const CGM_OAUTH_CONFIG = {
  DEXCOM: {
    // Sandbox URLs (use for development)
    SANDBOX_AUTH_URL: 'https://sandbox-api.dexcom.com/v2/oauth2/login',
    SANDBOX_TOKEN_URL: 'https://sandbox-api.dexcom.com/v2/oauth2/token',
    SANDBOX_API_BASE: 'https://sandbox-api.dexcom.com/v3',
    // Production URLs
    AUTH_URL: 'https://api.dexcom.com/v2/oauth2/login',
    TOKEN_URL: 'https://api.dexcom.com/v2/oauth2/token',
    API_BASE: 'https://api.dexcom.com/v3',
    // Scopes
    SCOPES: ['offline_access', 'egv.read'],
    // Rate limits
    RATE_LIMIT_PER_HOUR: 60,
  },
  LIBRE: {
    // LibreView API (requires partnership)
    AUTH_URL: 'https://api.libreview.io/oauth2/authorize',
    TOKEN_URL: 'https://api.libreview.io/oauth2/token',
    API_BASE: 'https://api.libreview.io',
    SCOPES: ['read_data'],
    RATE_LIMIT_PER_HOUR: 60,
  },
  LEVELS: {
    // Levels Health API
    API_BASE: 'https://api.levels.link/v1',
    // Uses API key instead of OAuth
    RATE_LIMIT_PER_HOUR: 100,
  },
} as const;

/**
 * CGM data constants
 */
export const CGM_DATA = {
  // Typical CGM reading interval (5 minutes)
  READING_INTERVAL_MINUTES: 5,
  // Default glucose readings per day (288 = 24 hours * 12 readings/hour)
  READINGS_PER_DAY: 288,
  // Maximum readings to fetch in one sync
  MAX_SYNC_READINGS: 2000,
  // Default sync lookback in hours
  DEFAULT_SYNC_HOURS: 24,
} as const;

/**
 * Glucose range thresholds (mg/dL)
 * Based on standard clinical definitions
 */
export const GLUCOSE_RANGES = {
  // Hypoglycemia (low blood sugar)
  HYPOGLYCEMIA_SEVERE: 54,
  HYPOGLYCEMIA: 70,
  // Target range
  TARGET_LOW: 70,
  TARGET_HIGH: 180,
  // Hyperglycemia (high blood sugar)
  HYPERGLYCEMIA: 180,
  HYPERGLYCEMIA_SEVERE: 250,
  // Danger zones
  CRITICAL_LOW: 40,
  CRITICAL_HIGH: 400,
} as const;

/**
 * Glucose score calculation weights
 * Used for meal glucose response scoring
 */
export const GLUCOSE_SCORE_WEIGHTS = {
  PEAK_GLUCOSE: 0.35, // Lower peak is better
  TIME_TO_PEAK: 0.15, // Longer time to peak is better
  RETURN_TO_BASELINE: 0.25, // Faster return is better
  AREA_UNDER_CURVE: 0.25, // Smaller AUC is better
} as const;

/**
 * Meal glucose analysis window settings (in minutes)
 */
export const MEAL_GLUCOSE_ANALYSIS = {
  PRE_MEAL_BASELINE_MINUTES: 30,
  POST_MEAL_WINDOW_MINUTES: 180, // 3 hours
  MIN_READINGS_FOR_ANALYSIS: 12, // At least 1 hour of data
  RETURN_TO_BASELINE_THRESHOLD_PERCENT: 10, // Within 10% of baseline
} as const;

/**
 * Token refresh settings
 */
export const CGM_TOKEN_REFRESH = {
  // Refresh token before it expires (in milliseconds)
  REFRESH_BUFFER_MS: 5 * MINUTE_MS,
  // Maximum retry attempts for token refresh
  MAX_REFRESH_RETRIES: 3,
  // Delay between retries (exponential backoff base)
  RETRY_DELAY_MS: 1000,
} as const;

/**
 * CGM sync settings
 */
export const CGM_SYNC_SETTINGS = {
  // Minimum time between syncs (in milliseconds)
  MIN_SYNC_INTERVAL_MS: 5 * MINUTE_MS,
  // Maximum historical data to fetch on initial sync (in days)
  INITIAL_SYNC_DAYS: 30,
  // Batch size for database inserts
  BATCH_INSERT_SIZE: 100,
} as const;

/**
 * CGM-related error messages
 */
export const CGM_ERROR_MESSAGES = {
  CONNECTION_NOT_FOUND: 'CGM connection not found',
  CONNECTION_INACTIVE: 'CGM connection is not active',
  PROVIDER_NOT_SUPPORTED: 'CGM provider is not supported',
  OAUTH_FAILED: 'OAuth authentication failed',
  TOKEN_EXPIRED: 'CGM access token has expired',
  TOKEN_REFRESH_FAILED: 'Failed to refresh CGM access token',
  SYNC_FAILED: 'Failed to sync CGM data',
  RATE_LIMITED: 'CGM API rate limit exceeded',
  INVALID_GLUCOSE_VALUE: 'Invalid glucose value',
  ANALYSIS_INSUFFICIENT_DATA: 'Insufficient glucose data for meal analysis',
} as const;

/**
 * CGM success messages
 */
export const CGM_SUCCESS_MESSAGES = {
  CONNECTED: (provider: string) => `Successfully connected to ${provider}`,
  DISCONNECTED: (provider: string) => `Successfully disconnected from ${provider}`,
  SYNCED: (count: number) => `Successfully synced ${count} glucose readings`,
  ANALYSIS_COMPLETE: 'Meal glucose analysis completed',
} as const;
