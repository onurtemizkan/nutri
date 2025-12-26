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
