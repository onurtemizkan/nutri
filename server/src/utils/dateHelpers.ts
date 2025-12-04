/**
 * Date utility functions for consistent date handling across the application
 * Eliminates code duplication for common date operations
 */

/**
 * Day boundary result containing start and end of day timestamps
 */
export interface DayBoundaries {
  startOfDay: Date;
  endOfDay: Date;
}

/**
 * Get the start and end of day for a given date
 * Useful for querying all records within a specific day
 *
 * @param date - The date to get boundaries for (defaults to today)
 * @returns Object with startOfDay and endOfDay dates
 *
 * @example
 * const { startOfDay, endOfDay } = getDayBoundaries(new Date('2024-01-15'));
 * // startOfDay: 2024-01-15T00:00:00.000Z
 * // endOfDay: 2024-01-15T23:59:59.999Z
 */
export function getDayBoundaries(date: Date = new Date()): DayBoundaries {
  const startOfDay = new Date(date);
  startOfDay.setHours(0, 0, 0, 0);

  const endOfDay = new Date(date);
  endOfDay.setHours(23, 59, 59, 999);

  return { startOfDay, endOfDay };
}

/**
 * Get a date that is a specified number of days in the past
 * Optionally sets the time to start of day
 *
 * @param days - Number of days to go back
 * @param setToStartOfDay - Whether to set time to 00:00:00.000 (default: true)
 * @returns Date object for the past date
 *
 * @example
 * const sevenDaysAgo = getDaysAgo(7);
 * const thirtyDaysAgo = getDaysAgo(30);
 */
export function getDaysAgo(days: number, setToStartOfDay: boolean = true): Date {
  const date = new Date();
  date.setDate(date.getDate() - days);

  if (setToStartOfDay) {
    date.setHours(0, 0, 0, 0);
  }

  return date;
}

/**
 * Get start of day for a given date
 * Sets time to 00:00:00.000
 *
 * @param date - The date to get start of day for (defaults to today)
 * @returns Date object set to start of day
 */
export function getStartOfDay(date: Date = new Date()): Date {
  const startOfDay = new Date(date);
  startOfDay.setHours(0, 0, 0, 0);
  return startOfDay;
}

/**
 * Get end of day for a given date
 * Sets time to 23:59:59.999
 *
 * @param date - The date to get end of day for (defaults to today)
 * @returns Date object set to end of day
 */
export function getEndOfDay(date: Date = new Date()): Date {
  const endOfDay = new Date(date);
  endOfDay.setHours(23, 59, 59, 999);
  return endOfDay;
}

/**
 * Parse an optional date string from query parameters
 * Returns undefined if the string is undefined or empty
 *
 * @param dateString - The date string to parse
 * @returns Parsed Date or undefined
 */
export function parseOptionalDate(dateString: string | undefined): Date | undefined {
  return dateString ? new Date(dateString) : undefined;
}

/**
 * Get the date range for the past N days, including today
 *
 * @param days - Number of days to include (e.g., 7 for a week)
 * @returns Object with startDate and endDate
 */
export function getDateRangeForPastDays(days: number): { startDate: Date; endDate: Date } {
  return {
    startDate: getDaysAgo(days),
    endDate: getEndOfDay(),
  };
}
