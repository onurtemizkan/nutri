/**
 * Utility functions for formatting data display
 */

/**
 * Format a date to time string using device locale
 * @param date - ISO date string or Date object
 * @returns Formatted time string (e.g., "2:30 PM" or "14:30" based on locale)
 */
export function formatMealTime(date: string | Date): string {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  return dateObj.toLocaleTimeString(undefined, {
    hour: 'numeric',
    minute: '2-digit',
  });
}

/**
 * Format a date to a readable date string using device locale
 * @param date - ISO date string or Date object
 * @returns Formatted date string (e.g., "Dec 11, 2025")
 */
export function formatDate(date: string | Date): string {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  return dateObj.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
}

/**
 * Format a date to both date and time
 * @param date - ISO date string or Date object
 * @returns Formatted datetime string (e.g., "Dec 11, 2025 at 2:30 PM")
 */
export function formatDateTime(date: string | Date): string {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  return dateObj.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}
