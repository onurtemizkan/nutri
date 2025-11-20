/**
 * Pagination types and utilities
 */

export interface PaginationQuery {
  page?: number;
  limit?: number;
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
    hasNext: boolean;
    hasPrev: boolean;
  };
}

export interface PaginationParams {
  skip: number;
  take: number;
}

/**
 * Calculate pagination parameters from query
 */
export function getPaginationParams(
  query: PaginationQuery,
  defaultLimit: number = 100,
  maxLimit: number = 1000
): PaginationParams {
  const page = Math.max(1, query.page || 1);
  const limit = Math.min(maxLimit, Math.max(1, query.limit || defaultLimit));
  const skip = (page - 1) * limit;

  return { skip, take: limit };
}

/**
 * Create paginated response
 */
export function createPaginatedResponse<T>(
  data: T[],
  total: number,
  page: number,
  limit: number
): PaginatedResponse<T> {
  const totalPages = Math.ceil(total / limit);

  return {
    data,
    pagination: {
      page,
      limit,
      total,
      totalPages,
      hasNext: page < totalPages,
      hasPrev: page > 1,
    },
  };
}
