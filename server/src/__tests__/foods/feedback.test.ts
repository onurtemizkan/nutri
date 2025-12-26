/**
 * Food Feedback Service Tests
 *
 * Tests for:
 * - Feedback submission and storage
 * - Aggregation pattern detection
 * - Feedback statistics
 * - ML service sync
 * - Admin workflow (approval/rejection)
 */

// Mock logger before imports
jest.mock('../../config/logger', () => ({
  logger: {
    warn: jest.fn(),
    debug: jest.fn(),
    error: jest.fn(),
    info: jest.fn(),
  },
  createChildLogger: jest.fn(() => ({
    warn: jest.fn(),
    debug: jest.fn(),
    error: jest.fn(),
    info: jest.fn(),
  })),
}));

// Mock Redis
const mockRedisGet = jest.fn();
const mockRedisSetex = jest.fn();
const mockRedisDel = jest.fn();
let mockRedisAvailable = true;

jest.mock('../../config/redis', () => ({
  getRedisClient: jest.fn(() =>
    mockRedisAvailable
      ? { get: mockRedisGet, setex: mockRedisSetex, del: mockRedisDel }
      : null
  ),
  isRedisAvailable: jest.fn(() => mockRedisAvailable),
}));

// Mock Prisma
const mockFoodFeedbackCreate = jest.fn();
const mockFoodFeedbackFindMany = jest.fn();
const mockFoodFeedbackCount = jest.fn();
const mockFoodFeedbackUpdate = jest.fn();
const mockFoodFeedbackGroupBy = jest.fn();
const mockAggregationFindUnique = jest.fn();
const mockAggregationFindMany = jest.fn();
const mockAggregationCreate = jest.fn();
const mockAggregationUpdate = jest.fn();
const mockAggregationCount = jest.fn();
const mockTransaction = jest.fn();

jest.mock('../../config/database', () => ({
  __esModule: true,
  default: {
    foodFeedback: {
      create: (...args: unknown[]) => mockFoodFeedbackCreate(...args),
      findMany: (...args: unknown[]) => mockFoodFeedbackFindMany(...args),
      count: (...args: unknown[]) => mockFoodFeedbackCount(...args),
      update: (...args: unknown[]) => mockFoodFeedbackUpdate(...args),
      groupBy: (...args: unknown[]) => mockFoodFeedbackGroupBy(...args),
    },
    foodFeedbackAggregation: {
      findUnique: (...args: unknown[]) => mockAggregationFindUnique(...args),
      findMany: (...args: unknown[]) => mockAggregationFindMany(...args),
      create: (...args: unknown[]) => mockAggregationCreate(...args),
      update: (...args: unknown[]) => mockAggregationUpdate(...args),
      count: (...args: unknown[]) => mockAggregationCount(...args),
    },
    $transaction: (fn: (tx: unknown) => Promise<unknown>) =>
      mockTransaction(fn),
  },
}));

// Mock fetch for ML service sync
global.fetch = jest.fn();

import { foodFeedbackService, type FeedbackSubmission } from '../../services/foodFeedbackService';

// ============================================================================
// TEST DATA
// ============================================================================

const mockFeedbackSubmission: FeedbackSubmission = {
  userId: 'user123',
  imageHash: 'abc123hash',
  classificationId: 'clf123',
  originalPrediction: 'Apple',
  originalConfidence: 0.85,
  originalCategory: 'fruits_fresh',
  selectedFdcId: 171688,
  selectedFoodName: 'Apples, raw, with skin',
  wasCorrect: false,
  classificationHints: { texture: 'smooth', color: 'red' },
  userDescription: 'This is a red apple',
};

const mockFeedbackRecord = {
  id: 'feedback123',
  userId: 'user123',
  imageHash: 'abc123hash',
  classificationId: 'clf123',
  originalPrediction: 'apple',
  originalConfidence: 0.85,
  originalCategory: 'fruits_fresh',
  selectedFdcId: 171688,
  selectedFoodName: 'apples,_raw,_with_skin',
  wasCorrect: false,
  status: 'pending',
  createdAt: new Date(),
};

const mockAggregation = {
  id: 'agg123',
  originalPrediction: 'apple',
  correctedFood: 'apples,_raw,_with_skin',
  correctionCount: 5,
  avgConfidence: 0.82,
  needsReview: false,
  firstOccurrence: new Date('2024-01-01'),
  lastOccurrence: new Date('2024-01-15'),
  reviewedAt: null,
};

// ============================================================================
// FEEDBACK SUBMISSION TESTS
// ============================================================================

describe('FoodFeedbackService - Submission', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockRedisAvailable = true;
    mockRedisGet.mockResolvedValue(null);
    mockRedisDel.mockResolvedValue(1);
  });

  describe('submitFeedback', () => {
    it('should submit feedback successfully for incorrect classification', async () => {
      // Setup transaction mock
      mockTransaction.mockImplementation(async (fn: (tx: unknown) => Promise<unknown>) => {
        const txMock = {
          foodFeedback: {
            create: jest.fn().mockResolvedValue(mockFeedbackRecord),
          },
          foodFeedbackAggregation: {
            findUnique: jest.fn().mockResolvedValue(null),
            create: jest.fn().mockResolvedValue(mockAggregation),
          },
        };
        return fn(txMock);
      });

      const result = await foodFeedbackService.submitFeedback(mockFeedbackSubmission);

      expect(result.feedbackId).toBe('feedback123');
      expect(result.isDuplicate).toBe(false);
    });

    it('should normalize prediction and food names', async () => {
      let capturedData: Record<string, unknown> | null = null;

      mockTransaction.mockImplementation(async (fn: (tx: unknown) => Promise<unknown>) => {
        const txMock = {
          foodFeedback: {
            create: jest.fn().mockImplementation(async (args: { data: Record<string, unknown> }) => {
              capturedData = args.data;
              return mockFeedbackRecord;
            }),
          },
          foodFeedbackAggregation: {
            findUnique: jest.fn().mockResolvedValue(null),
            create: jest.fn().mockResolvedValue(mockAggregation),
          },
        };
        return fn(txMock);
      });

      const submission: FeedbackSubmission = {
        ...mockFeedbackSubmission,
        originalPrediction: 'Apple Pie  ',
        selectedFoodName: 'Apple Pie With Cream',
      };

      await foodFeedbackService.submitFeedback(submission);

      expect(capturedData).not.toBeNull();
      const data = capturedData as unknown as Record<string, unknown>;
      expect(data.originalPrediction).toBe('apple_pie');
      expect(data.selectedFoodName).toBe('apple_pie_with_cream');
    });

    it('should update existing aggregation when correction pattern exists', async () => {
      const existingAggregation = {
        ...mockAggregation,
        correctionCount: 5,
        avgConfidence: 0.80,
      };

      const updateMock = jest.fn().mockResolvedValue({
        ...existingAggregation,
        correctionCount: 6,
      });

      mockTransaction.mockImplementation(async (fn: (tx: unknown) => Promise<unknown>) => {
        const txMock = {
          foodFeedback: {
            create: jest.fn().mockResolvedValue(mockFeedbackRecord),
          },
          foodFeedbackAggregation: {
            findUnique: jest.fn().mockResolvedValue(existingAggregation),
            update: updateMock,
          },
        };
        return fn(txMock);
      });

      await foodFeedbackService.submitFeedback(mockFeedbackSubmission);

      expect(updateMock).toHaveBeenCalled();
    });

    it('should flag pattern when correction threshold is reached', async () => {
      const existingAggregation = {
        ...mockAggregation,
        correctionCount: 9, // At threshold - 1
        needsReview: false,
      };

      mockTransaction.mockImplementation(async (fn: (tx: unknown) => Promise<unknown>) => {
        const txMock = {
          foodFeedback: {
            create: jest.fn().mockResolvedValue(mockFeedbackRecord),
          },
          foodFeedbackAggregation: {
            findUnique: jest.fn().mockResolvedValue(existingAggregation),
            update: jest.fn().mockResolvedValue({
              ...existingAggregation,
              correctionCount: 10,
              needsReview: true,
            }),
          },
        };
        return fn(txMock);
      });

      const result = await foodFeedbackService.submitFeedback(mockFeedbackSubmission);

      expect(result.patternFlagged).toBe(true);
    });

    it('should not update aggregation for correct classifications', async () => {
      const correctSubmission: FeedbackSubmission = {
        ...mockFeedbackSubmission,
        wasCorrect: true,
      };

      const aggregationCreateMock = jest.fn();

      mockTransaction.mockImplementation(async (fn: (tx: unknown) => Promise<unknown>) => {
        const txMock = {
          foodFeedback: {
            create: jest.fn().mockResolvedValue({ ...mockFeedbackRecord, wasCorrect: true }),
          },
          foodFeedbackAggregation: {
            findUnique: jest.fn(),
            create: aggregationCreateMock,
          },
        };
        return fn(txMock);
      });

      await foodFeedbackService.submitFeedback(correctSubmission);

      expect(aggregationCreateMock).not.toHaveBeenCalled();
    });

    it('should handle duplicate feedback gracefully', async () => {
      // Import Prisma to get the actual error class
      const { Prisma } = require('@prisma/client');

      // Create a real PrismaClientKnownRequestError
      const duplicateError = new Prisma.PrismaClientKnownRequestError(
        'Unique constraint violation',
        {
          code: 'P2002',
          clientVersion: '5.0.0',
          meta: { target: ['userId', 'imageHash'] },
        }
      );

      mockTransaction.mockRejectedValue(duplicateError);

      const result = await foodFeedbackService.submitFeedback(mockFeedbackSubmission);

      expect(result.isDuplicate).toBe(true);
      expect(result.feedbackId).toBe('');
    });

    it('should invalidate stats cache after submission', async () => {
      mockTransaction.mockImplementation(async (fn: (tx: unknown) => Promise<unknown>) => {
        const txMock = {
          foodFeedback: {
            create: jest.fn().mockResolvedValue(mockFeedbackRecord),
          },
          foodFeedbackAggregation: {
            findUnique: jest.fn().mockResolvedValue(null),
            create: jest.fn().mockResolvedValue(mockAggregation),
          },
        };
        return fn(txMock);
      });

      await foodFeedbackService.submitFeedback(mockFeedbackSubmission);

      expect(mockRedisDel).toHaveBeenCalledWith('food:feedback:stats');
    });
  });
});

// ============================================================================
// FEEDBACK STATISTICS TESTS
// ============================================================================

describe('FoodFeedbackService - Statistics', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockRedisAvailable = true;
    mockRedisGet.mockResolvedValue(null);
  });

  describe('getStats', () => {
    it('should return cached stats when available', async () => {
      const cachedStats = {
        totalFeedback: 100,
        pendingFeedback: 50,
        approvedFeedback: 40,
        rejectedFeedback: 10,
        topMisclassifications: [],
        problemFoods: [],
        patternsNeedingReview: 5,
      };

      mockRedisGet.mockResolvedValueOnce(JSON.stringify(cachedStats));

      const result = await foodFeedbackService.getStats();

      expect(result).toEqual(cachedStats);
      expect(mockFoodFeedbackGroupBy).not.toHaveBeenCalled();
    });

    it('should compute stats when cache is empty', async () => {
      mockRedisGet.mockResolvedValueOnce(null);
      mockRedisSetex.mockResolvedValueOnce('OK');

      mockFoodFeedbackGroupBy
        .mockResolvedValueOnce([
          { status: 'pending', _count: { id: 50 } },
          { status: 'approved', _count: { id: 40 } },
          { status: 'rejected', _count: { id: 10 } },
        ])
        .mockResolvedValueOnce([
          {
            originalPrediction: 'banana',
            _count: { id: 15 },
            _avg: { originalConfidence: 0.75 },
          },
        ]);

      mockAggregationFindMany.mockResolvedValueOnce([
        {
          originalPrediction: 'apple',
          correctedFood: 'banana',
          correctionCount: 20,
        },
      ]);

      mockAggregationCount.mockResolvedValueOnce(5);

      const result = await foodFeedbackService.getStats();

      expect(result.totalFeedback).toBe(100);
      expect(result.pendingFeedback).toBe(50);
      expect(result.approvedFeedback).toBe(40);
      expect(result.rejectedFeedback).toBe(10);
      expect(result.patternsNeedingReview).toBe(5);
    });

    it('should cache computed stats', async () => {
      mockRedisGet.mockResolvedValueOnce(null);
      mockRedisSetex.mockResolvedValueOnce('OK');

      mockFoodFeedbackGroupBy
        .mockResolvedValueOnce([])
        .mockResolvedValueOnce([]);
      mockAggregationFindMany.mockResolvedValueOnce([]);
      mockAggregationCount.mockResolvedValueOnce(0);

      await foodFeedbackService.getStats();

      expect(mockRedisSetex).toHaveBeenCalledWith(
        'food:feedback:stats',
        3600, // 1 hour TTL
        expect.any(String)
      );
    });
  });
});

// ============================================================================
// PATTERN REVIEW TESTS
// ============================================================================

describe('FoodFeedbackService - Pattern Review', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('getPatternsNeedingReview', () => {
    it('should return patterns that need review', async () => {
      const patterns = [
        {
          ...mockAggregation,
          correctionCount: 15,
          needsReview: true,
        },
        {
          ...mockAggregation,
          id: 'agg456',
          correctionCount: 12,
          needsReview: true,
        },
      ];

      mockAggregationFindMany.mockResolvedValueOnce(patterns);

      const result = await foodFeedbackService.getPatternsNeedingReview();

      expect(result).toHaveLength(2);
      expect(mockAggregationFindMany).toHaveBeenCalledWith(
        expect.objectContaining({
          where: {
            needsReview: true,
            reviewedAt: null,
          },
          orderBy: { correctionCount: 'desc' },
        })
      );
    });

    it('should respect limit parameter', async () => {
      mockAggregationFindMany.mockResolvedValueOnce([mockAggregation]);

      await foodFeedbackService.getPatternsNeedingReview(10);

      expect(mockAggregationFindMany).toHaveBeenCalledWith(
        expect.objectContaining({
          take: 10,
        })
      );
    });
  });

  describe('getMisclassificationsForFood', () => {
    it('should find patterns for specific food', async () => {
      mockAggregationFindMany.mockResolvedValueOnce([mockAggregation]);

      const result = await foodFeedbackService.getMisclassificationsForFood('apple');

      expect(result).toHaveLength(1);
      expect(mockAggregationFindMany).toHaveBeenCalledWith(
        expect.objectContaining({
          where: {
            OR: [
              { originalPrediction: 'apple' },
              { correctedFood: 'apple' },
            ],
          },
        })
      );
    });

    it('should normalize food key before searching', async () => {
      mockAggregationFindMany.mockResolvedValueOnce([]);

      await foodFeedbackService.getMisclassificationsForFood('Apple Pie ');

      expect(mockAggregationFindMany).toHaveBeenCalledWith(
        expect.objectContaining({
          where: {
            OR: [
              { originalPrediction: 'apple_pie' },
              { correctedFood: 'apple_pie' },
            ],
          },
        })
      );
    });
  });

  describe('markPatternReviewed', () => {
    it('should update pattern with review timestamp', async () => {
      mockAggregationUpdate.mockResolvedValueOnce({
        ...mockAggregation,
        reviewedAt: new Date(),
      });

      await foodFeedbackService.markPatternReviewed('agg123');

      expect(mockAggregationUpdate).toHaveBeenCalledWith({
        where: { id: 'agg123' },
        data: {
          reviewedAt: expect.any(Date),
        },
      });
    });
  });
});

// ============================================================================
// ADMIN WORKFLOW TESTS
// ============================================================================

describe('FoodFeedbackService - Admin Workflow', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockRedisAvailable = true;
    mockRedisDel.mockResolvedValue(1);
  });

  describe('updateFeedbackStatus', () => {
    it('should update feedback status', async () => {
      mockFoodFeedbackUpdate.mockResolvedValueOnce({
        ...mockFeedbackRecord,
        status: 'approved',
      });

      await foodFeedbackService.updateFeedbackStatus('feedback123', 'approved');

      expect(mockFoodFeedbackUpdate).toHaveBeenCalledWith({
        where: { id: 'feedback123' },
        data: { status: 'approved' },
      });
    });

    it('should invalidate stats cache after status update', async () => {
      mockFoodFeedbackUpdate.mockResolvedValueOnce(mockFeedbackRecord);

      await foodFeedbackService.updateFeedbackStatus('feedback123', 'rejected');

      expect(mockRedisDel).toHaveBeenCalledWith('food:feedback:stats');
    });
  });

  describe('getFeedbackList', () => {
    it('should return paginated feedback list', async () => {
      const feedbackItems = [mockFeedbackRecord, { ...mockFeedbackRecord, id: 'feedback456' }];

      mockFoodFeedbackFindMany.mockResolvedValueOnce(feedbackItems);
      mockFoodFeedbackCount.mockResolvedValueOnce(50);

      const result = await foodFeedbackService.getFeedbackList({
        page: 1,
        limit: 20,
      });

      expect(result.items).toHaveLength(2);
      expect(result.total).toBe(50);
      expect(result.page).toBe(1);
      expect(result.totalPages).toBe(3);
    });

    it('should filter by status', async () => {
      mockFoodFeedbackFindMany.mockResolvedValueOnce([]);
      mockFoodFeedbackCount.mockResolvedValueOnce(0);

      await foodFeedbackService.getFeedbackList({
        status: 'pending',
      });

      expect(mockFoodFeedbackFindMany).toHaveBeenCalledWith(
        expect.objectContaining({
          where: { status: 'pending' },
        })
      );
    });

    it('should filter by userId', async () => {
      mockFoodFeedbackFindMany.mockResolvedValueOnce([]);
      mockFoodFeedbackCount.mockResolvedValueOnce(0);

      await foodFeedbackService.getFeedbackList({
        userId: 'user123',
      });

      expect(mockFoodFeedbackFindMany).toHaveBeenCalledWith(
        expect.objectContaining({
          where: { userId: 'user123' },
        })
      );
    });

    it('should cap limit at 100', async () => {
      mockFoodFeedbackFindMany.mockResolvedValueOnce([]);
      mockFoodFeedbackCount.mockResolvedValueOnce(0);

      await foodFeedbackService.getFeedbackList({
        limit: 500,
      });

      expect(mockFoodFeedbackFindMany).toHaveBeenCalledWith(
        expect.objectContaining({
          take: 100,
        })
      );
    });
  });
});

// ============================================================================
// ML SERVICE SYNC TESTS
// ============================================================================

describe('FoodFeedbackService - ML Service Sync', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    process.env.ML_SERVICE_URL = 'http://localhost:8000';
    mockRedisDel.mockResolvedValue(1);
  });

  afterEach(() => {
    delete process.env.ML_SERVICE_URL;
  });

  describe('syncFeedbackToMLService', () => {
    it('should sync approved feedback to ML service', async () => {
      const approvedFeedback = [
        { ...mockFeedbackRecord, status: 'approved' },
        { ...mockFeedbackRecord, id: 'feedback456', status: 'approved' },
      ];

      mockFoodFeedbackFindMany.mockResolvedValueOnce(approvedFeedback);
      mockFoodFeedbackUpdate.mockResolvedValue({ status: 'applied' });

      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({ ok: true })
        .mockResolvedValueOnce({ ok: true });

      const result = await foodFeedbackService.syncFeedbackToMLService();

      expect(result.synced).toBe(2);
      expect(result.failed).toBe(0);
      expect(global.fetch).toHaveBeenCalledTimes(2);
    });

    it('should mark feedback as applied after successful sync', async () => {
      mockFoodFeedbackFindMany.mockResolvedValueOnce([mockFeedbackRecord]);
      (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: true });
      mockFoodFeedbackUpdate.mockResolvedValue({ status: 'applied' });

      await foodFeedbackService.syncFeedbackToMLService();

      expect(mockFoodFeedbackUpdate).toHaveBeenCalledWith({
        where: { id: 'feedback123' },
        data: { status: 'applied' },
      });
    });

    it('should handle ML service errors gracefully', async () => {
      mockFoodFeedbackFindMany.mockResolvedValueOnce([mockFeedbackRecord]);
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 500,
        text: () => Promise.resolve('Internal Server Error'),
      });

      const result = await foodFeedbackService.syncFeedbackToMLService();

      expect(result.synced).toBe(0);
      expect(result.failed).toBe(1);
    });

    it('should handle network errors gracefully', async () => {
      mockFoodFeedbackFindMany.mockResolvedValueOnce([mockFeedbackRecord]);
      (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'));

      const result = await foodFeedbackService.syncFeedbackToMLService();

      expect(result.synced).toBe(0);
      expect(result.failed).toBe(1);
    });

    it('should send correct payload to ML service', async () => {
      mockFoodFeedbackFindMany.mockResolvedValueOnce([mockFeedbackRecord]);
      (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: true });
      mockFoodFeedbackUpdate.mockResolvedValue({ status: 'applied' });

      await foodFeedbackService.syncFeedbackToMLService();

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/food/feedback',
        expect.objectContaining({
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: expect.stringContaining('image_hash'),
        })
      );

      const callBody = JSON.parse((global.fetch as jest.Mock).mock.calls[0][1].body);
      expect(callBody).toEqual({
        image_hash: mockFeedbackRecord.imageHash,
        original_prediction: mockFeedbackRecord.originalPrediction,
        original_confidence: mockFeedbackRecord.originalConfidence,
        corrected_label: mockFeedbackRecord.selectedFoodName,
        user_description: undefined,
        user_id: mockFeedbackRecord.userId,
      });
    });
  });
});

// ============================================================================
// AGGREGATION PATTERN TESTS
// ============================================================================

describe('FoodFeedbackService - Aggregation Patterns', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockRedisDel.mockResolvedValue(1);
  });

  it('should calculate correct average confidence when updating aggregation', async () => {
    const existingAggregation = {
      ...mockAggregation,
      correctionCount: 4,
      avgConfidence: 0.80, // Previous average from 4 submissions
    };

    let capturedUpdate: Record<string, unknown> | null = null;

    mockTransaction.mockImplementation(async (fn: (tx: unknown) => Promise<unknown>) => {
      const txMock = {
        foodFeedback: {
          create: jest.fn().mockResolvedValue(mockFeedbackRecord),
        },
        foodFeedbackAggregation: {
          findUnique: jest.fn().mockResolvedValue(existingAggregation),
          update: jest.fn().mockImplementation(async (args: { data: Record<string, unknown> }) => {
            capturedUpdate = args.data;
            return { ...existingAggregation, ...args.data };
          }),
        },
      };
      return fn(txMock);
    });

    // Submit feedback with confidence 0.85
    const submission = { ...mockFeedbackSubmission, originalConfidence: 0.85 };
    await foodFeedbackService.submitFeedback(submission);

    // New average should be (0.80 * 4 + 0.85) / 5 = 0.81
    expect(capturedUpdate).not.toBeNull();
    const updateData = capturedUpdate as unknown as Record<string, unknown>;
    expect(updateData.avgConfidence).toBeCloseTo(0.81, 2);
    expect(updateData.correctionCount).toBe(5);
  });

  it('should track first and last occurrence dates', async () => {
    let capturedCreate: Record<string, unknown> | null = null;

    mockTransaction.mockImplementation(async (fn: (tx: unknown) => Promise<unknown>) => {
      const txMock = {
        foodFeedback: {
          create: jest.fn().mockResolvedValue(mockFeedbackRecord),
        },
        foodFeedbackAggregation: {
          findUnique: jest.fn().mockResolvedValue(null),
          create: jest.fn().mockImplementation(async (args: { data: Record<string, unknown> }) => {
            capturedCreate = args.data;
            return mockAggregation;
          }),
        },
      };
      return fn(txMock);
    });

    await foodFeedbackService.submitFeedback(mockFeedbackSubmission);

    expect(capturedCreate).not.toBeNull();
    const createData = capturedCreate as unknown as Record<string, unknown>;
    expect(createData.firstOccurrence).toBeDefined();
    expect(createData.lastOccurrence).toBeDefined();
    expect(createData.firstOccurrence instanceof Date).toBe(true);
    expect(createData.lastOccurrence instanceof Date).toBe(true);
  });
});
