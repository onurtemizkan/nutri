import { Response } from 'express';
import { onboardingService, ONBOARDING_STEPS } from '../services/onboardingService';
import { AuthenticatedRequest } from '../types';
import { requireAuth } from '../utils/authHelpers';
import { withErrorHandling, ErrorHandlers } from '../utils/controllerHelpers';
import { HTTP_STATUS } from '../config/constants';
import {
  startOnboardingSchema,
  onboardingStep1Schema,
  onboardingStep2Schema,
  onboardingStep3Schema,
  onboardingStep4Schema,
  onboardingStep5Schema,
  onboardingStep6Schema,
  completeOnboardingSchema,
} from '../validation/schemas';

export class OnboardingController {
  /**
   * Start or resume onboarding
   * POST /api/onboarding/start
   */
  start = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = startOnboardingSchema.parse(req.body);
    const status = await onboardingService.startOnboarding(userId, validatedData.version);

    res.status(HTTP_STATUS.OK).json(status);
  });

  /**
   * Get current onboarding status
   * GET /api/onboarding/status
   */
  getStatus = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const status = await onboardingService.getStatus(userId);
    if (!status) {
      throw new Error('Onboarding not started');
    }

    res.status(HTTP_STATUS.OK).json(status);
  });

  /**
   * Get all onboarding data
   * GET /api/onboarding/data
   */
  getData = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const data = await onboardingService.getOnboardingData(userId);
    res.status(HTTP_STATUS.OK).json(data);
  });

  /**
   * Save Step 1: Profile Basics
   * PUT /api/onboarding/step/1
   */
  saveStep1 = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = onboardingStep1Schema.parse(req.body);
    const status = await onboardingService.saveStep1(userId, validatedData);

    res.status(HTTP_STATUS.OK).json(status);
  });

  /**
   * Save Step 2: Health Goals
   * PUT /api/onboarding/step/2
   */
  saveStep2 = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = onboardingStep2Schema.parse(req.body);
    const status = await onboardingService.saveStep2(userId, validatedData);

    res.status(HTTP_STATUS.OK).json(status);
  });

  /**
   * Save Step 3: Permissions
   * PUT /api/onboarding/step/3
   */
  saveStep3 = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = onboardingStep3Schema.parse(req.body);
    const status = await onboardingService.saveStep3(userId, validatedData);

    res.status(HTTP_STATUS.OK).json(status);
  });

  /**
   * Save Step 4: Health Background
   * PUT /api/onboarding/step/4
   */
  saveStep4 = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = onboardingStep4Schema.parse(req.body);
    const status = await onboardingService.saveStep4(userId, validatedData);

    res.status(HTTP_STATUS.OK).json(status);
  });

  /**
   * Save Step 5: Lifestyle
   * PUT /api/onboarding/step/5
   */
  saveStep5 = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = onboardingStep5Schema.parse(req.body);
    const status = await onboardingService.saveStep5(userId, validatedData);

    res.status(HTTP_STATUS.OK).json(status);
  });

  /**
   * Save Step 6: Completion (acknowledgment)
   * PUT /api/onboarding/step/6
   */
  saveStep6 = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    // Validate that user acknowledged
    onboardingStep6Schema.parse(req.body);
    const status = await onboardingService.completeOnboarding(userId);

    res.status(HTTP_STATUS.OK).json(status);
  });

  /**
   * Generic step save handler (uses step number from URL)
   * PUT /api/onboarding/step/:stepNumber
   */
  saveStep = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const stepNumber = parseInt(req.params.stepNumber, 10);

    if (isNaN(stepNumber) || stepNumber < 1 || stepNumber > 6) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({ error: 'Invalid step number' });
      return;
    }

    // Route to appropriate handler based on step number
    const stepHandlers: Record<number, (req: AuthenticatedRequest, res: Response) => Promise<void>> = {
      [ONBOARDING_STEPS.PROFILE_BASICS]: this.saveStep1,
      [ONBOARDING_STEPS.HEALTH_GOALS]: this.saveStep2,
      [ONBOARDING_STEPS.PERMISSIONS]: this.saveStep3,
      [ONBOARDING_STEPS.HEALTH_BACKGROUND]: this.saveStep4,
      [ONBOARDING_STEPS.LIFESTYLE]: this.saveStep5,
      [ONBOARDING_STEPS.COMPLETION]: this.saveStep6,
    };

    await stepHandlers[stepNumber](req, res);
  });

  /**
   * Skip a step (only allowed for certain steps)
   * POST /api/onboarding/skip/:stepNumber
   */
  skipStep = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const stepNumber = parseInt(req.params.stepNumber, 10);

    if (isNaN(stepNumber)) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({ error: 'Invalid step number' });
      return;
    }

    const status = await onboardingService.skipStep(userId, stepNumber);
    res.status(HTTP_STATUS.OK).json(status);
  });

  /**
   * Complete onboarding (optionally skipping remaining steps)
   * POST /api/onboarding/complete
   */
  complete = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = completeOnboardingSchema.parse(req.body);
    const status = await onboardingService.completeOnboarding(userId, validatedData.skipRemaining);

    res.status(HTTP_STATUS.OK).json(status);
  });
}

export const onboardingController = new OnboardingController();
