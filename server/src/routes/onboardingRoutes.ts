import { Router } from 'express';
import { onboardingController } from '../controllers/onboardingController';
import { authenticate } from '../middleware/auth';

const router = Router();

// All onboarding routes require authentication
router.use(authenticate);

// Start or resume onboarding
router.post('/start', (req, res) => onboardingController.start(req, res));

// Get current onboarding status
router.get('/status', (req, res) => onboardingController.getStatus(req, res));

// Get all onboarding data
router.get('/data', (req, res) => onboardingController.getData(req, res));

// Save step data (generic endpoint)
router.put('/step/:stepNumber', (req, res) => onboardingController.saveStep(req, res));

// Skip a step
router.post('/skip/:stepNumber', (req, res) => onboardingController.skipStep(req, res));

// Complete onboarding
router.post('/complete', (req, res) => onboardingController.complete(req, res));

export default router;
