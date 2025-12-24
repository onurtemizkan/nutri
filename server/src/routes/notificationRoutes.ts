import { Router } from 'express';
import { notificationController } from '../controllers/notificationController';
import { authenticate } from '../middleware/auth';

const router = Router();

// All notification routes are protected
router.use(authenticate);

// Device registration
router.post('/register-device', (req, res) => notificationController.registerDevice(req, res));
router.delete('/unregister-device', (req, res) => notificationController.unregisterDevice(req, res));
router.get('/devices', (req, res) => notificationController.getDevices(req, res));

// Preferences
router.get('/preferences', (req, res) => notificationController.getPreferences(req, res));
router.put('/preferences', (req, res) => notificationController.updatePreferences(req, res));

// History
router.get('/history', (req, res) => notificationController.getHistory(req, res));

// Tracking
router.post('/track', (req, res) => notificationController.trackNotification(req, res));

// Test (development only)
router.post('/test', (req, res) => notificationController.testNotification(req, res));

export default router;
