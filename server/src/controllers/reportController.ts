import { reportService } from '../services/reportService';
import { AuthenticatedRequest } from '../types';
import { requireAuth } from '../utils/authHelpers';
import {
  weeklyReportQuerySchema,
  monthlyReportQuerySchema,
  weeklyReportExportQuerySchema,
  monthlyReportExportQuerySchema,
} from '../validation/schemas';
import { withErrorHandling } from '../utils/controllerHelpers';
import { HTTP_STATUS } from '../config/constants';

export class ReportController {
  /**
   * GET /api/reports/weekly
   * Generate a weekly nutrition report
   * Query params: date (optional, YYYY-MM-DD)
   */
  getWeeklyReport = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    // Validate query parameters
    const { date } = weeklyReportQuerySchema.parse(req.query);

    // Parse date if provided
    const weekDate = date ? new Date(date) : undefined;

    const report = await reportService.generateWeeklyReport(userId, weekDate);

    res.status(HTTP_STATUS.OK).json(report);
  });

  /**
   * GET /api/reports/monthly
   * Generate a monthly nutrition report
   * Query params: month (optional, YYYY-MM)
   */
  getMonthlyReport = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    // Validate query parameters
    const { month } = monthlyReportQuerySchema.parse(req.query);

    const report = await reportService.generateMonthlyReport(userId, month);

    res.status(HTTP_STATUS.OK).json(report);
  });

  /**
   * GET /api/reports/weekly/export
   * Export weekly report as PDF, image, or JSON
   * Query params: date (optional), format (required: pdf|image|json)
   */
  exportWeeklyReport = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    // Validate query parameters
    const { date, format } = weeklyReportExportQuerySchema.parse(req.query);

    // Parse date if provided
    const weekDate = date ? new Date(date) : undefined;

    // Generate the report
    const report = await reportService.generateWeeklyReport(userId, weekDate);

    // Handle different export formats
    if (format === 'json') {
      // Return raw JSON report
      res.status(HTTP_STATUS.OK).json({
        success: true,
        format: 'json',
        data: report,
        filename: `weekly-report-${report.periodStart.split('T')[0]}.json`,
        mimeType: 'application/json',
      });
    } else {
      // PDF and image export will be implemented in reportExportService (subtask 46.9)
      // For now, return the JSON with a message
      res.status(HTTP_STATUS.OK).json({
        success: true,
        format,
        message: `${format.toUpperCase()} export will be available soon. Returning JSON for now.`,
        data: report,
        filename: `weekly-report-${report.periodStart.split('T')[0]}.${format === 'pdf' ? 'pdf' : 'png'}`,
        mimeType: format === 'pdf' ? 'application/pdf' : 'image/png',
      });
    }
  });

  /**
   * GET /api/reports/monthly/export
   * Export monthly report as PDF, image, or JSON
   * Query params: month (optional), format (required: pdf|image|json)
   */
  exportMonthlyReport = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    // Validate query parameters
    const { month, format } = monthlyReportExportQuerySchema.parse(req.query);

    // Generate the report
    const report = await reportService.generateMonthlyReport(userId, month);

    // Handle different export formats
    if (format === 'json') {
      // Return raw JSON report
      res.status(HTTP_STATUS.OK).json({
        success: true,
        format: 'json',
        data: report,
        filename: `monthly-report-${report.month}.json`,
        mimeType: 'application/json',
      });
    } else {
      // PDF and image export will be implemented in reportExportService (subtask 46.9)
      // For now, return the JSON with a message
      res.status(HTTP_STATUS.OK).json({
        success: true,
        format,
        message: `${format.toUpperCase()} export will be available soon. Returning JSON for now.`,
        data: report,
        filename: `monthly-report-${report.month}.${format === 'pdf' ? 'pdf' : 'png'}`,
        mimeType: format === 'pdf' ? 'application/pdf' : 'image/png',
      });
    }
  });
}

export const reportController = new ReportController();
