import { auth } from '@/lib/auth';
import { NextResponse } from 'next/server';

/**
 * Middleware for auth protection and security headers
 * Uses NextAuth for session management
 *
 * Note: Basic auth redirects are handled by NextAuth's authorized callback.
 * This middleware handles additional logic like role-based access and security headers.
 */

// Routes that require SUPER_ADMIN role
export const superAdminRoutes = ['/dashboard/settings/team'];

export default auth((req) => {
  const { pathname } = req.nextUrl;
  const session = req.auth;

  // Check role for SUPER_ADMIN routes (auth is already verified by authorized callback)
  if (pathname.startsWith('/dashboard') && session?.user) {
    if (
      superAdminRoutes.some((route) => pathname.startsWith(route)) &&
      session.user.role !== 'SUPER_ADMIN'
    ) {
      return NextResponse.redirect(
        new URL('/dashboard?error=insufficient_permissions', req.url)
      );
    }
  }

  return addSecurityHeaders(NextResponse.next());
});

/**
 * Add security headers to response
 */
function addSecurityHeaders(response: NextResponse): NextResponse {
  // Prevent clickjacking
  response.headers.set('X-Frame-Options', 'DENY');

  // Prevent MIME type sniffing
  response.headers.set('X-Content-Type-Options', 'nosniff');

  // XSS protection
  response.headers.set('X-XSS-Protection', '1; mode=block');

  // Referrer policy
  response.headers.set('Referrer-Policy', 'strict-origin-when-cross-origin');

  // Permissions policy
  response.headers.set(
    'Permissions-Policy',
    'camera=(), microphone=(), geolocation=()'
  );

  return response;
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public folder
     */
    '/((?!_next/static|_next/image|favicon.ico|public).*)',
  ],
};
