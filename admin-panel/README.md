# Nutri Admin Panel

A secure Next.js 14 admin dashboard for managing Nutri application users and subscriptions.

## Features

- **User Management**: View, search, and manage users
- **Subscription Management**: Grant, extend, and revoke subscriptions
- **Analytics Dashboard**: MRR, churn rates, trial conversions
- **Webhook Monitoring**: View Apple App Store webhook events
- **Audit Logging**: All admin actions are logged for compliance
- **MFA Required**: TOTP-based two-factor authentication
- **Role-Based Access Control**: SUPER_ADMIN, SUPPORT, ANALYST, VIEWER roles

## Prerequisites

- Node.js 18+
- npm 9+
- Backend API running (see ../server)
- PostgreSQL database

## Environment Variables

Create a `.env.local` file with the following variables:

```env
# NextAuth Configuration
NEXTAUTH_SECRET=your-secret-key-minimum-32-characters-long
NEXTAUTH_URL=http://localhost:3001

# Backend API URL
API_URL=http://localhost:3000
```

### Production Environment Variables (Vercel)

Configure these in your Vercel dashboard:

| Variable | Description | Example |
|----------|-------------|---------|
| `NEXTAUTH_SECRET` | Secret for JWT encryption (min 32 chars) | `your-production-secret` |
| `NEXTAUTH_URL` | Production URL of admin panel | `https://admin.nutri.app` |
| `API_URL` | Production backend API URL | `https://api.nutri.app` |

## Development

```bash
# Install dependencies
npm install

# Start development server (runs on port 3001)
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Run tests
npm test

# Run tests with coverage
npm run test:coverage
```

## Deployment to Vercel

### Automatic Deployment

1. Connect your repository to Vercel
2. Vercel will automatically detect Next.js and configure the build
3. Set environment variables in Vercel dashboard
4. Deploy

### Manual Deployment

```bash
# Install Vercel CLI
npm i -g vercel

# Login to Vercel
vercel login

# Deploy to preview
vercel

# Deploy to production
vercel --prod
```

### Post-Deployment Steps

1. **Create Initial Admin User**
   ```bash
   # In the server directory
   cd ../server
   npm run db:seed
   ```

   This creates the first SUPER_ADMIN account:
   - Email: `admin@nutri.app`
   - Password: `AdminPass123!`

2. **Login and Set Up MFA**
   - Navigate to your admin panel URL
   - Login with the initial credentials
   - You will be prompted to set up MFA
   - Scan the QR code with an authenticator app (Google Authenticator, Authy, etc.)
   - Enter the 6-digit code to complete setup

3. **Create Additional Admin Users**
   - After logging in as SUPER_ADMIN, you can create additional admin users via the database
   - Use the seed script or direct database access

## Security Features

### Authentication
- JWT-based authentication with 8-hour session timeout
- TOTP-based MFA (required for all users)
- Secure, httpOnly cookies with sameSite=strict
- Rate limiting on login endpoints (5 requests per 15 minutes)

### Authorization
- Role-based access control (RBAC)
- SUPER_ADMIN: Full access
- SUPPORT: User management access
- ANALYST: Analytics read-only access
- VIEWER: Read-only access

### Security Headers
- Content Security Policy (CSP)
- X-Frame-Options: DENY
- X-Content-Type-Options: nosniff
- Referrer-Policy: strict-origin-when-cross-origin
- Permissions-Policy (restricts browser features)
- Strict-Transport-Security (production only)

### IP Allowlisting (Optional)
Configure `ADMIN_IP_ALLOWLIST` environment variable on the backend to restrict access to specific IPs:
```env
ADMIN_IP_ALLOWLIST=192.168.1.1,10.0.0.1,::1,127.0.0.1
```

## Testing

```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage report
npm run test:coverage
```

### Test Structure
- `__tests__/components/` - Component unit tests
- `__tests__/hooks/` - Custom hook tests
- `__tests__/pages/` - Page integration tests

## Project Structure

```
admin-panel/
├── app/                    # Next.js App Router pages
│   ├── (auth)/            # Auth routes (login, MFA)
│   ├── (dashboard)/       # Protected dashboard routes
│   │   ├── users/
│   │   ├── subscriptions/
│   │   ├── analytics/
│   │   └── webhooks/
│   ├── api/auth/          # NextAuth API routes
│   └── layout.tsx
├── components/            # Reusable UI components
│   ├── ui/               # Base UI components (shadcn/ui)
│   ├── users/            # User management components
│   ├── subscriptions/    # Subscription components
│   └── analytics/        # Analytics components
├── lib/                  # Utilities and configuration
│   ├── auth.ts           # NextAuth configuration
│   ├── api.ts            # API client
│   └── types.ts          # TypeScript types
├── public/               # Static assets
├── vercel.json           # Vercel deployment config
└── next.config.js        # Next.js configuration
```

## API Routes

The admin panel proxies requests to the backend API:

- `/api/backend/*` -> `${API_URL}/api/*`

This allows the frontend to call backend APIs without CORS issues in development.

## Troubleshooting

### Login Issues
1. Ensure backend API is running
2. Check `API_URL` environment variable
3. Verify admin user exists in database

### MFA Issues
1. Ensure system time is synchronized
2. Try using a different authenticator app
3. Contact SUPER_ADMIN to reset MFA

### Build Errors
1. Clear `.next` directory: `rm -rf .next`
2. Clear node_modules: `rm -rf node_modules && npm install`
3. Check Node.js version (18+ required)

## License

Proprietary - Nutri App
