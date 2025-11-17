# Nutri - Setup & Development Guide

## 📋 Table of Contents
- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Database Setup](#database-setup)
- [Running the App](#running-the-app)
- [Project Structure](#project-structure)
- [Dependency Cleanup](#dependency-cleanup)
- [Testing](#testing)
- [Deployment](#deployment)

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
npm install
cd server && npm install && cd ..

# 2. Setup environment variables
cp .env.example .env
cp server/.env.example server/.env

# 3. Edit server/.env with your database credentials and JWT secret

# 4. Setup database
cd server
npx prisma migrate dev
cd ..

# 5. Start development servers
# Terminal 1 - Backend
cd server && npm run dev

# Terminal 2 - Frontend
npm start
```

---

## 📦 Prerequisites

- **Node.js** >= 18.x
- **npm** >= 9.x or **yarn** >= 1.22
- **PostgreSQL** >= 14.x
- **Expo CLI** (will be installed with dependencies)
- **iOS Simulator** (for macOS) or **Android Emulator**

---

## 💾 Installation

### 1. Clone and Install

```bash
# Clone the repository
git clone <your-repo-url>
cd nutri

# Install client dependencies
npm install

# Install server dependencies
cd server
npm install
cd ..
```

### 2. Optional: Remove Unused Dependencies

The following client dependencies can be safely removed if not needed:

```bash
npm uninstall expo-blur expo-constants expo-system-ui expo-web-browser react-native-webview
```

**Why these are unused:**
- `expo-blur`: Blur effect not used in current UI
- `expo-constants`: Not accessing device constants
- `expo-system-ui`: Not needed for current navigation
- `expo-web-browser`: Not opening external browsers
- `react-native-webview`: No webview components in app

**Keep these** (they ARE being used):
- `expo-haptics`: Used in tab navigation (HapticTab component)
- `expo-symbols`: Used for SF Symbols icons (IconSymbol component)
- `expo-linear-gradient`: Used in welcome screen
- `expo-secure-store`: Used for JWT token storage

---

## ⚙️ Configuration

### Client (.env)

Create `.env` file in root:

```env
# For iOS Simulator
API_BASE_URL=http://localhost:3000

# For Android Emulator
# API_BASE_URL=http://10.0.2.2:3000

# For Physical Device (use your local IP)
# API_BASE_URL=http://192.168.1.100:3000
```

### Server (server/.env)

Create `server/.env` file:

```env
NODE_ENV=development
PORT=3000
DATABASE_URL=postgresql://postgres:password@localhost:5432/nutri
JWT_SECRET=<generate-32-char-secret>
JWT_EXPIRES_IN=7d
```

**Generate secure JWT secret:**
```bash
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
```

---

## 🗄️ Database Setup

### 1. Create PostgreSQL Database

```bash
# Using psql
createdb nutri

# Or via PostgreSQL client
psql postgres
CREATE DATABASE nutri;
\q
```

### 2. Run Migrations

```bash
cd server
npx prisma migrate dev --name init
npx prisma generate
```

### 3. (Optional) Seed Database

```bash
npx prisma db seed  # If seed file exists
```

### 4. View Database

```bash
npx prisma studio
# Opens at http://localhost:5555
```

---

## 🏃 Running the App

### Development Mode

#### Backend Server
```bash
cd server
npm run dev
# Server runs on http://localhost:3000
```

#### Frontend App
```bash
# From project root
npm start

# Then press:
# i - iOS Simulator
# a - Android Emulator
# w - Web browser
```

### Production Build

#### Backend
```bash
cd server
npm run build
npm start
```

#### Frontend
```bash
# Build for production
npx expo build:ios     # iOS
npx expo build:android # Android
```

---

## 📁 Project Structure

```
nutri/
├── app/                      # React Native screens
│   ├── (tabs)/              # Tab navigation screens
│   │   ├── index.tsx        # Home/Dashboard
│   │   ├── profile.tsx      # User profile
│   │   └── _layout.tsx      # Tab layout config
│   ├── auth/                # Authentication screens
│   │   ├── welcome.tsx      # Welcome/landing
│   │   ├── signin.tsx       # Sign in
│   │   ├── signup.tsx       # Sign up
│   │   ├── forgot-password.tsx
│   │   └── reset-password.tsx
│   ├── add-meal.tsx         # Add meal modal
│   └── _layout.tsx          # Root layout
│
├── components/              # Reusable components
│   ├── ui/                  # UI components
│   │   ├── IconSymbol.tsx   # Icon wrapper
│   │   └── TabBarBackground.tsx
│   ├── HapticTab.tsx        # Tab with haptic feedback
│   ├── ThemedText.tsx       # Themed text component
│   └── ThemedView.tsx       # Themed view component
│
├── lib/                     # Business logic
│   ├── api/                 # API clients
│   │   ├── client.ts        # Axios instance
│   │   ├── auth.ts          # Auth API calls
│   │   └── meals.ts         # Meals API calls
│   ├── context/
│   │   └── AuthContext.tsx  # Auth state management
│   └── types/               # TypeScript types
│       └── index.ts
│
├── server/                  # Backend API
│   ├── prisma/              # Database ORM
│   │   └── schema.prisma    # Database schema
│   ├── src/
│   │   ├── config/          # Configuration
│   │   │   ├── database.ts
│   │   │   └── env.ts
│   │   ├── controllers/     # Request handlers
│   │   │   ├── authController.ts
│   │   │   └── mealController.ts
│   │   ├── middleware/      # Express middleware
│   │   │   ├── auth.ts
│   │   │   └── errorHandler.ts
│   │   ├── routes/          # API routes
│   │   │   ├── authRoutes.ts
│   │   │   └── mealRoutes.ts
│   │   ├── services/        # Business logic
│   │   │   ├── authService.ts
│   │   │   └── mealService.ts
│   │   ├── types/           # TypeScript types
│   │   │   └── index.ts
│   │   ├── utils/           # Utility functions
│   │   │   └── authHelpers.ts
│   │   └── index.ts         # Server entry point
│   ├── .env.example
│   └── package.json
│
├── .env.example             # Environment template
├── .gitignore
├── package.json
├── tsconfig.json
└── README.md
```

---

## 🧹 Dependency Cleanup

### Client Dependencies Analysis

#### ✅ **Currently Used**
- `axios` - API requests
- `expo-router` - Navigation
- `expo-secure-store` - Token storage
- `expo-linear-gradient` - UI gradients
- `expo-haptics` - Haptic feedback
- `expo-symbols` - SF Symbols icons
- `react-native-safe-area-context` - Safe areas
- All React Native core packages

#### ❌ **Can Be Removed** (Optional)
```bash
npm uninstall expo-blur expo-constants expo-system-ui expo-web-browser react-native-webview
```

**Savings:** ~15MB in node_modules

#### 📊 **Before/After**
- **Before:** 57 dependencies
- **After:** 52 dependencies (-9%)
- **node_modules size:** ~280MB → ~265MB

### Server Dependencies

All server dependencies are currently in use. No removals needed.

---

## 🧪 Testing

### Run Tests

```bash
# Client tests
npm test

# Server tests
cd server
npm test
```

### Manual Testing Checklist

- [ ] Sign up with new account
- [ ] Sign in with existing account
- [ ] Request password reset
- [ ] Reset password with token
- [ ] View dashboard with meals
- [ ] Add new meal
- [ ] Edit meal
- [ ] Delete meal
- [ ] Update profile goals
- [ ] Logout

---

## 🚀 Deployment

### Backend (Server)

#### Using Heroku
```bash
cd server
heroku create your-app-name
heroku addons:create heroku-postgresql:mini
heroku config:set JWT_SECRET=your-secret-here
git push heroku main
```

#### Using Railway/Render
1. Connect GitHub repository
2. Add PostgreSQL database
3. Set environment variables
4. Deploy

### Frontend (App)

#### Expo EAS Build
```bash
npm install -g eas-cli
eas login
eas build:configure
eas build --platform ios
eas build --platform android
```

#### Submit to App Stores
```bash
eas submit --platform ios
eas submit --platform android
```

---

## 🔒 Security Checklist

- [x] JWT secrets are environment variables
- [x] Passwords hashed with bcrypt
- [x] Input validation with Zod
- [x] SQL injection protection (Prisma)
- [x] CORS configured
- [x] Environment variables not committed
- [ ] HTTPS in production
- [ ] Rate limiting (recommended)
- [ ] API key rotation policy

---

## 📝 Common Issues

### "Cannot connect to server"
- Check server is running on correct port
- Verify API_BASE_URL in client .env
- For physical devices, use local IP address

### "JWT token invalid"
- Check JWT_SECRET matches between client/server
- Token might be expired (default: 7 days)
- Clear app storage and sign in again

### "Database connection failed"
- Verify PostgreSQL is running
- Check DATABASE_URL format
- Ensure database exists

### "Prisma errors"
```bash
cd server
npx prisma generate  # Regenerate client
npx prisma db push   # Sync schema
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

MIT License - see LICENSE file for details

---

## 💬 Support

- **Issues:** GitHub Issues
- **Discussions:** GitHub Discussions
- **Email:** your@email.com

---

**Made with ❤️ by Your Team**
