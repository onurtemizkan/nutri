# Nutri - Nutrition Tracking App

A full-stack nutrition tracking application built with React Native (Expo) and Node.js. Track your daily meals, calories, and macronutrients with a beautiful, modern interface.

## Features

### Mobile App
- **Authentication**: Secure sign up/sign in with JWT tokens
- **Daily Dashboard**: View calorie and macronutrient progress at a glance
- **Meal Tracking**: Log breakfast, lunch, dinner, and snacks
- **Nutrition Breakdown**: Track calories, protein, carbs, fat, and fiber
- **User Goals**: Set and manage your daily nutrition goals
- **Profile Management**: Update your goals and account settings

### Backend API
- **RESTful API**: Built with Express.js and TypeScript
- **PostgreSQL Database**: Robust data storage with Prisma ORM
- **JWT Authentication**: Secure token-based authentication
- **Meal Management**: Full CRUD operations for meals
- **Daily/Weekly Summaries**: Get nutrition insights over time

## Tech Stack

### Mobile App
- React Native (Expo)
- TypeScript
- Expo Router (file-based routing)
- Axios (API calls)
- Expo Secure Store (token storage)

### Backend
- Node.js
- TypeScript
- Express.js
- Prisma ORM
- PostgreSQL
- JWT for authentication
- bcryptjs for password hashing
- Zod for validation

## Project Structure

```
nutri/
├── app/                    # Mobile app screens (Expo Router)
│   ├── (tabs)/            # Tab navigation screens
│   │   ├── index.tsx      # Dashboard/Home screen
│   │   └── profile.tsx    # Profile screen
│   ├── auth/              # Authentication screens
│   │   ├── welcome.tsx    # Welcome screen
│   │   ├── signin.tsx     # Sign in screen
│   │   └── signup.tsx     # Sign up screen
│   ├── add-meal.tsx       # Add meal modal
│   └── _layout.tsx        # Root layout with auth routing
├── lib/                   # Shared libraries
│   ├── api/               # API client
│   ├── context/           # React contexts (Auth)
│   └── types/             # TypeScript types
├── server/                # Backend API
│   ├── src/
│   │   ├── controllers/   # Request handlers
│   │   ├── services/      # Business logic
│   │   ├── routes/        # API routes
│   │   ├── middleware/    # Auth & error handling
│   │   ├── config/        # Configuration
│   │   └── types/         # TypeScript types
│   └── prisma/            # Database schema
└── components/            # Reusable UI components
```

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- PostgreSQL (v12 or higher)
- npm or yarn
- Expo CLI (`npm install -g expo-cli`)
- iOS Simulator (for Mac) or Android Emulator

### Backend Setup

1. **Navigate to the server directory**:
   ```bash
   cd server
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Set up environment variables**:
   Create a `.env` file in the `server` directory:
   ```env
   DATABASE_URL="postgresql://postgres:password@localhost:5432/nutri_db"
   PORT=3000
   NODE_ENV=development
   JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
   JWT_EXPIRES_IN=7d
   ```

4. **Create PostgreSQL database**:
   ```bash
   # Using psql
   createdb nutri_db

   # Or using PostgreSQL CLI
   psql -U postgres
   CREATE DATABASE nutri_db;
   \q
   ```

5. **Generate Prisma client and push schema**:
   ```bash
   npm run db:generate
   npm run db:push
   ```

6. **Start the development server**:
   ```bash
   npm run dev
   ```

   The API will be available at `http://localhost:3000`

### Mobile App Setup

1. **Navigate to the project root**:
   ```bash
   cd ..
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Update API URL** (if needed):

   For physical devices, update the API URL in `lib/api/client.ts`:
   ```typescript
   // Change localhost to your computer's IP address
   const API_BASE_URL = __DEV__
     ? 'http://192.168.1.XXX:3000/api'  // Replace with your IP
     : 'https://your-production-api.com/api';
   ```

4. **Start the Expo development server**:
   ```bash
   npm start
   ```

5. **Run on iOS/Android**:
   ```bash
   # iOS Simulator (Mac only)
   npm run ios

   # Android Emulator
   npm run android

   # Web
   npm run web
   ```

## API Documentation

### Base URL
```
http://localhost:3000/api
```

### Authentication Endpoints

#### Register
```http
POST /api/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123",
  "name": "John Doe"
}
```

#### Login
```http
POST /api/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123"
}
```

#### Get Profile (Protected)
```http
GET /api/auth/profile
Authorization: Bearer {token}
```

#### Update Profile (Protected)
```http
PUT /api/auth/profile
Authorization: Bearer {token}
Content-Type: application/json

{
  "goalCalories": 2200,
  "goalProtein": 160,
  "goalCarbs": 220,
  "goalFat": 70
}
```

### Meal Endpoints (All Protected)

#### Create Meal
```http
POST /api/meals
Authorization: Bearer {token}
Content-Type: application/json

{
  "name": "Grilled Chicken Salad",
  "mealType": "lunch",
  "calories": 450,
  "protein": 35,
  "carbs": 30,
  "fat": 18,
  "fiber": 8,
  "servingSize": "1 bowl"
}
```

#### Get Today's Meals
```http
GET /api/meals
Authorization: Bearer {token}
```

#### Get Meals by Date
```http
GET /api/meals?date=2025-01-25T00:00:00.000Z
Authorization: Bearer {token}
```

#### Get Daily Summary
```http
GET /api/meals/summary/daily
Authorization: Bearer {token}
```

#### Get Weekly Summary
```http
GET /api/meals/summary/weekly
Authorization: Bearer {token}
```

#### Update Meal
```http
PUT /api/meals/{mealId}
Authorization: Bearer {token}
Content-Type: application/json

{
  "calories": 500,
  "protein": 40
}
```

#### Delete Meal
```http
DELETE /api/meals/{mealId}
Authorization: Bearer {token}
```

## Database Schema

### User
- id (String, Primary Key)
- email (String, Unique)
- password (String, Hashed)
- name (String)
- goalCalories (Int, Default: 2000)
- goalProtein (Float, Default: 150)
- goalCarbs (Float, Default: 200)
- goalFat (Float, Default: 65)
- currentWeight (Float, Optional)
- goalWeight (Float, Optional)
- height (Float, Optional)
- activityLevel (String, Default: "moderate")

### Meal
- id (String, Primary Key)
- userId (String, Foreign Key → User)
- name (String)
- mealType (String: breakfast | lunch | dinner | snack)
- calories (Float)
- protein (Float)
- carbs (Float)
- fat (Float)
- fiber (Float, Optional)
- sugar (Float, Optional)
- servingSize (String, Optional)
- notes (String, Optional)
- consumedAt (DateTime)

### WaterIntake
- id (String, Primary Key)
- userId (String, Foreign Key → User)
- amount (Float, in ml)
- recordedAt (DateTime)

### WeightRecord
- id (String, Primary Key)
- userId (String, Foreign Key → User)
- weight (Float, in kg)
- recordedAt (DateTime)

## Development Scripts

### Mobile App
```bash
npm start          # Start Expo development server
npm run ios        # Run on iOS simulator
npm run android    # Run on Android emulator
npm run web        # Run in web browser
npm run lint       # Run ESLint
npm test          # Run tests
```

### Backend
```bash
npm run dev        # Start development server with hot reload
npm run build      # Build TypeScript to JavaScript
npm start          # Run production server
npm run db:generate    # Generate Prisma client
npm run db:push        # Push schema to database
npm run db:migrate     # Run migrations
npm run db:studio      # Open Prisma Studio (database GUI)
npm run lint       # Run ESLint
```

## Environment Variables

### Backend (.env)
```env
DATABASE_URL="postgresql://user:password@localhost:5432/nutri_db"
PORT=3000
NODE_ENV=development
JWT_SECRET=your-secret-key
JWT_EXPIRES_IN=7d
```

## Design Philosophy

The app follows modern iOS/Android design patterns with:
- **Minimal UI**: Clean, uncluttered interface
- **Visual Progress**: Circular gauges and progress bars
- **Card-based Layout**: Information grouped in digestible cards
- **Color-coded Data**: Macronutrients use distinct colors
- **Pull-to-Refresh**: Quick data updates
- **Modal Navigation**: Add meal screen as modal overlay

## Future Enhancements

- [ ] Barcode scanning for packaged foods
- [ ] Food database integration (USDA, etc.)
- [ ] Meal history and search
- [ ] Weekly/monthly charts and analytics
- [ ] Water intake tracking
- [ ] Weight tracking with progress charts
- [ ] Photo uploads for meals
- [ ] Recipe creation and sharing
- [ ] Meal plans and suggestions
- [ ] Dark mode support
- [ ] Export data (CSV, PDF)
- [ ] Integration with fitness trackers

## License

MIT

## Author

Built with ❤️ using modern web and mobile technologies
