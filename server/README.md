# Nutri Backend API

RESTful API for the Nutri nutrition tracking application. Built with Node.js, Express, TypeScript, and PostgreSQL.

## Quick Start

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Set up environment variables**:
   Copy `.env.example` to `.env` and configure:
   ```bash
   cp .env.example .env
   ```

3. **Create PostgreSQL database**:
   ```bash
   createdb nutri_db
   ```

4. **Generate Prisma client and sync database**:
   ```bash
   npm run db:generate
   npm run db:push
   ```

5. **Start development server**:
   ```bash
   npm run dev
   ```

   Server will run on `http://localhost:3000`

## Scripts

- `npm run dev` - Start development server with hot reload
- `npm run build` - Build TypeScript to JavaScript
- `npm start` - Run production server
- `npm run db:generate` - Generate Prisma client
- `npm run db:push` - Push schema changes to database
- `npm run db:migrate` - Run database migrations
- `npm run db:studio` - Open Prisma Studio (database GUI)
- `npm run lint` - Run ESLint

## API Endpoints

### Health Check
```
GET /health
```
Returns server status and timestamp.

### Authentication (Public)
- `POST /api/auth/register` - Create new user account
- `POST /api/auth/login` - Login and receive JWT token

### User Profile (Protected)
- `GET /api/auth/profile` - Get current user profile
- `PUT /api/auth/profile` - Update user profile

### Meals (Protected)
- `POST /api/meals` - Create a new meal
- `GET /api/meals` - Get meals (optionally by date)
- `GET /api/meals/:id` - Get specific meal
- `PUT /api/meals/:id` - Update meal
- `DELETE /api/meals/:id` - Delete meal
- `GET /api/meals/summary/daily` - Get daily nutrition summary
- `GET /api/meals/summary/weekly` - Get weekly nutrition summary

## Authentication

All protected endpoints require a JWT token in the Authorization header:

```
Authorization: Bearer {your-jwt-token}
```

Tokens are returned from `/api/auth/register` and `/api/auth/login` endpoints.

## Environment Variables

Create a `.env` file in the server directory:

```env
DATABASE_URL="postgresql://postgres:password@localhost:5432/nutri_db"
PORT=3000
NODE_ENV=development
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
JWT_EXPIRES_IN=7d
```

## Database

This project uses Prisma ORM with PostgreSQL. The schema is defined in `prisma/schema.prisma`.

### Useful Prisma Commands

```bash
# Generate Prisma client after schema changes
npm run db:generate

# Push schema to database (dev only)
npm run db:push

# Create and run migrations (production)
npm run db:migrate

# Open Prisma Studio to view/edit data
npm run db:studio

# Reset database (WARNING: deletes all data)
npx prisma migrate reset
```

## Project Structure

```
server/
├── src/
│   ├── controllers/       # HTTP request handlers
│   ├── services/          # Business logic
│   ├── routes/            # API route definitions
│   ├── middleware/        # Custom middleware (auth, errors)
│   ├── config/            # Configuration files
│   ├── types/             # TypeScript type definitions
│   └── index.ts           # Application entry point
├── prisma/
│   └── schema.prisma      # Database schema
├── .env                   # Environment variables (create this)
├── .env.example          # Example environment variables
├── package.json
└── tsconfig.json
```

## Development

The server uses `tsx` for development with hot reload. Any changes to TypeScript files will automatically restart the server.

## Production

1. Build the TypeScript code:
   ```bash
   npm run build
   ```

2. Start the production server:
   ```bash
   npm start
   ```

## Testing

Example requests using curl:

### Register
```bash
curl -X POST http://localhost:3000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "password123",
    "name": "John Doe"
  }'
```

### Login
```bash
curl -X POST http://localhost:3000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "password123"
  }'
```

### Create Meal
```bash
curl -X POST http://localhost:3000/api/meals \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -d '{
    "name": "Grilled Chicken Salad",
    "mealType": "lunch",
    "calories": 450,
    "protein": 35,
    "carbs": 30,
    "fat": 18,
    "fiber": 8,
    "servingSize": "1 bowl"
  }'
```

## License

MIT
