# Quick Start Guide

Get the Nutri app running in 5 minutes!

## Prerequisites

- Node.js installed
- PostgreSQL installed and running
- Expo CLI (`npm install -g expo-cli`)

## Step 1: Clone and Install

```bash
# Install root dependencies (mobile app)
npm install

# Install server dependencies
cd server
npm install
cd ..
```

## Step 2: Set Up Database

```bash
# Create PostgreSQL database
createdb nutri_db

# Navigate to server directory
cd server

# Create .env file
cp .env.example .env

# Edit .env and update DATABASE_URL if needed
# Default: postgresql://postgres:password@localhost:5432/nutri_db

# Generate Prisma client and sync database
npm run db:generate
npm run db:push

cd ..
```

## Step 3: Start the Backend

```bash
# In a new terminal, from the server directory
cd server
npm run dev
```

Server will run on `http://localhost:3000`

## Step 4: Start the Mobile App

```bash
# In another terminal, from the project root
npm start

# Then choose:
# - Press 'i' for iOS simulator
# - Press 'a' for Android emulator
# - Scan QR code with Expo Go app on your phone
```

## Step 5: Create an Account

1. App will open to the Welcome screen
2. Tap "Create Account"
3. Fill in your details (name, email, password)
4. You're in! Start tracking your meals

## Common Issues

### Can't connect to database
- Make sure PostgreSQL is running: `pg_isready`
- Check your DATABASE_URL in `server/.env`
- Try creating the database manually: `createdb nutri_db`

### Mobile app can't reach server
- If using a physical device, update API URL in `lib/api/client.ts` with your computer's IP address
- Make sure server is running on port 3000
- Check firewall settings

### Port 3000 already in use
- Change the PORT in `server/.env` to another port (e.g., 3001)
- Update API URL in `lib/api/client.ts` to match

## Next Steps

- Add your first meal from the + button on the home screen
- Update your daily goals in the Profile tab
- Track your nutrition progress!

## Need Help?

See the full [README.md](README.md) for detailed documentation.
