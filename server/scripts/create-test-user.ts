#!/usr/bin/env tsx

/**
 * Create Test User Script
 * Creates the test user required for integration tests if it doesn't exist
 */

import { PrismaClient } from '@prisma/client';
import bcrypt from 'bcryptjs';

const prisma = new PrismaClient();

const TEST_USER = {
  email: 'testuser@example.com',
  password: 'Test123456',
  name: 'Test User',
  goalCalories: 2000,
  goalProtein: 150,
  goalCarbs: 200,
  goalFat: 65,
};

async function createTestUser() {
  try {
    console.log('Checking for existing test user...');

    // Check if test user already exists
    const existingUser = await prisma.user.findUnique({
      where: { email: TEST_USER.email },
    });

    if (existingUser) {
      console.log('✓ Test user already exists');
      console.log(`  Email: ${existingUser.email}`);
      console.log(`  Name: ${existingUser.name}`);
      console.log(`  ID: ${existingUser.id}`);
      return;
    }

    console.log('Creating test user...');

    // Hash password
    const hashedPassword = await bcrypt.hash(TEST_USER.password, 10);

    // Create test user
    const user = await prisma.user.create({
      data: {
        email: TEST_USER.email,
        password: hashedPassword,
        name: TEST_USER.name,
        goalCalories: TEST_USER.goalCalories,
        goalProtein: TEST_USER.goalProtein,
        goalCarbs: TEST_USER.goalCarbs,
        goalFat: TEST_USER.goalFat,
      },
    });

    console.log('✓ Test user created successfully');
    console.log(`  Email: ${user.email}`);
    console.log(`  Name: ${user.name}`);
    console.log(`  ID: ${user.id}`);
  } catch (error) {
    console.error('Error creating test user:', error);
    process.exit(1);
  } finally {
    await prisma.$disconnect();
  }
}

createTestUser();
