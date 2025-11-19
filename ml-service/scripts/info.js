#!/usr/bin/env node

/**
 * Project Information - Display useful project info
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

console.log('\n╔════════════════════════════════════════════════════════════╗');
console.log('║         Nutri ML Service - Project Information          ║');
console.log('╚════════════════════════════════════════════════════════════╝\n');

// Project Info
console.log('📦 PROJECT');
console.log('  Name:        Nutri ML Service');
console.log('  Description: LSTM predictions, SHAP explainability, What-if analysis');
console.log('  Version:     1.0.0\n');

// Python Version
try {
  const pythonVersion = execSync('python3 --version', { encoding: 'utf-8' }).trim();
  console.log('🐍 PYTHON');
  console.log(`  Version:     ${pythonVersion.replace('Python ', '')}`);

  // Check venv
  const venvPath = path.join(__dirname, '..', 'venv');
  if (fs.existsSync(venvPath)) {
    console.log('  VirtualEnv:  ✅ Created (./venv)');

    // Check if venv is activated
    if (process.env.VIRTUAL_ENV) {
      console.log('  Status:      ✅ Activated');
    } else {
      console.log('  Status:      ⚠️  Not activated');
    }
  } else {
    console.log('  VirtualEnv:  ❌ Not created');
  }
  console.log('');
} catch (error) {
  console.log('🐍 PYTHON');
  console.log('  Version:     ❌ Python 3 not found\n');
}

// Redis Status
console.log('📡 REDIS');
try {
  const redisVersion = execSync('redis-server --version', { encoding: 'utf-8' }).split('\n')[0];
  console.log(`  Installed:   ✅ ${redisVersion.match(/v=([0-9.]+)/)?.[1] || 'Yes'}`);

  try {
    execSync('redis-cli -p 6379 ping', { encoding: 'utf-8', stdio: 'pipe' });
    console.log('  Status:      ✅ Running (port 6379)');
  } catch {
    console.log('  Status:      ⚠️  Not running');
  }
} catch (error) {
  console.log('  Installed:   ❌ Not installed');
  console.log('  Install:     npm run setup:redis');
}
console.log('');

// Environment Files
console.log('⚙️  ENVIRONMENT');
const envFile = path.join(__dirname, '..', '.env');
const envTestFile = path.join(__dirname, '..', '.env.test');
console.log(`  .env:        ${fs.existsSync(envFile) ? '✅ Created' : '❌ Missing'}`);
console.log(`  .env.test:   ${fs.existsSync(envTestFile) ? '✅ Created' : '❌ Missing'}`);
console.log('');

// Test Results
console.log('🧪 TESTS');
const pytestCache = path.join(__dirname, '..', '.pytest_cache');
if (fs.existsSync(pytestCache)) {
  console.log('  Last Run:    ✅ Tests have been run');

  // Try to read last test results
  const lastFailedFile = path.join(pytestCache, 'v', 'cache', 'lastfailed');
  if (fs.existsSync(lastFailedFile)) {
    try {
      const lastFailed = JSON.parse(fs.readFileSync(lastFailedFile, 'utf-8'));
      const failedCount = Object.keys(lastFailed).length;
      if (failedCount === 0) {
        console.log('  Status:      ✅ All tests passed');
      } else {
        console.log(`  Status:      ⚠️  ${failedCount} test(s) failed`);
      }
    } catch (error) {
      console.log('  Status:      Unknown');
    }
  }
} else {
  console.log('  Last Run:    ⚠️  No test runs yet');
}
console.log('');

// Models Directory
console.log('🤖 ML MODELS');
const modelsDir = path.join(__dirname, '..', 'models');
if (fs.existsSync(modelsDir)) {
  const modelFiles = fs.readdirSync(modelsDir).filter(f => f.endsWith('.pth'));
  if (modelFiles.length > 0) {
    console.log(`  Trained:     ✅ ${modelFiles.length} model(s) saved`);
    modelFiles.slice(0, 3).forEach(file => {
      console.log(`               - ${file}`);
    });
    if (modelFiles.length > 3) {
      console.log(`               ... and ${modelFiles.length - 3} more`);
    }
  } else {
    console.log('  Trained:     ⚠️  No trained models yet');
  }
} else {
  console.log('  Directory:   ⚠️  Not created yet');
}
console.log('');

// Quick Commands
console.log('🚀 QUICK COMMANDS');
console.log('  Setup:       npm run setup');
console.log('  Test:        npm test');
console.log('  Dev Server:  npm run dev');
console.log('  Redis:       npm run redis:start');
console.log('  Full Guide:  npm run help\n');

console.log('═══════════════════════════════════════════════════════════\n');
