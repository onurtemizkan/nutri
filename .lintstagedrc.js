/**
 * lint-staged configuration
 * Runs formatters and linters on staged files before commit
 */
const path = require('path');

module.exports = {
  // JavaScript/TypeScript files in mobile app
  'app/**/*.{js,jsx,ts,tsx}': ['eslint --fix', 'prettier --write'],
  'lib/**/*.{js,jsx,ts,tsx}': ['eslint --fix', 'prettier --write'],
  'hooks/**/*.{js,jsx,ts,tsx}': ['eslint --fix', 'prettier --write'],

  // JavaScript/TypeScript files in server
  // Run eslint from server directory for proper tsconfig resolution
  'server/src/**/*.ts': (filenames) => {
    const relativeFiles = filenames.map((f) =>
      path.relative(path.join(__dirname, 'server'), f)
    );
    return [
      `cd server && eslint --fix ${relativeFiles.join(' ')}`,
      `prettier --write ${filenames.join(' ')}`,
    ];
  },

  // Python files in ml-service (excluding venv)
  'ml-service/app/**/*.py': (filenames) => {
    return [
      // Black for formatting
      `black ${filenames.join(' ')}`,
      // isort for import sorting
      `isort ${filenames.join(' ')}`,
    ];
  },
  'ml-service/tests/**/*.py': (filenames) => {
    return [
      `black ${filenames.join(' ')}`,
      `isort ${filenames.join(' ')}`,
    ];
  },

  // JSON files (excluding package-lock.json)
  '*.json': ['prettier --write'],
  'server/*.json': ['prettier --write'],

  // Markdown files
  '*.md': ['prettier --write'],

  // YAML files
  '*.{yml,yaml}': ['prettier --write'],
  '.github/**/*.{yml,yaml}': ['prettier --write'],
};
