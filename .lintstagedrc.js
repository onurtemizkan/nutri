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
  // Uses venv-installed tools via explicit shell, falls back gracefully
  'ml-service/app/**/*.py': (filenames) => {
    // Quote each file path to handle spaces
    const files = filenames.map((f) => `"${f}"`).join(' ');
    return [
      // Run through sh -c to enable shell constructs (cd, &&, ||)
      `sh -c 'cd ml-service && . venv/bin/activate 2>/dev/null && black ${files} || black ${files} 2>/dev/null || echo "Skipping black"'`,
      `sh -c 'cd ml-service && . venv/bin/activate 2>/dev/null && isort ${files} || isort ${files} 2>/dev/null || echo "Skipping isort"'`,
    ];
  },
  'ml-service/tests/**/*.py': (filenames) => {
    const files = filenames.map((f) => `"${f}"`).join(' ');
    return [
      `sh -c 'cd ml-service && . venv/bin/activate 2>/dev/null && black ${files} || black ${files} 2>/dev/null || echo "Skipping black"'`,
      `sh -c 'cd ml-service && . venv/bin/activate 2>/dev/null && isort ${files} || isort ${files} 2>/dev/null || echo "Skipping isort"'`,
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
