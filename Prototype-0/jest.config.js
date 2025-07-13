module.exports = {
  testEnvironment: 'node',
  testMatch: ['**/test/**/*.test.js'],
  collectCoverageFrom: [
    'server/**/*.js',
    '!server/index.js'
  ],
  coverageDirectory: 'coverage',
  testTimeout: 10000
};