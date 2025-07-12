const request = require('supertest');
const express = require('express');
const chatRoutes = require('../server/routes/chat');
const ingestRoutes = require('../server/routes/ingest');
const visualizationRoutes = require('../server/routes/visualization');

describe('API Tests', () => {
  let app;

  beforeEach(() => {
    app = express();
    app.use(express.json());
    app.use('/api/chat', chatRoutes);
    app.use('/api/ingest', ingestRoutes);
    app.use('/api/visualizations', visualizationRoutes);
  });

  describe('Chat API', () => {
    test('POST /api/chat should respond to messages', async () => {
      const response = await request(app)
        .post('/api/chat')
        .send({ message: 'Hello' })
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('response');
      expect(response.body.response).toContain('Claude-Flow');
    });

    test('POST /api/chat should require message', async () => {
      const response = await request(app)
        .post('/api/chat')
        .send({})
        .expect(400);

      expect(response.body).toHaveProperty('error', 'Message is required');
    });
  });

  describe('Ingest API', () => {
    test('GET /api/ingest should list files', async () => {
      const response = await request(app)
        .get('/api/ingest')
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('files');
      expect(Array.isArray(response.body.files)).toBe(true);
    });

    test('POST /api/ingest should accept JSON data', async () => {
      const testData = { test: 'data', values: [1, 2, 3] };
      
      const response = await request(app)
        .post('/api/ingest')
        .send({ data: testData })
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('fileId');
      expect(response.body).toHaveProperty('filename');
    });
  });

  describe('Visualization API', () => {
    test('GET /api/visualizations should list visualizations', async () => {
      const response = await request(app)
        .get('/api/visualizations')
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('visualizations');
      expect(Array.isArray(response.body.visualizations)).toBe(true);
    });
  });
});