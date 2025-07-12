const express = require('express');
const router = express.Router();
const multer = require('multer');
const fs = require('fs').promises;
const path = require('path');

// Configure multer for audio file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, '../uploads/audio');
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const timestamp = Date.now();
    const filename = `audio-${timestamp}-${file.originalname}`;
    cb(null, filename);
  }
});

const upload = multer({ 
  storage,
  fileFilter: (req, file, cb) => {
    const allowedTypes = ['audio/wav', 'audio/mp3', 'audio/webm', 'audio/ogg'];
    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid audio file type'), false);
    }
  },
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB limit
  }
});

class DataCollectionAgent {
  constructor() {
    this.experiments = new Map();
    this.protocols = new Map();
    this.reagentDatabase = this.initializeReagentDatabase();
    this.currentExperiment = null;
  }

  initializeReagentDatabase() {
    return {
      'gold_compound': {
        name: 'Gold Chloride (HAuCl4)',
        type: 'metal_salt',
        units: 'g',
        molecular_weight: 339.79,
        hazards: ['corrosive', 'oxidizer'],
        storage_temp: 'room_temperature'
      },
      'sulfur_compound': {
        name: 'Sulfur Powder',
        type: 'element',
        units: 'g',
        molecular_weight: 32.06,
        hazards: ['flammable'],
        storage_temp: 'room_temperature'
      },
      'dichloromethane': {
        name: 'Dichloromethane (DCM)',
        type: 'solvent',
        units: 'mL',
        molecular_weight: 84.93,
        hazards: ['toxic', 'volatile'],
        storage_temp: 'cool'
      },
      'toluene': {
        name: 'Toluene',
        type: 'solvent',
        units: 'mL',
        molecular_weight: 92.14,
        hazards: ['flammable', 'toxic'],
        storage_temp: 'room_temperature'
      }
    };
  }

  createExperiment(protocol, researcher) {
    const experimentId = `exp_${Date.now()}`;
    const experiment = {
      id: experimentId,
      protocol,
      researcher,
      startTime: new Date().toISOString(),
      dataPoints: [],
      observations: [],
      reagents: {},
      status: 'active',
      phase: 'preparation'
    };
    
    this.experiments.set(experimentId, experiment);
    this.currentExperiment = experimentId;
    
    return experiment;
  }

  processVoiceInput(transcription, experimentId) {
    const experiment = this.experiments.get(experimentId || this.currentExperiment);
    if (!experiment) {
      throw new Error('No active experiment found');
    }

    const result = this.parseTranscription(transcription);
    
    if (result.type === 'reagent_data') {
      this.updateReagentData(experiment, result.data);
    } else if (result.type === 'observation') {
      this.addObservation(experiment, result.data);
    } else if (result.type === 'measurement') {
      this.addMeasurement(experiment, result.data);
    } else if (result.type === 'question') {
      return this.handleQuestion(result.data);
    }

    this.experiments.set(experiment.id, experiment);
    return {
      success: true,
      processed: result,
      experiment: experiment
    };
  }

  parseTranscription(text) {
    const lowerText = text.toLowerCase();
    
    // Reagent mass patterns
    const massPatterns = [
      /(?:mass|weight) of (?:the )?(\w+)(?: compound)? is ([\d.]+)\s*g/i,
      /(\w+)(?: compound)? (?:mass|weight):?\s*([\d.]+)\s*g/i,
      /([\d.]+)\s*g of (\w+)/i
    ];

    // Volume patterns
    const volumePatterns = [
      /volume of (?:the )?(\w+) is ([\d.]+)\s*ml/i,
      /(\w+) volume:?\s*([\d.]+)\s*ml/i,
      /([\d.]+)\s*ml of (\w+)/i
    ];

    // Temperature patterns
    const tempPatterns = [
      /temperature is ([\d.]+)\s*(?:degrees? )?(?:celsius|c)/i,
      /temp:?\s*([\d.]+)\s*(?:degrees? )?(?:celsius|c)/i,
      /([\d.]+)\s*(?:degrees? )?(?:celsius|c)/i
    ];

    // Pressure patterns
    const pressurePatterns = [
      /pressure is ([\d.]+)\s*(?:kpa|kilopascals?)/i,
      /pressure:?\s*([\d.]+)\s*(?:kpa|kilopascals?)/i,
      /([\d.]+)\s*(?:kpa|kilopascals?)/i
    ];

    // Observation patterns
    const observationPatterns = [
      /(?:i (?:see|observe|notice))|(?:there (?:is|are))|(?:formation of)|(?:color chang)/i,
      /(?:black particles)|(?:orange solution)|(?:steam formation)/i,
      /(?:heating|cooling|mixing|stirring)/i
    ];

    // Question patterns
    const questionPatterns = [
      /(?:which|what|how|where|when|why)/i,
      /(?:do you mean|unclear|unsure)/i
    ];

    // Check for reagent mass
    for (const pattern of massPatterns) {
      const match = text.match(pattern);
      if (match) {
        const [, compound, value] = match;
        return {
          type: 'reagent_data',
          data: {
            reagent: this.identifyReagent(compound),
            property: 'mass',
            value: parseFloat(value),
            units: 'g',
            rawText: text
          }
        };
      }
    }

    // Check for volume
    for (const pattern of volumePatterns) {
      const match = text.match(pattern);
      if (match) {
        const [, compound, value] = match;
        return {
          type: 'reagent_data',
          data: {
            reagent: this.identifyReagent(compound),
            property: 'volume',
            value: parseFloat(value),
            units: 'mL',
            rawText: text
          }
        };
      }
    }

    // Check for temperature
    for (const pattern of tempPatterns) {
      const match = text.match(pattern);
      if (match) {
        return {
          type: 'measurement',
          data: {
            parameter: 'temperature',
            value: parseFloat(match[1]),
            units: '°C',
            timestamp: new Date().toISOString(),
            rawText: text
          }
        };
      }
    }

    // Check for pressure
    for (const pattern of pressurePatterns) {
      const match = text.match(pattern);
      if (match) {
        return {
          type: 'measurement',
          data: {
            parameter: 'pressure',
            value: parseFloat(match[1]),
            units: 'kPa',
            timestamp: new Date().toISOString(),
            rawText: text
          }
        };
      }
    }

    // Check for observations
    if (observationPatterns.some(pattern => pattern.test(text))) {
      return {
        type: 'observation',
        data: {
          description: text,
          timestamp: new Date().toISOString(),
          phase: 'observation'
        }
      };
    }

    // Check for questions
    if (questionPatterns.some(pattern => pattern.test(text))) {
      return {
        type: 'question',
        data: {
          question: text,
          timestamp: new Date().toISOString()
        }
      };
    }

    // Default to observation if no specific pattern matched
    return {
      type: 'observation',
      data: {
        description: text,
        timestamp: new Date().toISOString(),
        phase: 'general'
      }
    };
  }

  identifyReagent(compound) {
    const normalized = compound.toLowerCase();
    
    const reagentMap = {
      'gold': 'gold_compound',
      'sulfur': 'sulfur_compound',
      'dichloromethane': 'dichloromethane',
      'dcm': 'dichloromethane',
      'toluene': 'toluene',
      'solvent': 'dichloromethane' // Default solvent
    };

    return reagentMap[normalized] || normalized;
  }

  updateReagentData(experiment, data) {
    if (!experiment.reagents[data.reagent]) {
      experiment.reagents[data.reagent] = {};
    }
    
    experiment.reagents[data.reagent][data.property] = {
      value: data.value,
      units: data.units,
      timestamp: new Date().toISOString(),
      rawText: data.rawText
    };
  }

  addObservation(experiment, data) {
    experiment.observations.push({
      ...data,
      id: `obs_${Date.now()}`
    });
  }

  addMeasurement(experiment, data) {
    experiment.dataPoints.push({
      ...data,
      id: `meas_${Date.now()}`
    });
  }

  handleQuestion(data) {
    // This would implement intelligent question handling
    // For now, return a structured response for clarification
    return {
      type: 'clarification_needed',
      question: data.question,
      suggestions: this.generateSuggestions(data.question)
    };
  }

  generateSuggestions(question) {
    const availableReagents = Object.keys(this.reagentDatabase);
    return availableReagents.map(reagent => ({
      reagent,
      name: this.reagentDatabase[reagent].name
    }));
  }

  generateDataSheet(experimentId) {
    const experiment = this.experiments.get(experimentId);
    if (!experiment) {
      throw new Error('Experiment not found');
    }

    return {
      experiment_info: {
        id: experiment.id,
        researcher: experiment.researcher,
        start_time: experiment.startTime,
        protocol: experiment.protocol,
        status: experiment.status
      },
      reagents: Object.entries(experiment.reagents).map(([name, data]) => ({
        name,
        details: this.reagentDatabase[name] || {},
        measurements: data
      })),
      observations: experiment.observations,
      measurements: experiment.dataPoints,
      summary: this.generateSummary(experiment)
    };
  }

  generateSummary(experiment) {
    return {
      total_observations: experiment.observations.length,
      total_measurements: experiment.dataPoints.length,
      reagents_used: Object.keys(experiment.reagents).length,
      duration: this.calculateDuration(experiment.startTime),
      key_findings: this.extractKeyFindings(experiment)
    };
  }

  calculateDuration(startTime) {
    const start = new Date(startTime);
    const now = new Date();
    const durationMs = now - start;
    const hours = Math.floor(durationMs / 3600000);
    const minutes = Math.floor((durationMs % 3600000) / 60000);
    return `${hours}h ${minutes}m`;
  }

  extractKeyFindings(experiment) {
    const findings = [];
    
    // Analyze observations for key events
    experiment.observations.forEach(obs => {
      if (obs.description.toLowerCase().includes('black particles')) {
        findings.push('Nanoparticle formation observed');
      }
      if (obs.description.toLowerCase().includes('color change')) {
        findings.push('Color change documented');
      }
      if (obs.description.toLowerCase().includes('steam')) {
        findings.push('Heat generation detected');
      }
    });

    return findings;
  }
}

// Global data collection agent instance
const dataAgent = new DataCollectionAgent();

// Ensure upload directory exists
const uploadDir = path.join(__dirname, '../uploads/audio');
fs.mkdir(uploadDir, { recursive: true }).catch(console.error);

// Routes
router.post('/experiment/create', (req, res) => {
  try {
    const { protocol, researcher } = req.body;
    
    if (!protocol || !researcher) {
      return res.status(400).json({ 
        error: 'Protocol and researcher are required' 
      });
    }

    const experiment = dataAgent.createExperiment(protocol, researcher);
    
    res.json({
      success: true,
      experiment
    });
  } catch (error) {
    console.error('Failed to create experiment:', error);
    res.status(500).json({ 
      error: 'Failed to create experiment',
      details: error.message 
    });
  }
});

router.post('/voice/upload', upload.single('audio'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'Audio file is required' });
    }

    // In a real implementation, this would use speech-to-text service
    // For demo purposes, we'll simulate transcription
    const mockTranscription = req.body.transcript || 
      "The mass of the gold compound is 0.1598 grams";

    const { experimentId } = req.body;
    const result = dataAgent.processVoiceInput(mockTranscription, experimentId);

    // Clean up uploaded file after processing
    await fs.unlink(req.file.path);

    res.json({
      success: true,
      transcription: mockTranscription,
      processed: result
    });
  } catch (error) {
    console.error('Voice processing error:', error);
    res.status(500).json({ 
      error: 'Failed to process voice input',
      details: error.message 
    });
  }
});

router.post('/text/process', (req, res) => {
  try {
    const { text, experimentId } = req.body;
    
    if (!text) {
      return res.status(400).json({ error: 'Text input is required' });
    }

    const result = dataAgent.processVoiceInput(text, experimentId);
    
    res.json({
      success: true,
      processed: result
    });
  } catch (error) {
    console.error('Text processing error:', error);
    res.status(500).json({ 
      error: 'Failed to process text input',
      details: error.message 
    });
  }
});

router.get('/experiment/:id', (req, res) => {
  try {
    const { id } = req.params;
    const experiment = dataAgent.experiments.get(id);
    
    if (!experiment) {
      return res.status(404).json({ error: 'Experiment not found' });
    }

    res.json({
      success: true,
      experiment
    });
  } catch (error) {
    console.error('Failed to get experiment:', error);
    res.status(500).json({ 
      error: 'Failed to retrieve experiment',
      details: error.message 
    });
  }
});

router.get('/experiment/:id/datasheet', (req, res) => {
  try {
    const { id } = req.params;
    const dataSheet = dataAgent.generateDataSheet(id);
    
    res.json({
      success: true,
      dataSheet
    });
  } catch (error) {
    console.error('Failed to generate datasheet:', error);
    res.status(500).json({ 
      error: 'Failed to generate datasheet',
      details: error.message 
    });
  }
});

router.get('/experiments', (req, res) => {
  try {
    const experiments = Array.from(dataAgent.experiments.values());
    
    res.json({
      success: true,
      experiments,
      total: experiments.length
    });
  } catch (error) {
    console.error('Failed to list experiments:', error);
    res.status(500).json({ 
      error: 'Failed to list experiments',
      details: error.message 
    });
  }
});

router.get('/reagents', (req, res) => {
  try {
    res.json({
      success: true,
      reagents: dataAgent.reagentDatabase
    });
  } catch (error) {
    console.error('Failed to get reagents:', error);
    res.status(500).json({ 
      error: 'Failed to retrieve reagents',
      details: error.message 
    });
  }
});

module.exports = { router, dataAgent };