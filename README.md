# weavehacks-2025
The public repo for WeaveHacks 2025, focused on agentic workflows for research scientists

## Vision 

Scientists are tackling some of the world’s most pressing problems; however, their industry remains highly manual, slowing down these brilliant minds in achieving the breakthroughs we need. We are introducing agentic workflows to the research space, with a focus on wet lab scientists, to reduce the burden on these brilliant minds.

## Use Case
Nanoparticles are a hot topic right now; we are exploring applications in energy, cosmetics, robotic technologies, and cancer treatments. Nanoparticles are unique because they allow us a gateway to understand the connection between classical and quantum states of being. As a nanoparticle researcher, I am working on building a nanoparticle that I believe will be a great carrier for targeted cancer therapy. 

This involves up to 16 hours of experiments, the first 3 of which I have to do by hand. All of my work occurs in the fume hood, and I have to wear special gloves and wrist covers to conduct my work because of the nature of the chemicals I am working with. 

Thus, my hands are tied and it is a huge disruption when I need to perform calculations, turn on and off instruments, and monitor for safety conditions. I need to record the mass of my gold compound (I’ll find a name for this I forget), the mass of my sulfur compound (also will find name), and the volume of my solvent (dichloromethane) to do my computations later. 

When I perform an experiment, I want to be able to scan my parameters effectively, record one experiment on one page, and have the agent record both my numbers and my qualitative results. I want to be able to visualize my parameters (volume, heat, etc.) and compare how these parameters relate to my UV-Vis peak (extracted from a chromatogram, determines whether or not I was successful in making my nanoparticle, or how close I am to the right structure)

I would want the agent to: turn on / off lab instruments (centrifuge, UV-Vis, etc.)-- let the agent figure it out -- also implement safety checks and controls on those instruments for shut off, inventory management (notie I'm running out of chemical, tell agent to find it and order it for me), agentic video feed monitoring to automate data collection for overinght experiments, agent communicates to scientist
# WeaveHacks 2025: AI-Powered Lab Assistant

Welcome to the **WeaveHacks 2025 AI-Powered Lab Assistant** project! This repository contains a modular and extensible framework designed to assist wet lab scientists in automating data collection, safety monitoring, lab instrument control, and experiment management. By leveraging CrewAI, voice recognition, and advanced caching, this project streamlines workflows and reduces manual effort in the lab.

## 🚀 Features

### 1. **Data Collection**
- Automates the recording of experimental data using text or voice input.
- Extracts numerical values from prompts or transcriptions.
- Updates the experiment state dynamically.

### 2. **Safety Monitoring**
- Monitors critical parameters like temperature, pressure, and gas levels.
- Notifies scientists of safety concerns and can shut down instruments if necessary.

### 3. **Lab Instrument Control**
- Automates turning lab instruments on/off via voice or programmatic commands.
- Ensures instruments are safely managed during experiments.

### 4. **Voice Recognition**
- Uses voice input for hands-free data collection.
- Provides a fallback to text-based input if voice recognition fails.

### 5. **State Management and Export**
- Tracks the state of the experiment in real-time.
- Converts the experiment state into a CSV format for easy sharing and analysis.

### 6. **Video Monitoring**
- Monitors experiments via video feeds and logs observations.

## 🧠 Key Components

### 1. Agents

DataCollectionAgent: Handles data recording and updates the experiment state.
LabControlAgent: Automates lab instrument control.
SafetyMonitoringAgent: Monitors safety parameters and notifies scientists.
VoiceRecognitionAgent: Converts voice input into text for hands-free operation.

### 2. Crews

DataCollectionCrew: Coordinates data collection tasks.
LabControlCrew: Manages lab instrument control.
SafetyMonitoringCrew: Combines safety monitoring and video monitoring.

### 3. Utilities

CacheManager: Handles caching of experiment states.
Helpers: Provides helper functions for calculations and file I/O.

## 🧑‍🔬 Example Workflow

Initialize Experiment:

The experiment flow initializes the state and sets up the workflow.
Data Collection:

Record masses, volumes, and observations using voice or text input.
Safety Monitoring:

Continuously monitor safety parameters and notify scientists if needed.
Instrument Control:

Automate turning lab instruments on/off during the experiment.
Export Data:

Save the experiment state to a CSV file for analysis.

## 🧪 Testing

Run the test suite to ensure all components are working correctly:


