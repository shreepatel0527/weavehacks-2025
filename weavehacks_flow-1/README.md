# README.md

# WeaveHacks Flow

WeaveHacks Flow is a project designed to automate data collection, lab instrument control, and safety monitoring for laboratory experiments. This project utilizes CrewAI to streamline workflows and enhance the efficiency of lab operations.

## Project Structure

```
weavehacks_flow
├── src
│   ├── weavehacks_flow
│   │   ├── main.py                  # Entry point for the application
│   │   ├── agents
│   │   │   ├── data_collection_agent.py  # Automates data collection tasks
│   │   │   ├── lab_control_agent.py      # Controls lab instruments
│   │   │   └── safety_monitoring_agent.py # Monitors safety parameters
│   │   ├── crews
│   │   │   ├── data_collection_crew
│   │   │   │   └── data_collection_crew.py # Orchestrates data collection tasks
│   │   │   ├── lab_control_crew
│   │   │   │   └── lab_control_crew.py     # Manages lab instrument operations
│   │   │   └── safety_monitoring_crew
│   │   │       └── safety_monitoring_crew.py # Oversees safety monitoring tasks
│   │   └── utils
│   │       └── helpers.py                # Utility functions for various tasks
├── requirements.txt                       # Lists project dependencies
└── README.md                              # Project documentation
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd weavehacks_flow
pip install -r requirements.txt
```

## Usage

To run the application, execute the following command:

```bash
python src/weavehacks_flow/main.py
```

This will initialize the experiment flow, allowing for automated data collection, lab control, and safety monitoring.

## Components Overview

### Main Flow

- **ExperimentFlow**: Manages the overall workflow, integrating various agents and crews for data collection, lab control, and safety monitoring.

### Agents

- **DataCollectionAgent**: Automates data collection tasks, including recording data and managing reagent information.
- **LabControlAgent**: Controls lab instruments, allowing for operations based on voice commands or agent interactions.
- **SafetyMonitoringAgent**: Monitors safety parameters during experiments and notifies scientists of potential safety concerns.

### Crews

- **DataCollectionCrew**: Orchestrates data collection tasks using the DataCollectionAgent.
- **LabControlCrew**: Manages the operation of lab instruments using the LabControlAgent.
- **SafetyMonitoringCrew**: Oversees safety monitoring tasks using the SafetyMonitoringAgent.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.