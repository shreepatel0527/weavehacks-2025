#!/bin/bash

echo "Setting up Python environment for Claude-Flow Web App..."

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Please install Homebrew first:"
    echo "/bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

# Install Python 3.11+ via Homebrew
echo "Installing Python via Homebrew..."
brew install python@3.11

# Install pipx for isolated Python applications
echo "Installing pipx..."
brew install pipx
pipx ensurepath

# Install essential Python tools
echo "Installing Python development tools..."
pipx install poetry
pipx install black
pipx install ruff

# Create a virtual environment for the project
echo "Creating project virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install scientific computing libraries
echo "Installing scientific computing libraries..."
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scikit-learn
pip install jupyter notebook ipython
pip install plotly dash
pip install scipy statsmodels

# Install data processing libraries
echo "Installing data processing libraries..."
pip install openpyxl xlrd
pip install sqlalchemy psycopg2-binary
pip install requests httpx aiohttp
pip install pydantic

# Install machine learning libraries (optional)
echo "Installing ML libraries..."
pip install torch torchvision torchaudio
pip install transformers datasets
pip install langchain openai

# Install visualization libraries
echo "Installing advanced visualization libraries..."
pip install bokeh altair
pip install folium geopandas
pip install networkx pygraphviz

# Install audio processing (for future voice features)
echo "Installing audio processing libraries..."
pip install pyaudio soundfile librosa
pip install webrtcvad speech_recognition

# Save requirements
echo "Saving requirements..."
pip freeze > requirements-full.txt

# Create a minimal requirements file
cat > requirements-minimal.txt << EOL
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
scipy>=1.10.0
scikit-learn>=1.2.0
jupyter>=1.0.0
requests>=2.28.0
python-dotenv>=1.0.0
EOL

echo "Python environment setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "source venv/bin/activate"
echo ""
echo "Installed tools available via pipx:"
echo "- poetry (dependency management)"
echo "- black (code formatting)"
echo "- ruff (linting)"
echo ""
echo "Key libraries installed:"
echo "- Scientific: numpy, pandas, scipy, scikit-learn"
echo "- Visualization: matplotlib, seaborn, plotly, bokeh"
echo "- ML/AI: torch, transformers, langchain"
echo "- Audio: pyaudio, speech_recognition"