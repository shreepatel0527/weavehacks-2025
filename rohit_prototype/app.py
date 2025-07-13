import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import speech_recognition as sr
from datetime import datetime
import re
import wandb
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize W&B with API key
wandb.login(key=os.getenv('WANDB_API_KEY'))
wandb.init(project="research-helper")

st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")

# Initialize session state for data storage
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame()
    st.session_state.voice_input_values = {
        'name': '',
        'age': 25,
        'category': 'A',
        'sales': 1000.0,
        'region': 'North',
        'date': datetime.now().date()
    }

# Voice Recognition Class with W&B Integration
class VoiceRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()

    def listen(self):
        try:
            with self.mic as source:
                st.write("🎤 Listening...")
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source)
                st.write("Processing...")
                text = self.recognizer.recognize_google(audio)
                
                # Log voice recognition event
                wandb.log({
                    "event_type": "voice_recognition",
                    "status": "success",
                    "text": text.lower(),
                    "timestamp": str(datetime.now())
                })
                
                return text.lower()
        except Exception as e:
            # Log error event
            wandb.log({
                "event_type": "voice_recognition",
                "status": "error",
                "error_message": str(e),
                "timestamp": str(datetime.now())
            })
            st.error(f"Error: {str(e)}")
            return None

    def extract_info(self, text):
        info = {
            'name': None,
            'age': None,
            'category': None,
            'sales': None,
            'region': None,
            'date': None
        }
        
        # Extract name (assume it's mentioned after 'name is' or 'name')
        name_match = re.search(r'name (?:is )?([a-zA-Z]+)', text)
        if name_match:
            info['name'] = name_match.group(1).capitalize()

        # Extract age
        age_match = re.search(r'age (?:is )?([0-9]+)', text)
        if age_match:
            info['age'] = int(age_match.group(1))

        # Extract category
        if 'category a' in text:
            info['category'] = 'A'
        elif 'category b' in text:
            info['category'] = 'B'
        elif 'category c' in text:
            info['category'] = 'C'
        elif 'category d' in text:
            info['category'] = 'D'

        # Extract sales amount
        sales_match = re.search(r'sales (?:is |amount )?([0-9]+(?:\.[0-9]+)?)', text)
        if sales_match:
            info['sales'] = float(sales_match.group(1))

        # Extract region
        regions = ['north', 'south', 'east', 'west']
        for region in regions:
            if region in text:
                info['region'] = region.capitalize()
                break

        # Extract date
        if 'today' in text:
            info['date'] = datetime.now().date()
        else:
            date_match = re.search(r'date (?:is )?([0-9]{4}/[0-9]{2}/[0-9]{2})', text)
            if date_match:
                info['date'] = datetime.strptime(date_match.group(1), '%Y/%m/%d').date()
            else:
                info['date'] = datetime.now().date()

        # Log extraction results
        wandb.log({
            "event_type": "info_extraction",
            "input_text": text,
            "extracted_info": {
                k: str(v) if v is not None else None 
                for k, v in info.items()
            },
            "timestamp": str(datetime.now())
        })

        return info

    def handle_navigation(self, text):
        # Log navigation attempt
        wandb.log({
            "event_type": "navigation",
            "command": text,
            "timestamp": str(datetime.now())
        })
        
        # Navigation commands
        if any(phrase in text for phrase in ['go to sales', 'show sales', 'open sales']):
            st.switch_page("pages/01_Sales_Analysis.py")
        elif any(phrase in text for phrase in ['go to demographics', 'show demographics', 'open demographics']):
            st.switch_page("pages/02_Demographics.py")
        elif any(phrase in text for phrase in ['go to data', 'show data management', 'open management']):
            st.switch_page("pages/03_Data_Management.py")
        elif any(phrase in text for phrase in ['go home', 'main page', 'home page']):
            st.switch_page("app.py")

# Initialize voice recognizer
if 'recognizer' not in st.session_state:
    st.session_state.recognizer = VoiceRecognizer()

st.title('📊 Interactive Data Analysis Dashboard')

# Voice Navigation Section
st.sidebar.header('🎤 Voice Navigation')
if st.sidebar.button('🎯 Navigate by Voice'):
    st.sidebar.write("Please speak a navigation command:")
    st.sidebar.write("""
    Available commands:
    - "Go to sales" or "Show sales"
    - "Go to demographics"
    - "Go to data management"
    - "Go home" or "Main page"
    """)
    
    voice_input = st.session_state.recognizer.listen()
    if voice_input:
        st.sidebar.write("You said:", voice_input)
        st.session_state.recognizer.handle_navigation(voice_input)

# Voice input button
st.header('Voice Data Entry')
if st.button('🎤 Start Voice Input'):
    st.write("Please speak the data in this format:")
    st.write("'Name is John, age is 30, category A, sales 5000, region north, date is 2023/07/12'")
    
    voice_input = st.session_state.recognizer.listen()
    if voice_input:
        st.write("You said:", voice_input)
        info = st.session_state.recognizer.extract_info(voice_input)
        
        # Update session state values
        for key, value in info.items():
            if value is not None:
                st.session_state.voice_input_values[key] = value
        st.rerun()

# Data Input Section with values from voice input
col1, col2 = st.columns(2)

with col1:
    name = st.text_input('Name', value=st.session_state.voice_input_values['name'])
    age = st.number_input('Age', 0, 120, value=st.session_state.voice_input_values['age'])
    category = st.selectbox('Category', ['A', 'B', 'C', 'D'], 
                          index=['A', 'B', 'C', 'D'].index(st.session_state.voice_input_values['category']))

with col2:
    sales = st.number_input('Sales Amount', 0.0, 1000000.0, 
                           value=float(st.session_state.voice_input_values['sales']))
    region = st.selectbox('Region', ['North', 'South', 'East', 'West'],
                         index=['North', 'South', 'East', 'West'].index(st.session_state.voice_input_values['region']))
    date = st.date_input('Date', value=st.session_state.voice_input_values['date'])

if st.button('Add Data'):
    new_data = pd.DataFrame({
        'Name': [name],
        'Age': [age],
        'Category': [category],
        'Sales': [sales],
        'Region': [region],
        'Date': [date]
    })
    if st.session_state.data.empty:
        st.session_state.data = new_data
    else:
        st.session_state.data = pd.concat([st.session_state.data, new_data], ignore_index=True)
    
    # Log data addition event
    wandb.log({
        "event_type": "data_addition",
        "new_entry": {
            "name": name,
            "age": age,
            "category": category,
            "sales": sales,
            "region": region,
            "date": str(date)
        },
        "total_entries": len(st.session_state.data),
        "timestamp": str(datetime.now())
    })
    
    st.success('Data added successfully!')
    
    # Reset voice input values
    st.session_state.voice_input_values = {
        'name': '',
        'age': 25,
        'category': 'A',
        'sales': 1000.0,
        'region': 'North',
        'date': datetime.now().date()
    }
    st.rerun()

# Display current data
if not st.session_state.data.empty:
    st.header('Current Data')
    st.dataframe(st.session_state.data)
