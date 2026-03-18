# Personal Energy Forecast Planner

A Streamlit app that predicts your energy score over the next three days based on lifestyle habits, powered by a RandomForest model trained on Oura ring data. Assignment 2 adds three LLM-powered features built on top of the core forecast engine.

## Features

**Core forecast**
- 3-day energy score prediction with Monte Carlo uncertainty bands
- Baseline vs. scenario comparison across six lifestyle parameters (bedtime, sleep duration, training load, caffeine cutoff, alcohol, late meals)
- Feature importance and model performance metrics

**Feature A - Smart Scenario Builder** (structured output + few-shot prompting)
Describe your upcoming days in plain English and the app extracts all six forecast parameters automatically. No sliders needed.

**Feature B - AI Energy Optimizer** (multi-call agentic tool use loop)
Tell the app your goal (e.g. "maximise energy on Day 2") and an LLM agent autonomously explores the parameter space by calling the forecast simulation as a tool, iterating until it finds the best scenario.

**Feature C - Experiment Log Analyst** (RAG pipeline)
Log your real lifestyle experiments and their energy outcomes. Ask natural language questions and the app retrieves semantically relevant past entries using `text-embedding-3-small` and answers strictly based on your personal data.

## Setup

**Requirements**
- Python 3.12+
- An OpenAI API key

**Installation**

```bash
git clone https://github.com/AlexandercSchumacher/pdai-assignment2.git
cd pdai-assignment2

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**Environment**

```bash
cp .env.example .env
# Open .env and add your OpenAI API key
```

**Run**

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Project structure

```
app.py                  # Streamlit UI (5 tabs)
src/
  llm.py                # All LLM features (scenario parsing, optimizer, RAG)
  forecast.py           # RandomForest model and Monte Carlo simulation
  feature_engineering.py
  data_load.py
  train.py
  viz.py
data/                   # Place your oura_personal.csv here (not tracked)
models/                 # Trained model files (not tracked)
.env.example            # Template for environment variables
requirements.txt
```

## Notes

- `.env` and personal data files (`oura_personal.csv`, `synthetic.csv`, `experiment_log.csv`) are excluded from the repository via `.gitignore`
- Model files (`energy_model.pkl`, `metadata.json`) are also excluded - run `src/train.py` to train locally on your own data
- All LLM calls use `gpt-4o-mini` and `text-embedding-3-small` via the OpenAI API
