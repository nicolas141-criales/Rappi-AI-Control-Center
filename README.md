# Rappi Analytics Assistant

Streamlit app for the Rappi AI Engineer technical assessment. Upload an Excel dataset and explore it through interactive tables, visualizations, AI-powered chat, and automated insights.

## Quickstart

```bash
# 1. Create and activate virtual environment (already present as /venv)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Anthropic API key (required for the Chat tab)
$env:ANTHROPIC_API_KEY = "sk-ant-..."   # PowerShell
# export ANTHROPIC_API_KEY="sk-ant-..." # bash

# 4. Run
streamlit run app.py
```

## Project structure

```
rappi-analytics/
├── app.py                  # Main Streamlit application
├── requirements.txt
├── README.md
├── data/                   # Drop your Excel files here
│   └── dataset.xlsx        # Default path (configurable in sidebar)
├── prompts/
│   └── system_prompt.txt   # System prompt for Claude
└── src/
    ├── __init__.py
    └── data_loader.py      # Excel loading + dataset utilities
```

## Features

| Tab | Description |
|-----|-------------|
| **Preview** | Paginated table view of raw data |
| **Statistics** | Descriptive stats + column type summary |
| **Visualize** | Histogram, scatter plot, and bar chart (Plotly) |
| **Chat** | Ask natural-language questions powered by Claude |
| **Insights** | Auto-detected skew, high CV, missing data, and correlations |

## Configuration

- Place Excel files in `data/` or upload via the sidebar.
- Edit `prompts/system_prompt.txt` to customize the assistant's behaviour.
- The chat tab requires `ANTHROPIC_API_KEY` to be set; all other tabs work offline.
