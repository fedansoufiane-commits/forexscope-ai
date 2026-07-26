#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
streamlit run app.py
