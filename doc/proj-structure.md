meeting-assistant/

├── data/
│   └── sample_transcripts/
│       └── example_meeting.txt
├── notebooks/                    ← optional, for experiments
├── src/
│   ├── __init__.py
│   ├── summarizer.py
│   ├── action_extractor.py
│   ├── sentiment.py
│   └── rag.py
├── models/                       ← HF models cache here automatically
├── reports/
│   └── figures/                  ← charts saved here
├── config/
│   └── config.yaml               ← model names, params (no hardcoded values)
├── tests/
│   └── test_summarizer.py
├── app.py                        ← Streamlit entry point
├── requirements.txt
├── README.md
└── .gitignore