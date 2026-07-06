# Summarization
Summarization uses facebook/bart-large-cnn, fine-tuned on the
CNN/DailyMail dataset. The model produces abstractive summaries
that rephrase rather than copy the original text. Minor factual
merging can occur on short transcripts — a known limitation of
seq2seq summarization models.

Summarization uses facebook/bart-large-cnn, a pre-trained model 
loaded from HuggingFace. The model was fine-tuned by Facebook AI 
on the CNN/DailyMail dataset and produces abstractive summaries 
that rephrase rather than copy the original text.

# Action Extractor
User clicks "Extract action items"
        ↓
flan-t5 reads the transcript
        ↓
Returns a list of tasks + who is responsible
        ↓
Displayed as a clean checklist in Streamlit
Action item extraction uses a rule-based approach combined with keyword matching. It works reliably for structured meetings where speakers use first-person commitments ('I'll', 'I will', 'I can'). Third-person assignments and informal language are partially supported. A limitation of this approach is its dependency on consistent speaker formatting and English-language action keywords.