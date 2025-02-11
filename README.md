#  Text Summarization with Hugging Face

## Introduction

This repository demonstrates how to use **Hugging Face Transformers** for text summarization. We focus on two state-of-the-art models:
- **BART (`facebook/bart-large-cnn`)**
- **T5 (`t5-large`)**

Both models are designed for sequence-to-sequence tasks, making them ideal for text summarization.

## What is Hugging Face?

[Hugging Face](https://huggingface.co/) is an AI company that provides an open-source library (`transformers`) for natural language processing (NLP). It offers pre-trained models for various NLP tasks, such as text classification, translation, summarization, and more.

## Overview of T5 and BART



### 1. T5 (Text-to-Text Transfer Transformer)
- Developed by Google, **T5** treats all NLP tasks as text-to-text problems.
- It is highly flexible and performs well in summarization, translation, and text generation.

### 2. BART (Bidirectional and Auto-Regressive Transformer)
- Developed by Facebook, **BART** is optimized for text generation and summarization.
- It is a combination of BERT (bidirectional) and GPT (autoregressive) architectures.

## Setup

To run the summarization models, install the required dependencies:

```bash
pip install transformers torch
```

## Code Implementation

### 1. Summarization using BART

```python
from transformers import pipeline

# Load the BART summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)

# Splitting text into chunks (max 1024 characters per chunk)
chunk_size = 1024
article_chunks = [article_text[i:i+chunk_size] for i in range(0, len(article_text), chunk_size)]

# Summarize each chunk
summaries = summarizer(article_chunks, max_length=150, min_length=50, do_sample=False, batch_size=8)
summaries = [summary['summary_text'] for summary in summaries]

# Combine all summaries
final_summary = " ".join(summaries)

print(final_summary)
```

### 2. Summarization using T5

```python
from transformers import pipeline

# Load the T5 summarization model
summarizer = pipeline("summarization", model="t5-large", device=0)

# Splitting text into chunks
chunk_size = 1024
article_chunks = [article_text[i:i+chunk_size] for i in range(0, len(article_text), chunk_size)]

# Summarize each chunk
summaries = summarizer(article_chunks, max_length=150, min_length=50, do_sample=False, batch_size=8)
summaries = [summary['summary_text'] for summary in summaries]

# Combine all summaries
final_summary = " ".join(summaries)

print(final_summary)
```

## How It Works
1. The text is divided into chunks of **1024 characters** (since transformers have input limits).
2. Each chunk is passed to the summarization model.
3. The model generates summaries for each chunk separately.
4. All summaries are concatenated to form the final summarized text.

## Conclusion
This repository provides an easy-to-use implementation for text summarization using **BART** and **T5** models from Hugging Face. These models can be fine-tuned for specific domains to improve summarization quality.

## References
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [BART Model](https://huggingface.co/facebook/bart-large-cnn)
- [T5 Model](https://huggingface.co/t5-large)


