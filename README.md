<div align="center">

# ♟️FactCheckmate: Preemptively Detecting and Mitigating Hallucinations in LMs

[![Paper](https://img.shields.io/badge/Paper-Arxiv-blue)](https://arxiv.org/abs/2410.02899)

## 📌 Overview
Language models (LMs) hallucinate. Can we detect and mitigate hallucinations before they occur? We introduce FactCheckmate, a method that preemptively detects hallucinations by classifying hidden states before decoding. If hallucination is predicted, FactCheckmate intervenes by adjusting hidden states to enhance factuality. It provides insights into LMs' internal mechanisms, operates efficiently with minimal overhead, and outperforms post-hoc methods. Evaluations across various LM families (Llama, Mistral, Qwen, Gemma) and QA datasets demonstrate over 70% detection accuracy and a 34.4% improvement in factuality.

## 📊 Main Results
Preemptive allucination detection test accuracy. I+O indicates a “reactive” baseline that classifies the LMs’ hidden
states produced over both input questions and output answers, while I preemptively classifies hallucinations based
on the hidden states over only the inputs. −n indicates that the classifier only sees a prefix of the input excluding
the last n tokens.

| LM                      | NQ (I+O) | NQ (I) | NQ (-1) | NQ (-2) | NQ (-3) | MMLU (I+O) | MMLU (I) | MMLU (-1) | MMLU (-2) | MMLU (-3) | MedMCQA (I+O) | MedMCQA (I) | MedMCQA (-1) | MedMCQA (-2) | MedMCQA (-3) | GSM8K (I+O) | GSM8K (I) | GSM8K (-1) | GSM8K (-2) | GSM8K (-3) |
|-------------------------|-----------|--------|---------|---------|---------|-------------|----------|-----------|-----------|-----------|---------------|-------------|--------------|--------------|--------------|-------------|-----------|------------|------------|------------|
| Llama2-7B               | 72.8      | 71.8   | 71.1    | 68.1    | 65.7    | 91.7        | 91.9     | 91.7      | 91.7      | 91.7      | 77.0          | 72.9        | 72.9         | 72.9         | 74.5         | 65.8        | 66.0       | 66.0       | 63.5       | 63.5       |
| Llama2-13B              | 74.4      | 72.0   | 70.6    | 71.4    | 69.7    | 94.0        | 93.0     | 84.1      | 92.7      | 85.7      | 76.0          | 78.3        | 78.6         | 78.3         | 74.2         | 68.4        | 69.1       | 68.4       | 66.8       | 63.8       |
| Llama3-8B               | 74.9      | 70.2   | 68.5    | 66.8    | 66.8    | 93.8        | 94.0     | 87.5      | 87.1      | 77.3      | 77.1          | 76.3        | 74.3         | 71.2         | 67.3         | 71.3        | 72.5       | 72.9       | 71.3       | 66.2       |
| Llama3.1-8B             | 74.3      | 73.1   | 70.9    | 68.9    | 68.1    | 94.5        | 92.3     | 86.3      | 80.0      | 78.0      | 78.4          | 76.2        | 74.9         | 73.6         | 69.4         | 72.3        | 69.1       | 61.2       | 60.2       | 60.6       |
| Mistral-7B              | 73.3      | 72.5   | 71.4    | 71.1    | 70.3    | 93.2        | 90.2     | 83.0      | 82.5      | 82.8      | 77.9          | 75.4        | 75.2         | 73.9         | 72.8         | 69.4        | 70.0       | 70.0       | 71.8       | 71.8       |
| Gemma-7B                | 80.2      | 74.5   | 74.4    | 74.2    | 73.9    | 92.2        | 96.9     | 91.3      | 81.5      | 89.6      | 77.0          | 77.5        | 74.7         | 75.2         | 75.2         | 70.9        | 67.4       | 67.0       | 67.4       | 67.8       |
| Qwen2.5-7B              | 76.1      | 74.5   | 72.7    | 71.2    | 69.2    | 94.3        | 94.0     | 71.3      | 85.4      | 74.4      | 78.9          | 76.6        | 76.9         | 75.7         | 74.9         | 67.0        | 67.2       | 67.2       | 67.2       | 67.0       |
| Llama2-7B-chat          | 76.2      | 74.5   | 67.5    | 69.4    | 69.2    | 94.8        | 90.9     | 79.1      | 83.3      | 83.0      | 81.1          | 79.3        | 79.3         | 79.0         | 79.3         | 72.3        | 73.6       | 72.2       | 72.9       | 72.2       |
| Llama2-13B-chat         | 74.8      | 72.4   | 70.7    | 70.9    | 68.5    | 93.8        | 93.9     | 80.1      | 78.6      | 92.1      | 81.3          | 73.3        | 70.8         | 70.0         | 71.9         | 72.3        | 71.9       | 71.9       | 71.9       | 71.9       |
| Llama3-8B-Instruct      | 81.5      | 78.6   | 77.2    | 76.4    | 75.0    | 93.8        | 95.6     | 87.4      | 85.0      | 79.5      | 81.4          | 77.3        | 72.2         | 71.1         | 68.8         | 74.7        | 74.3       | 74.3       | 74.3       | 74.3       |
| Llama3.1-8B-Instruct    | 83.3      | 74.5   | 71.3    | 70.9    | 66.7    | 93.1        | 91.8     | 86.4      | 85.4      | 80.1      | 81.7          | 78.8        | 76.5         | 71.7         | 70.1         | 76.2        | 78.4       | 78.0       | 78.0       | 78.4       |
| Llama3-70B-Instruct     | 81.0      | 77.1   | 73.3    | 69.6    | 65.9    | 87.6        | 79.6     | 76.5      | 76.4      | 73.4      | 74.7          | 67.6        | 64.5         | 63.6         | 61.0         | 82.7        | 78.8       | 72.5       | 71.3       | 69.4       |


