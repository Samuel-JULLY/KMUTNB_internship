# Chatbot

## Project Overview

The purpose is to create chatbot specialized in natural disaster and electrical faults,
we focus on compact off-the-shelf model for an easy implementation that requires low hardware. 

## Objectives

- Chatbot comparison
- Chatbot for natural disaster
- Chatbot for Electrical faults and machine learning

# Methodology

- Chosing the models that we test
- Tests
- See results
- Implementing the best model to create 2 chatbot

## Models Used

- Phi3 / Phi4
- LLaMA 3-8B
- Mistral 7B

## Key Results

- Phi3 shows the best results

## Requirements

To run this project, you need Python ≥ 3.8 and the following libraries:

Required Python Packages : 

| Package        | Purpose                                                                                  |
| -------------- | ---------------------------------------------------------------------------------------- |
| `pandas`       | Data loading and manipulation                                                            |
| `gradio`       | Web interface                                                                            |
| `sacrebleu`    | LLM metrics                                                                              |
| `feedparser`   | Web research                                                                             |
| `torch`        | Deeplearning                                                                             |
| `transformers` | Deeplearning models from huggingface                                                     |

You can install all dependencies using pip:

``` python
pip install pandas gradio sacrebleu feedparser torch transformers
```

## Steps to do it

Copy a chatbot and modify the device_map parameter in model :

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=False,
)

if cuda is installed with a Nvidia GPU : set "cuda"
if not : set "cpu"

To change the topics of the chatbot you have to modify the "system_prompt" variable

for electrical faults and machine learning :

system_prompt = (
    "You are an AI assistant specialized strictly in electrical fault and machine learning. "
    "Answer only questions about fault in electrical system and machine learning"
    "If the question is unrelated, politely refuse."
)

## Authors

Samuel JULLY & Valentin OBERT

CESI GRADUATE SCHOOL OF ENGINEERING