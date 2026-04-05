# Judgment Model Pipeline (POC)

## Overview
This project is a proof-of-concept system for replicating human evaluative judgment.

It reconstructs how a person reasons about problems by:
- extracting belief primitives
- clustering them into a reasoning space
- retrieving relevant beliefs for new scenarios
- generating verdicts and explanations
- rendering answers in a consistent persona

Inspired by Delphi AI.

## Pipeline
Data.txt → dataset → beliefs → clusters → belief_map  
inference → verdict + reasons → mimic answer

## Project Structure
- src/build → preprocessing/build pipeline that parses the dataset, extracts beliefs, clusters them, and builds the belief map/persona
- src/inference → runtime inference to retrieve beliefs, generate verdicts + reasons, and produce a mimic answer
- data/processed → intermediate and final artifacts written by the build (dataset.json, beliefs_extracted.json[l], belief_map.json, persona_spec.json)
- notebooks/ → experimentation and exploratory work (source for the ported logic)

## How to Run

### 1. Install dependencies
    pip install -r requirements.txt

### 2. Set environment variables
    cp .env.example .env
    # add your OpenAI key to .env

### 3. Build artifacts
    python src/build/run_build.py --input data/raw/Data.txt

### 4. Run inference
    python src/inference/run_inference.py       --input data/inputs/questions.json       --output data/outputs/output.json

## Example Output
    {
      question: ...,
      verdict: ...,
      reasons: [..., ..., ...],
      mimic_answer: ...
    }
    
## Notes
- Not production optimized
- Designed as research + system POC

