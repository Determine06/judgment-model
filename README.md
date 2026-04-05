# Judgment Model Pipeline (POC)

## Overview
- Proof-of-concept pipeline that turns a raw Q&A text file into artifacts used for judgment-style inference.
- The build step parses a dataset, performs belief extraction, clusters those beliefs, and builds a belief map and persona spec.
- The inference step retrieves relevant beliefs for a new question, generates a verdict with reasons (reasoning replication), and produces a mimic answer.
- Key components include belief extraction, clustering, reasoning replication, and mimic generation using LLM APIs.

## Pipeline
Data.txt → dataset → beliefs → clusters → belief_map  
→ inference → verdict + reasons → mimic answer

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

