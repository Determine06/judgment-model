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
    
## Architecture Generation (Build Phase)

This phase constructs the internal reasoning space from raw Q&A data.

Process
- Parse raw Q&A dataset into structured format
- Extract belief primitives from each answer using an LLM
- Generate embeddings for each belief (optionally conditioned on question context)
- Perform hierarchical clustering to group semantically similar beliefs
- Apply NLI-based refinement within clusters to reduce contradictions and improve coherence
- Construct a belief_map.json linking:
    cluster → beliefs → source questions
- Generate a persona_spec.json capturing consistent response patterns (structure, tone, reasoning style)

Key Idea: Instead of storing full answers, the system builds a compressed belief space that represents how the individual reasons across scenarios.

Metrics Used (Build Validation)
- Cluster Cohesion (Embedding Similarity): Measures intra-cluster semantic similarity
- NLI Entailment Rate: % of belief pairs within a cluster that entail each other
- Contradiction Rate (NLI): % of belief pairs that contradict (target: low)
- Mean NLI Score (Soft Consistency): For a cluster with beliefs b1, b2, ..., bn, we compute:
  
    ![Mean NLI](https://latex.codecogs.com/png.latex?\frac{1}{n(n-1)}\sum_{i\ne%20j}P_{entail}(b_i,b_j))
