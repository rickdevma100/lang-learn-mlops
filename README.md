# lang-learn-mlops
A language learning app

## Experimentation Pipeline

This project uses DVC to track prompt engineering experiments and evaluate prompt quality metrics. The evaluation script is decoupled from the model and calls the BentoML inference service via HTTP.

### 1. Start the inference service (Docker)

Before running evaluations, you must start the inference service using Docker:

```bash
cd inference
docker run --rm -p 3001:3000 \
  -v $(pwd)/../models/gemma-q4:/app/models/gemma-q4 \
  <your-image-name>
```
*(Replace `<your-image-name>` with the actual image tag you use for the BentoML container).*

### 2. Run experiments

Run DVC experiments from the repository root:

```bash
# Run a named experiment
.venv/bin/dvc exp run --name "baseline"

# Edit a prompt file, then run another experiment
vim inference/prompts/scenario_dialogue.txt
.venv/bin/dvc exp run --name "explicit-grammar"

# Run an experiment by overriding parameters in params.yaml on the fly
.venv/bin/dvc exp run --name "high-temp" -S "inference.temperature=0.9"
.venv/bin/dvc exp run --name "long-output" -S "inference.max_tokens=1024"
```

### 3. Compare experiments

```bash
# Show all experiments in a table
.venv/bin/dvc exp show

# Show current metrics
.venv/bin/dvc metrics show

# Compare metrics between two experiments
.venv/bin/dvc metrics diff
```

### 4. Apply the best experiment

When you are satisfied with a specific experiment, you can apply it to your workspace:

```bash
# Apply winning experiment to your workspace
.venv/bin/dvc exp apply <experiment-name>

# Commit the changes
git add params.yaml inference/prompts/ metrics/
git commit -m "feat: use <experiment-name> prompt"
git push
```

### 5. Troubleshooting

```bash
# If you encounter a lock error during DVC execution
rm -f .dvc/tmp/rwlock

# View the DVC pipeline DAG
.venv/bin/dvc dag

# Perform a dry run to see what would execute without actually running it
.venv/bin/dvc repro --dry
```
