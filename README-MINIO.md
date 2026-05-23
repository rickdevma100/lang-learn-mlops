# Comprehensive Guide: Integrating MinIO + DVC + BentoML + Helm for Inference on Kubernetes

This document explains the complete end-to-end workflow for:

1. Deploying MinIO inside Kubernetes using Helm
2. Using MinIO as a DVC remote
3. Uploading model artifacts to MinIO through DVC
4. Automatically pulling models during deployment/inference
5. Integrating everything into your BentoML/KServe/Helm architecture

This guide is designed so an AI coding agent like Antigravity 2 can execute it with minimal ambiguity.

---

# 1. Target Architecture

## Final Flow

```text
Local Machine
    |
    | dvc push
    v
MinIO Bucket (S3-compatible object store)
    |
    | dvc pull
    v
Model Loader Kubernetes Job
    |
    v
Persistent Volume (PVC)
    |
    v
BentoML Inference Service
```

---

# 2. Recommended Repository Structure

```text
lang-learn-mlops/
│
├── app/
│   ├── bentoml_service.py
│   └── requirements.txt
│
├── models/
│   └── gemma-4/
│
├── infra/
│   └── helm/
│
│           ├── Chart.yaml
│           ├── values.yaml
│           ├── templates/
│           │   ├── namespace.yaml
│           │   ├── model-pvc.yaml
│           │   ├── model-loader-job.yaml
│           │   ├── inference-deployment.yaml
│           │   ├── inference-service.yaml
│           │   └── ingress.yaml
│           │
│           └── charts/
│
├── .dvc/
├── dvc.yaml
├── params.yaml
└── README.md
```

---

# 3. Deploy MinIO as a Helm Subchart

This is the cleanest approach because:

* Helm manages MinIO lifecycle
* Easy upgrades
* Easy GitOps
* Easy rollback
* No custom YAML maintenance

Official/community Helm chart references: ([GitHub][1])

---

# 4. Configure `Chart.yaml`

File:

```text
infra/helm/lang-learn/Chart.yaml
```

Contents:

```yaml
apiVersion: v2
name: lang-learn
description: Language Learning AI Stack
type: application
version: 0.1.0
appVersion: "1.0.0"

dependencies:
  - name: minio
    version: "5.4.0"
    repository: "https://charts.min.io/"
    condition: minio.enabled
```

---

# 5. Configure `values.yaml`

File:

```text
infra/helm/lang-learn/values.yaml
```

Contents:

```yaml
namespace: lang-learn

minio:
  enabled: true

  rootUser: minioadmin
  rootPassword: minioadmin

  mode: standalone

  persistence:
    enabled: true
    size: 30Gi
    storageClass: "microk8s-hostpath"

  resources:
    requests:
      memory: 256Mi
      cpu: 100m

    limits:
      memory: 1Gi
      cpu: 500m

  buckets:
    - name: dvc-store
      policy: none
      purge: false

  service:
    type: ClusterIP
    port: 9000

  consoleService:
    type: ClusterIP
    port: 9001

  metrics:
    serviceMonitor:
      enabled: false
```

---

# 6. Fetch Helm Dependencies

Run:

```bash
cd infra/helm/lang-learn

helm dependency update
```

This creates:

```text
infra/helm/lang-learn/
├── Chart.lock
├── charts/
│   └── minio-5.4.0.tgz
```

Commit `Chart.lock`.

Do NOT commit:

```text
charts/
```

Add to `.gitignore`:

```gitignore
infra/helm/lang-learn/charts/
```

---

# 7. Deploy MinIO

```bash
helm upgrade --install lang-learn ./infra/helm/lang-learn \
  -n lang-learn \
  --create-namespace
```

Verify:

```bash
kubectl get pods -n lang-learn
```

You should see:

```text
lang-learn-minio-xxxxx
```

---

# 8. Access MinIO Locally

Port-forward:

```bash
kubectl port-forward -n lang-learn svc/lang-learn-minio 9000:9000
```

Console:

```bash
kubectl port-forward -n lang-learn svc/lang-learn-minio-console 9001:9001
```

Open:

```text
http://localhost:9001
```

Credentials:

```text
username: minioadmin
password: minioadmin
```

---

# 9. Install DVC S3 Support

On your local machine:

```bash
pip install "dvc[s3]"
```

---

# 10. Configure DVC Remote

Inside project root:

```bash
cd lang-learn-mlops
```

Remove old remote if present:

```bash
dvc remote remove localstore || true
```

Create MinIO remote:

```bash
dvc remote add -d minio s3://dvc-store
```

Configure endpoint:

```bash
dvc remote modify minio endpointurl http://localhost:9000
```

Configure credentials:

```bash
dvc remote modify minio access_key_id minioadmin

dvc remote modify minio secret_access_key minioadmin
```

Disable SSL:

```bash
dvc remote modify minio use_ssl false
```

---

# 11. Verify DVC Config

Check:

```bash
cat .dvc/config
```

Expected:

```ini
['remote "minio"']
    url = s3://dvc-store
    endpointurl = http://localhost:9000
    access_key_id = minioadmin
    secret_access_key = minioadmin
    use_ssl = false
```

Commit:

```bash
git add .dvc/config
git commit -m "Configure DVC remote for MinIO"
```

---

# 12. Track Model with DVC

Example:

```bash
dvc add models/gemma-4
```

This creates:

```text
models/gemma-4.dvc
```

Commit:

```bash
git add models/gemma-4.dvc .gitignore
git commit -m "Track model using DVC"
```

---

# 13. Push Model to MinIO

Run:

```bash
dvc push
```

DVC uploads artifacts into:

```text
MinIO Bucket -> dvc-store
```

Verify from MinIO UI.

---

# 14. Create Model PVC

File:

```text
infra/helm/lang-learn/templates/model-pvc.yaml
```

Contents:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-storage
  namespace: {{ .Values.namespace }}
spec:
  accessModes:
    - ReadWriteOnce

  resources:
    requests:
      storage: 20Gi

  storageClassName: microk8s-hostpath
```

---

# 15. Create Model Loader Job

This is the most important part.

The job:

1. Starts during deployment
2. Installs DVC
3. Clones repository
4. Connects to MinIO
5. Pulls model
6. Copies model to PVC

---

## File

```text
infra/helm/lang-learn/templates/model-loader-job.yaml
```

---

## Contents

```yaml
apiVersion: batch/v1
kind: Job

metadata:
  name: model-loader

  namespace: {{ .Values.namespace }}

  annotations:
    "helm.sh/hook": pre-install,pre-upgrade
    "helm.sh/hook-weight": "5"
    "helm.sh/hook-delete-policy": before-hook-creation,hook-succeeded

spec:
  backoffLimit: 3

  template:
    spec:
      restartPolicy: OnFailure

      containers:
        - name: dvc-pull

          image: python:3.11-slim

          command:
            - /bin/bash
            - -c
            - |

              set -e

              echo "Installing dependencies..."
              apt-get update
              apt-get install -y git

              pip install -U dvc[s3]

              echo "Cloning repository..."
              cd /workspace

              git clone https://github.com/YOUR_USERNAME/lang-learn-mlops.git

              cd lang-learn-mlops

              echo "Configure in-cluster MinIO endpoint..."

              dvc remote modify minio endpointurl \
                http://lang-learn-minio:9000

              echo "Pulling model from MinIO..."

              dvc pull models/gemma-4.dvc

              echo "Copying model to PVC..."

              mkdir -p /models/gemma-4

              cp -rv models/gemma-4/* /models/gemma-4/

              echo "Done."

          env:
            - name: AWS_ACCESS_KEY_ID
              value: minioadmin

            - name: AWS_SECRET_ACCESS_KEY
              value: minioadmin

          volumeMounts:
            - name: model-storage
              mountPath: /models

            - name: workspace
              mountPath: /workspace

      volumes:
        - name: model-storage
          persistentVolumeClaim:
            claimName: model-storage

        - name: workspace
          emptyDir: {}
```

---

# 16. Why Internal MinIO URL Changes

Outside cluster:

```text
http://localhost:9000
```

Inside cluster:

```text
http://lang-learn-minio:9000
```

Because Kubernetes services use internal DNS.

Format:

```text
http://<service-name>:<port>
```

Or fully qualified:

```text
http://lang-learn-minio.lang-learn.svc.cluster.local:9000
```

---

# 17. Mount PVC into BentoML Deployment

Example deployment:

```yaml
volumeMounts:
  - name: model-storage
    mountPath: /models

volumes:
  - name: model-storage
    persistentVolumeClaim:
      claimName: model-storage
```

Now BentoML can load:

```python
MODEL_PATH = "/models/gemma-4"
```

---

# 18. BentoML Example

```python
import bentoml
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/models/gemma-4"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

@bentoml.service
class GemmaService:

    @bentoml.api
    def generate(self, prompt: str) -> str:

        inputs = tokenizer(prompt, return_tensors="pt")

        outputs = model.generate(**inputs, max_new_tokens=100)

        return tokenizer.decode(outputs[0])
```

---

# 19. Deployment Lifecycle

## What Happens During `helm upgrade`

```text
1. Helm starts deployment

2. model-loader Job executes

3. Job installs DVC

4. Job clones Git repo

5. Job connects to MinIO

6. Job runs dvc pull

7. Model downloaded from MinIO

8. Model copied to PVC

9. Job finishes

10. BentoML pod starts

11. BentoML mounts PVC

12. Model already exists

13. Inference service becomes ready
```

---

# 20. Deploy Everything

```bash
helm upgrade --install lang-learn ./infra/helm/lang-learn \
  -n lang-learn \
  --create-namespace
```

---

# 21. Verify Everything

## Check Jobs

```bash
kubectl get jobs -n lang-learn
```

---

## Check Pods

```bash
kubectl get pods -n lang-learn
```

---

## Check Logs

```bash
kubectl logs job/model-loader -n lang-learn
```

Expected:

```text
Installing dependencies...
Cloning repository...
Pulling model from MinIO...
Copying model to PVC...
Done.
```

---

# 22. Verify Model Exists in PVC

Open shell:

```bash
kubectl exec -it <bentoml-pod> -n lang-learn -- bash
```

Check:

```bash
ls /models/gemma-4
```

---

# 23. Common Problems

---

## DVC Push Fails

### Cause

Port-forward not running.

### Fix

```bash
kubectl port-forward -n lang-learn svc/lang-learn-minio 9000:9000
```

---

## MinIO Pod Pending

### Cause

StorageClass missing.

### Fix

Enable MicroK8s storage:

```bash
microk8s enable hostpath-storage
```

Verify:

```bash
kubectl get storageclass
```

---

## Model Loader Job Stuck

Check:

```bash
kubectl describe job model-loader -n lang-learn
```

And:

```bash
kubectl logs job/model-loader -n lang-learn
```

---

## DVC Pull Fails Inside Cluster

Cause:

Wrong MinIO URL.

Wrong:

```text
localhost:9000
```

Correct:

```text
http://lang-learn-minio:9000
```

