# Lang-Learn MLOps Platform — Comprehensive Infrastructure Context Document

This document explains the current architecture, deployment flow, Helm structure, KServe integration, DVC model loading approach, ArgoCD GitOps setup, and operational assumptions for the `lang-learn` project.

This document is intended as a complete engineering context for Antigravity 2 so it can continue implementation, enhancement, debugging, and operationalization without ambiguity.

---

# 1. Project Goal

The project is an AI-powered language learning platform using:

* Gemma 4 model
* BentoML inference container
* KServe for model serving
* Helm for Kubernetes packaging
* ArgoCD for GitOps deployment
* DVC + MinIO for model artifact management
* MicroK8s running inside Multipass VM
* GPU acceleration (NVIDIA)
* Frontend UI communicating with KServe predictor service

---

# 2. Current Infrastructure Assumptions

## Environment

Host Machine:

* macOS

Virtualization:

* Multipass VM

Inside VM:

* Ubuntu
* MicroK8s Kubernetes cluster

MLOps Components:

* KServe
* ArgoCD
* MinIO
* DVC
* BentoML

Inference Model:

* Gemma 4

---

# 3. Important Architectural Clarification

## PVC Already Exists

The PersistentVolumeClaim is ALREADY created manually.

PVC Name:

```yaml
model-storage
```

Therefore:

* DO NOT recreate PVC
* DO NOT apply `model-pvc.yaml`
* DO NOT generate PVC resources automatically
* Helm chart should assume PVC already exists

This is a very important customization.

---

# 4. Current Folder Structure (Actual Structure Used)

The actual folder structure differs from the original screenshots.

Use THIS as the source of truth:

```text
infra/
├── argocd/
│   ├── app-of-apps.yaml
│   └── inference-app.yaml
│
├── helm/
│   ├── charts/
│   │
│   ├── templates/
│   │   ├── _helpers.tpl
│   │   ├── configmap-prompts.yaml
│   │   ├── frontend-deployment.yaml
│   │   ├── frontend-ingress.yaml
│   │   ├── frontend-service.yaml
│   │   ├── inference-kserve.yaml
│   │   ├── model-loader-job.yaml
│   │   ├── namespace.yaml
│   │   └── prometheus-rules.yaml
│   │
│   ├── Chart.lock
│   ├── Chart.yaml
│   └── values.yaml
```

---

# 5. Core Deployment Architecture

The deployment contains these major components:

| Component                   | Purpose                       |
| --------------------------- | ----------------------------- |
| BentoML Inference Container | Runs Gemma inference          |
| KServe InferenceService     | Handles serving/autoscaling   |
| Frontend Deployment         | UI/API frontend               |
| ConfigMap Prompts           | Prompt templates              |
| Model Loader Job            | Pulls model from DVC into PVC |
| ArgoCD                      | GitOps deployment             |
| Helm                        | Packaging and templating      |
| MinIO + DVC                 | Model artifact storage        |

---

# 6. Deployment Flow

## High-Level Flow

```text
Git Push
   ↓
ArgoCD Sync
   ↓
Helm Chart Applied
   ↓
Namespace Created
   ↓
ConfigMaps Created
   ↓
Model Loader Job Runs
   ↓
Model Pulled from DVC
   ↓
Model Copied to PVC
   ↓
KServe InferenceService Starts
   ↓
Frontend Connects to Predictor Service
```

---

# 7. Helm Chart Overview

## Chart.yaml

Purpose:
Defines Helm chart metadata.

Example:

```yaml
apiVersion: v2
name: lang-learn
description: Language Learning MLOps Stack
type: application
version: 0.1.0
appVersion: "1.0.0"
```

---

# 8. values.yaml Structure

This is the central configuration file.

## Example Structure

```yaml
namespace: lang-learn

inference:
  image:
    repository: localhost:32000/inference
    tag: latest
    pullPolicy: Always

  replicas: 1

  minReplicas: 1
  maxReplicas: 3

  scaleTarget: 1

  model:
    path: /app/models/gemma-4-e4b-it
    dtype: bfloat16
    quantization: none
    useFlashAttention: true

  resources:
    requests:
      memory: "16Gi"
      cpu: "4"
      nvidia.com/gpu: "1"

    limits:
      memory: "24Gi"
      nvidia.com/gpu: "1"

frontend:
  image:
    repository: localhost:32000/frontend
    tag: latest

  replicas: 1

ingress:
  enabled: true
  host: langlearn.local
```

---

# 9. _helpers.tpl

Purpose:
Reusable Helm labels/helpers.

Example:

```yaml
{{- define "lang-learn.labels" -}}
app.kubernetes.io/name: lang-learn
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
```

---

# 10. namespace.yaml

Purpose:
Creates namespace.

Example:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: {{ .Values.namespace }}
```

---

# 11. configmap-prompts.yaml

Purpose:
Mount prompt templates into inference container.

This allows:

* prompt engineering
* dynamic prompt updates
* version-controlled prompts

Example:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prompts
  namespace: {{ .Values.namespace }}

data:
  image_describe.txt: |-
{{ .Files.Get "../../prompts/image_describe.txt" | indent 4 }}

  scenario_dialogue.txt: |-
{{ .Files.Get "../../prompts/scenario_dialogue.txt" | indent 4 }}
```

---

# 12. inference-kserve.yaml (Most Important File)

This is the core deployment file.

It defines the KServe `InferenceService`.

---

# 13. KServe Deployment Details

## Key Features Used

### RawDeployment Mode

```yaml
serving.kserve.io/deploymentMode: RawDeployment
```

This bypasses Knative complexity.

---

### HPA Autoscaling

```yaml
serving.kserve.io/autoscalerClass: hpa
```

Uses Kubernetes HPA instead of Knative autoscaler.

---

### Concurrency Scaling

```yaml
scaleMetric: concurrency
```

Scale based on concurrent requests.

---

# 14. KServe Predictor Container

The predictor container runs BentoML inference.

Example Structure:

```yaml
containers:
  - name: kserve-container

    image: "{{ .Values.inference.image.repository }}:{{ .Values.inference.image.tag }}"

    imagePullPolicy: {{ .Values.inference.image.pullPolicy }}

    ports:
      - containerPort: 3000

    env:
      - name: MODEL_PATH
        value: {{ .Values.inference.model.path | quote }}

      - name: MODEL_DTYPE
        value: {{ .Values.inference.model.dtype | quote }}

      - name: MODEL_QUANTIZATION
        value: {{ .Values.inference.model.quantization | quote }}

      - name: USE_FLASH_ATTENTION
        value: {{ .Values.inference.model.useFlashAttention | quote }}

      - name: PROMPTS_DIR
        value: "/app/prompts"

      - name: HF_HUB_OFFLINE
        value: "1"
```

---

# 15. Volume Mounts

## Model Mount

The existing PVC is mounted:

```yaml
volumeMounts:
  - name: model-storage
    mountPath: /app/models/gemma-4-e4b-it
    readOnly: true
```

## Prompt Mount

```yaml
  - name: prompts
    mountPath: /app/prompts
    readOnly: true
```

---

# 16. Existing PVC Usage

IMPORTANT:

The deployment MUST use existing PVC:

```yaml
volumes:
  - name: model-storage
    persistentVolumeClaim:
      claimName: model-storage
```

Again:

* Do NOT create PVC
* Assume it already exists

---

# 17. Readiness and Liveness Probes

## Readiness

```yaml
readinessProbe:
  httpGet:
    path: /healthz
    port: 3000

  initialDelaySeconds: 180
```

Large delay is needed because:

* Gemma model loading is slow

---

## Liveness

```yaml
livenessProbe:
  httpGet:
    path: /healthz
    port: 3000
```

---

# 18. Frontend Deployment

Frontend communicates with KServe predictor.

## Important API URL

```yaml
NEXT_PUBLIC_API_URL
```

Example:

```yaml
value: "http://lang-learn-inference-predictor.{{ .Values.namespace }}.svc.cluster.local"
```

This is critical.

KServe automatically creates:

```text
<inferenceservice-name>-predictor
```

service internally.

---

# 19. Frontend Service

Exposes frontend internally.

Example:

```yaml
kind: Service
spec:
  selector:
    app: frontend

  ports:
    - port: 80
      targetPort: 3000
```

---

# 20. Frontend Ingress

Exposes frontend externally.

Example:

```yaml
ingress:
  enabled: true
  host: langlearn.local
```

---

# 21. Model Loader Job (Very Important)

File:

```text
templates/model-loader-job.yaml
```

Purpose:

* Pull model from DVC
* Copy into existing PVC
* Execute BEFORE KServe starts

---

# 22. Why Model Loader Job Exists

MicroK8s inside Multipass cannot directly access macOS filesystem safely.

Therefore:

* DVC artifacts must be pulled INSIDE cluster
* Then copied into PVC

This avoids:

* hostPath instability
* VM mount issues
* Mac filesystem dependency

---

# 23. Model Loader Job Flow

```text
Job Starts
   ↓
Install git + dvc
   ↓
Clone repository
   ↓
DVC pull model
   ↓
Copy model to PVC
   ↓
Job completes
```

---

# 24. Helm Hook Strategy

The job runs automatically BEFORE deployment.

Example:

```yaml
annotations:
  "helm.sh/hook": pre-install,pre-upgrade
```

This guarantees:

* model exists before inference starts

---

# 25. Model Loader Job Structure

Example:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: model-loader
```

---

# 26. DVC Pull Logic

Example:

```bash
git clone https://github.com/yourname/lang-learn-mlops.git

dvc pull models/gemma-4-e4b-it
```

---

# 27. Copy to PVC

Example:

```bash
cp -rv models/gemma-4-e4b-it/* /models/
```

Where:

```text
/models
```

is mounted from:

```yaml
claimName: model-storage
```

---

# 28. DVC Credentials

The job expects Kubernetes Secret.

Example:

```yaml
secretKeyRef:
  name: dvc-credentials
```

Keys:

```text
access_key
secret_key
```

---

# 29. Required Kubernetes Secret

Example creation:

```bash
kubectl create secret generic dvc-credentials \
  --from-literal=access_key=MINIO_ACCESS_KEY \
  --from-literal=secret_key=MINIO_SECRET_KEY \
  -n lang-learn
```

---

# 30. KServe Installation Assumption

KServe already installed using Helm.

Likely setup:

```bash
helm repo add kserve https://kserve.github.io/helm-charts
```

Then:

```bash
helm install kserve kserve/kserve
```

Using:

* RawDeployment
* HPA

---

# 31. ArgoCD Structure

Current structure:

```text
infra/argocd/
├── app-of-apps.yaml
└── inference-app.yaml
```

---

# 32. inference-app.yaml

This deploys the Helm chart through ArgoCD.

Core structure:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
```

---

# 33. ArgoCD Helm Source

Example:

```yaml
source:
  repoURL: https://github.com/yourname/lang-learn-mlops
  targetRevision: main
  path: infra/helm

  helm:
    valueFiles:
      - values.yaml
```

---

# 34. Destination

```yaml
destination:
  server: https://kubernetes.default.svc
  namespace: lang-learn
```

---

# 35. Automated Sync

```yaml
syncPolicy:
  automated:
    prune: true
    selfHeal: true
```

Benefits:

* self-healing
* GitOps
* automatic deployment

---

# 36. Important Sync Option

```yaml
syncOptions:
  - CreateNamespace=true
  - ServerSideApply=true
```

`ServerSideApply=true` is important for:

* CRDs
* KServe resources

---

# 37. Deployment Workflow

## First Manual Validation

Before ArgoCD:

```bash
helm lint infra/helm

helm install lang-learn infra/helm \
  --namespace lang-learn \
  --create-namespace
```

---

# 38. Observe KServe

```bash
kubectl get inferenceservice -n lang-learn -w
```

Expected:

```text
READY=True
```

---

# 39. Check Pods

```bash
kubectl get pods -n lang-learn
```

---

# 40. Check Logs

Inference logs:

```bash
kubectl logs -n lang-learn <pod-name>
```

Model loader logs:

```bash
kubectl logs job/model-loader -n lang-learn
```

---

# 41. Port Forward Testing

Example:

```bash
kubectl port-forward -n lang-learn \
svc/lang-learn-inference-predictor 3000:80
```

---

# 42. Test Endpoint

Example:

```bash
curl -X POST http://localhost:3000/describe-image \
  -F "image=@images/cat.jpg" \
  -F 'metadata={"language":"German","level":"A1"}'
```