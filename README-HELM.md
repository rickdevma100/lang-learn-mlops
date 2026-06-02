# Helm Deployment Guide & Reference (`lang-learn`)

This guide serves as a comprehensive reference for all Helm-related commands used to deploy, upgrade, and manage the `lang-learn` MLOps infrastructure stack.

---

## 1. Overview of the Helm Chart Structure

The unified Helm chart located in `infra/helm` manages the entire deployment lifecycle of:
1. **MinIO** (DVC Remote Object Store) - Deployed as a subchart dependency.
2. **Model Loader Job** - Downloads the Gemma models from MinIO to a Persistent Volume using a K8s job hook.
3. **KServe InferenceService** - Deploys the CPU-only BentoML inference runner in `RawDeployment` mode.
4. **Frontend Deployment & Service** - Serves the web interface (optional).
5. **Ingress Controller Paths** - Configures HTTP routing for the Inference endpoints via Ingress.

---

## 2. Helm Commands Directory

### 🚀 Dependency Management
Before deploying, Helm must resolve and pull downstream chart dependencies (specifically the `minio` subchart).

#### **Fetch and Download Subcharts**
```bash
helm dependency update ./infra/helm
```
* **Why it's used:** Reads `Chart.yaml` dependencies (MinIO) and downloads the corresponding `.tgz` archive to `infra/helm/charts/`.
* **When to run:** Run this once initially, or whenever you modify the dependency declarations in `Chart.yaml`.

---

### 🔍 Validation and Local Rendering
Always validate your Kubernetes manifests locally before submitting them to the cluster API.

#### **Static Syntax Checking**
```bash
helm lint ./infra/helm
```
* **Why it's used:** Analyzes templates for syntactic errors, indentation issues, or missing parameter bindings without modifying the cluster.

#### **Dry-Run Rendering**
```bash
helm template lang-learn ./infra/helm -f ./infra/helm/values.yaml
```
* **Why it's used:** Renders all templates locally and dumps the final computed Kubernetes YAML resource definitions directly to stdout. Very helpful for verifying that custom values, labels, and conditionals are processing exactly as expected.

---

### 📦 Installation & Upgrades
Deploying changes to the Kubernetes cluster.

#### **Bootstrap / First-Time Deploy (Bypassing Hook Deadlock)**
```bash
helm upgrade --install lang-learn ./infra/helm -n lang-learn --create-namespace --no-hooks
```
* **Why it's used:** Installs or upgrades the release in the `lang-learn` namespace.
* **Why `--no-hooks` is critical:** The Model Loader Job is structured as a Helm `pre-install/pre-upgrade` hook. However, the Model Loader depends on MinIO being fully ready to pull down model artifacts. Since MinIO is inside the *same* chart, running the installer normally would result in a **deadlock** (the job waits forever for MinIO to launch, but MinIO won't launch until the pre-install hook finishes). Bypassing hooks on first boot allows MinIO to start.
* **When to run:** Initial installation.

#### **Standard Upgrade (With Hooks Enabled)**
```bash
helm upgrade --install lang-learn ./infra/helm -n lang-learn
```
* **Why it's used:** Standard upgrade command. Once MinIO is running and DVC objects have been pushed, hooks can be safely processed to rerun model loaders during updates.
* **When to run:** Every subsequent update where you change parameters in `values.yaml` or make layout adjustments.

---

### 🩺 Status and Inspection
Commands to monitor active Helm deployments.

#### **List Active Releases**
```bash
helm list -n lang-learn
```
* **Why it's used:** Shows the release names, status (`deployed`, `failed`, etc.), revision number, and chart version.

#### **Check Status of a Specific Release**
```bash
helm status lang-learn -n lang-learn
```
* **Why it's used:** Shows the detailed state of the deployed release, including notes, custom hooks, and active resources.

#### **Track History of Revisions**
```bash
helm history lang-learn -n lang-learn
```
* **Why it's used:** Lists all past deployments of the chart with their respective revisions and execution status. Useful for audit trailing and troubleshooting upgrades.

---

### 🔄 Rollbacks and Deletion
Undoing deployments and performing full teardowns.

#### **Roll back to a Safe Version**
```bash
helm rollback lang-learn <revision-number> -n lang-learn
```
* **Why it's used:** Roll back the entire deployment to a previous revision in case a new Helm upgrade fails or causes performance issues.

#### **Completely Delete Release**
```bash
helm uninstall lang-learn -n lang-learn
```
* **Why it's used:** Tears down and cleanses all Kubernetes resources spawned by this Helm chart (Deployments, Services, ConfigMaps, Ingresses, Hooks) except for Persistent Volumes (depending on the reclaim policy).

---

## 3. Best Practices Applied Here

1. **Deterministic Namespace**: All templates are explicitly scoped using `namespace: {{ .Values.namespace }}` rather than relying on default kubectl contexts.
2. **Conditional Frontend Rendering**: Frontend templates utilize `{{- if .Values.frontend.enabled }}` gates so resources are only created if explicitly enabled.
3. **Ingress Inversion**: Solved KServe Istio class conflicts by introducing a standalone `inference-ingress-nginx` template targeted with `ingressClassName: public` and extended NGINX proxy timeouts.
