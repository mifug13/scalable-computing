# infrastructure

## Prerequisites

**Tools:**
- gcloud
- terraform
- kubectl

**Google cloud cli authentication:**

Authenticate to Google cloud:
```
gcloud auth login
```

Initialize Google Cloud CLI:
```
gcloud init
```

Set up Application Default Credentials:
```
gcloud auth application-default login
```

## Getting started

1. **Create tfvars**

Create a `terraform.tfvars` file in `./cluster` to configure the project_id using the following example:

```
project_id = "your-project-id"
```

2. **Run terraform**

In `./cluster` run:

```
terraform init
```

```
terraform apply -var-file=terraform.tfvars
```

3. **Authenticate kubectl to the cluster**

Authenticate kubectl to the cluster with the following command. **Remember** to add your Google cloud project ID.
```
gcloud container clusters get-credentials autopilot-cluster --zone us-central1 --project [Project ID]
```

4. **Apply Kubeflow training operator**

Install Kubeflow Training operator v1.7.0:
```
kubectl apply -k ./training-operator/manifests/overlays/standalone
```

## Deleting the platform

In `./cluster` run:

```
terraform destroy
```
