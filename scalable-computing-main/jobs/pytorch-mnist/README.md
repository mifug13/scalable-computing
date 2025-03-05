# pytorch-mnist

## Prerequisites

* The cluster is running
* The Kubeflow training-operator has been deployed to the cluster
* Kubectl has been authenticated to the cluster

Tools:
* python
* make
* docker
* gcloud
* kubectl

## Getting started

1. Upload dataset

```
make upload-dataset PROJECT_ID=[Project ID]
```

2. Build and push docker image

```
make build-and-push PROJECT_ID=[Project ID]
```

3. Deploy workload

```
kubectl apply -f ./pytorch-mnist.yaml
```

4. Profit

:)