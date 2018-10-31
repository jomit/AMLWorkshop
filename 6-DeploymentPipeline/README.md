# Deploy to AKS

- See [Deploy-AKS.ipynb](https://github.com/jomit/AMLWorkshop/blob/master/6-DeploymentPipeline/Deploy-AKS.ipynb) to deploy the model on an AKS Cluster

- See [Deploy_ACI.ipynb](https://github.com/jomit/AMLWorkshop/blob/master/6-DeploymentPipeline/Deploy_ACI.ipynb) to deploy the model via ACI

### Access AKS Dashboard

- Using Azure Cloud Shell

    - `az aks browse --resource-group <resource-group-name> --name <aks-name> --enable-cloud-console-aks-browse`

- Using Local Machine

    - `az login`

    - `az aks install-cli`

    - `az aks browse --resource-group <resource-group-name> --name <aks-name>`