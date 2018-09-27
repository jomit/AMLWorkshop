# Data Pipeline on Azure

For this excercise we have a simplified data architecture but for more comprehensive guide around [Azure Data Architecture](https://docs.microsoft.com/en-us/azure/architecture/data-guide/), please refer to [Big Data Architecure guide](https://docs.microsoft.com/en-us/azure/architecture/data-guide/big-data/) which covers many topics including Stream/Batch processing, Lamda/Kappa architectures, IoT, Analytics etc. in more details.

####  Training Data Set

- Download : [Turbofan Engine Degradation Simulation Data Set](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan)

- Extract `CMAPSSData.zip` file

#### Login to Azure

- `az login`

If you have multiple subscription than select the one you want to use:

- `az account list`

- `az account set -s <Subscrption ID>`

- `az account show`

#### Create resource group

- `az group create --name <Resource group name> --location eastus2`

#### Create storage account

- `az storage account create --name <Storage account name> --sku Standard_LRS --resource-group <Resource group Name> --location eastus2`

- `az storage account show -n <Storage account name> -g <Resource group Name>`

#### Get storage account keys

- `az storage account keys list -n <Storage account name> -o table`

- `export AZURE_STORAGE_ACCOUNT=<Storage account name>`
- `export AZURE_STORAGE_ACCESS_KEY=<Storage account access key>`

#### Create storage container

- `az storage container create --name dataset --public-access container --account-name <Storage account name> --account-key <Account Key 1>`

#### Upload training dataset

- `az storage blob upload-batch --destination dataset --source 'C:\CMAPSSData' --pattern *.txt --account-name <Storage account name> --account-key <Account Key 1>`

- `az storage blob list --container-name dataset --account-name <Storage account name> --account-key <Account Key 1>`

- Browse the file to verify the upload was successful.
