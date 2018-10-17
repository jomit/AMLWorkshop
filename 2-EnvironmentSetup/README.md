# Machine Learning Envionment Setup

### set up DSVM

### method 1: set up in the Portal

1. create new VM, using image: Data Science Virtual Machine for Linux (Ubuntu)
2. choose size, set username/password
3. set up DNS <amlworkshop>
4. connect using browser: https://amlworkshop.westus.cloudapp.azure.com:8000

### method 2: set up using command line
- 'az vm create \
    --name $VMNAME --resource-group $RGNAME --image $PUB\:$OFFER\:$SKU\:$VERSION \
    --plan-name $SKU --plan-product $OFFER --plan-publisher $PUB \
    --admin-username $USER --admin-password $PASS \
    --size $SIZE \
    --data-disk-sizes-gb $DATASIZEGB'

### verify you installation
1. connect using browser: https://amlworkshop.westus.cloudapp.azure.com:8000
2. create a new Jupyter Notebook and test this command: 
    import azureml
    print("Azure ML: ", azureml.core.VERSION)
3. If you see something like: "Azure ML: 0.1.59", congratulations!!! you are all set.
