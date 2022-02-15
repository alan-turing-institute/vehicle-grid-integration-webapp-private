# Project infrastructure

Deploy all infrastructure required to host [e4futuregrid](https://www.e4futuregrid.com/). This includes:

1. The VGI API
    - Written with [FastAPI](https://fastapi.tiangolo.com/) framework.
    - Deployed as a docker image to an Azure Web [App Service](https://docs.microsoft.com/en-us/azure/app-service/overview)
  
2. The VGI Frontend
    - A [Vue](https://vuejs.org/v2/guide/) app
    - Deployed as an Azure [Static WebApp](https://docs.microsoft.com/en-us/azure/static-web-apps/getting-started?tabs=vue)


The deployment steps are as follows:

1. Deploy Azure infrastructure.
2. Configure this repository to allow GitHub actions to deploy the API and Frontend.
3. Optionally configure DNS for a custom domain name.

## Step 1. Deploy Azure Infrastructure

### 📦 Requirements

This has grown from [Azure Sensible](https://github.com/alan-turing-institute/azure-sensible)

Before you start, you will need to install:

- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli): Used to authenticate communication with Azure
- [Terraform](https://learn.hashicorp.com/tutorials/terraform/install-cli) >=v1.0: Deploys infrastructure on Azure

### 1. Deployment

#### 🔑 Authentication and Azure subscription
To use terraform to deploy infrastructure on Azure, you will first need to
authenticate using the Azure CLI.

```bash
az login
```

which will launch a browser prompting you to login.

Then you will need to set the subscription you would like to deploy the infrastructure into. This requires an subscription Azure.

To see a list of subscriptions available to you, run:

```bash
az account list --output table
```

Set the subscription you would like to deploy to. The VGI Turing Azure subscription is currently called `Electric Vehicle Grid Integration`.

```bash
az account set --subscription "Electric Vehicle Grid Integration"
```

#### :wrench: Configuration

Set the `/infrastructure` directory as your current working directory:

```bash
cd ./infrastructure
```

Make a copy of the terraform config file. We will use this to configure the deployment.

```
cp terraform/terraform.vgi.tfvars terraform/terraform.tfvars
```

Next you can configure your deployment by editing [`terraform/terraform.tfvars`](terraform/terraform.tfvars).

This file has comments explaining the configuration options and their default values.

The default values given in the file are the same as those used for our production infrastructure. Note that you will need to fill in `networks_data_blobendpoint` yourself, after the blob storage account for the network data has been created. Instructions on how to do so are provided later in this document.

Initialise terraform

```bash
terraform init
```

Plan your changes

```bash
terraform plan
```

this will print a list of changes to your terminal so you can see what terraform
will do.
Run the terraform plan with

```bash
terraform apply
```

### VGI specific manual steps

Apart from the running of terraform, these are the manual steps to do:

- GitHub
  - Navigate to [vehicle-grid-integration-opendss-networks](https://github.com/alan-turing-institute/vehicle-grid-integration-opendss-networks/settings/secrets/actions)
    - Set secret `NETWORKS_DATA_CONTAINER_CONNECTION_STRING` to read/write networks storage container blob endpoint

- Azure
  - Terraform might [automate in the future](https://github.com/hashicorp/terraform-provider-azurerm/issues/8739)
  - For now, download and save the publish profile:
    - In the [Azure Portal](https://portal.azure.com)
    - Select the relevant resource group for this deployment (with default names, `vgiprodwebRG`, otherwise `<website_prefix>RG` based on the contents of `terraform.tfvars`)
    - Open up the Function App (`vgiwebprodfunctionapp`, or `<website_prefix><function_app>`)
    - From the menu on the left, open the Deployment Centre (in the Deployment section)
    - Click on "Manage publish profile" in the top panel
    - Download the publish profile

- GitHub
  - Navigate to [vehicle-grid-integration-webapp-private](https://github.com/alan-turing-institute/vehicle-grid-integration-webapp-private/settings/secrets/actions)
  - Set secret `FUNCTION_APP_PUBLISH_PROFILE` to the contents of the publish profile downloaded in the last step.

- Terraform
  - Run `terraform state show azurerm_static_site.static_site`
  - Copy `api_key` and make a note of the first part of `default_host_name`

- GitHub
  - vehicle-grid-integration-webapp-private
    - Set secret `AZURE_STATIC_WEB_APPS_API_TOKEN_<host-name>`
      - Obtain `<host_name>` from `default_host_name` above - remove the trailing `azurestaticapps.net` and capitalise (e.g. `salmon-forest-09e32d403.azurestaticapps.net` -> `salmon_forest_09E32D403`)
      -❓ Is there a particular reason for leaving the host name inside the secret name? Not as far as I can tell, but the GitHub Action (next point) does need the host name in the filename so we'll keep it this way for consistency.
    - Append the host name to the filename of the GitHub Action for CD of the website (e.g. `azure-static-web-apps-salmon-forest-09e32d403.yml`)

- Terraform
  - Change your terraform.tfvars to include settings:
    - `networks_data_blobendpoint = ` a read-only connection string to the networks storage container

### 💣 Destroy the resources

When you are finished, you can destroy the resources using Terraform. From the
terraform directory run

```
$ terraform destroy
```

This will delete all Azure resources and any data stored on these resources will
be lost.

### Generate SAS tokens

To enable files to be sent to the storage container:
- Generate a SAS token for EON data
    `az storage container generate-sas --account-name datatesteonrestricted --name datatesteonincoming --permissions acdlrw --expiry 2022-01-01`
- Use `azcopy` to transfer the file across to the storage container
    `azcopy copy "test.txt" "https://datatesteonrestricted.blob.core.windows.net/datatesteonincoming/test.txt?<sas>`

### Running the webapp locally

See the guide in `azure_funcs/README.md` in the [WebApp Repo](https://github.com/alan-turing-institute/vehicle-grid-integration-webapp-private).

You will need the following:
- `NETWORKS_DATA_CONTAINER_READONLY_CONNECTION_STRING`
- `NETWORKS_DATA_CONTAINER_READONLY`, which will be `<website_prefix><networks_data_container>`using the variables from your `terraform.tfvars` file (`vgiwebprodopendssnetworks` if using the default/production variable names).

### Refreshing the webapp

After pushing updates to the [WebApp Repo](https://github.com/alan-turing-institute/vehicle-grid-integration-webapp-private), the GitHub Actions should automatically redeploy the webapp and/or backend with your changes.

To see updates to the webapp pages, you may have to refresh your browser cache instead of doing a standard refresh of the page.
On a Mac, this can be done by pressing command + shift + R.
