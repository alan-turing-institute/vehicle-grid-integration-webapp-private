# Project infrastructure

### 📦 Requirements

Before you start, you will need to install some dependencies,

- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
- [Terraform](https://learn.hashicorp.com/tutorials/terraform/install-cli)
- [Ansible](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html)

Not required for VGI: Additionally for generating QR code images to be scanned with an authenticator
app you will need,

- [Python > 3.6](https://wiki.python.org/moin/BeginnersGuide/Download)
- [qrencode](https://fukuchi.org/works/qrencode/) (which you will likely be able
  to find on your distributions repositories or on
  [brew](https://formulae.brew.sh/formula/qrencode))

### 🏞️ Terraform, provisioning your virtual machine

To use terraform to deploy infrastructure on Azure, you will first need to
authenticate using the Azure CLI

```
$ az login
```

which will launch a browser prompting you to login.

Then you will need to enable the subscription you want to deploy the VM into.
Terraform will use your enabled-by-default subscription. The VGI Turing Azure subscription is `Electric Vehicle Grid Integration`

```
$ az account set --subscription <Subscription Name or ID>
```

To see a list of subscriptions available to you, run: `az account list --output table`

To allow for individual customisation and experimentation, you must copy the
default VGI variables file to your own instance.

```
cp terraform/terraform.vgi.tfvars terraform/terraform.tfvars
```

Next you can configure your deployment by editing
[`terraform/terraform.tfvars`](terraform/terraform.tfvars). This file has
comments explaining the configuration options and their default values.

Initialise terraform

```
$ cd terraform
$ terraform init
```

Plan your changes

```
$ terraform plan
```

this will print a list of changes to your terminal so you can see what terraform
will do. Run the terraform plan with

```
$ terraform apply
```

### 💣 Destroy the resources

When you are finished, you can destroy the resources using Terraform. From the
terraform directory run

```
$ terraform destroy
```

This will delete all Azure resources and any data stored on these resources will
be lost.

### Notes

Currently the contents of `ansible` `examples` and `scripts` are unused.

To enable files to be sent to the storage container:
- Generate a SAS token
    ```az storage container generate-sas --account-name <ACCOUNT_NAME> --name <CONTAINER_NAME> --permissions acdlrw --expiry <DATE>```
- Generate a URL for the storage container
    ```az storage blob url --account-name <ACCOUNT_NAME> --container-name <CONTAINER_NAME> --name <DATE_NAME> --sas-token="<SAS_TOKEN>"```
- Use `azcopy` to transfer the file across to the storage container
    ```azcopy copy "<FILE_NAME>" "<URL>"```
