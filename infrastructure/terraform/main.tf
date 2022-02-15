# Configure the Microsoft Azure Provider
terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = ">= 2.26"
    }
  }
}

# Declare Azure provider
provider "azurerm" {
  features {}
}

# Generate a random integer to create a globally unique name
resource "random_integer" "ri" {
  min = 10000
  max = 99999
}

# Create uniquely named resource group to deploy API and Frontend in
resource "azurerm_resource_group" "website_rg" {
  name     = "${var.website_prefix}RG-${random_integer.ri.result}"
  location = var.website_location
}


# Create networks storage account
resource "azurerm_storage_account" "networks_data_account" {
  name                     = "${var.website_prefix}${var.networks_data_account}${random_integer.ri.result}"
  resource_group_name      = azurerm_resource_group.website_rg.name
  location                 = var.website_location
  account_tier             = "Premium"
  account_kind             = "BlockBlobStorage"
  account_replication_type = "LRS"
}

# Create networks storage container
resource "azurerm_storage_container" "networks_data_container" {
  name                  = "${var.website_prefix}${var.networks_data_container}-${random_integer.ri.result}"
  storage_account_name  = azurerm_storage_account.networks_data_account.name
  container_access_type = "private"
}


data "azurerm_storage_account_blob_container_sas" "example" {
  connection_string = azurerm_storage_account.networks_data_account.primary_connection_string
  container_name    = azurerm_storage_container.networks_data_container.name
  https_only        = true

  start = "2022-01-01"
  expiry = "2030-01-01"
  permissions {
    read   = true
    add    = true
    create = false
    write  = false
    delete = true
    list   = true
  }

  cache_control       = "max-age=5"
  content_disposition = "inline"
  content_encoding    = "deflate"
  content_language    = "en-US"
  content_type        = "application/json"
}


# Create an app service plan - Hosts the API
resource "azurerm_app_service_plan" "appserviceplan" {
  name                = "${var.website_prefix}-ASP-${random_integer.ri.result}"
  location            = azurerm_resource_group.website_rg.location
  resource_group_name = azurerm_resource_group.website_rg.name
  kind                = "linux"
  reserved            = true
  sku {
    tier = "Standard"
    size = "P2V2"
  }
}

# Create a WebApp in the App service
# Azure container registry to push docker image to
resource "azurerm_container_registry" "acr" {
  name                = "${var.website_prefix}acr${random_integer.ri.result}"
  resource_group_name = azurerm_resource_group.website_rg.name
  location            = azurerm_resource_group.website_rg.location
  sku                 = "Basic"
  admin_enabled       = true
}

# WebApp to host API
resource "azurerm_app_service" "webapp" {
  name                = "${var.website_prefix}-webapp-${random_integer.ri.result}"
  location            = azurerm_resource_group.website_rg.location
  resource_group_name = azurerm_resource_group.website_rg.name
  app_service_plan_id = azurerm_app_service_plan.appserviceplan.id
  https_only          = true

  app_settings = {
    WEBSITES_ENABLE_APP_SERVICE_STORAGE = false
    # Settings for private Container Registires  
    DOCKER_REGISTRY_SERVER_URL      = "https://${azurerm_container_registry.acr.login_server}"
    DOCKER_REGISTRY_SERVER_USERNAME = azurerm_container_registry.acr.admin_username
    DOCKER_REGISTRY_SERVER_PASSWORD = azurerm_container_registry.acr.admin_password

    DOCKER_ENABLE_CI = true

    # Environment vars for Docker image
    APP_MODULE = "vgi_api:app"
    GRACEFUL_TIMEOUT = 300

    NETWORKS_DATA_CONTAINER_READONLY = "${var.website_prefix}${var.networks_data_account}${random_integer.ri.result}"
    NETWORKS_DATA_CONTAINER_READONLY_CONNECTION_STRING = data.azurerm_storage_account_blob_container_sas.example.sas
  }

  # Configure Docker Image to load on start
  site_config {
    linux_fx_version = "DOCKER|${var.website_prefix}acr${random_integer.ri.result}.azurecr.io/vgi_api:latest"
    always_on        = "true"
  }
}

# Create app service plan and its prerequsites
# resource "azurerm_app_service_plan" "app_service_plan" {
#   name                = "${var.website_prefix}${var.app_service_plan}"
#   resource_group_name = azurerm_resource_group.website_rg.name
#   location            = var.website_location
#   kind                = "elastic"
#   reserved            = true # same as az functionapp --is-linux true
#   sku {
#     tier = "ElasticPremium"
#     size = "EP2"
#   }
# }

# resource "azurerm_storage_account" "app_data_account" {
#   name                     = "${var.website_prefix}${var.app_data_account}"
#   resource_group_name      = azurerm_resource_group.website_rg.name
#   location                 = var.website_location
#   account_tier             = "Standard"
#   account_replication_type = "LRS"
# }

# resource "azurerm_function_app" "function_app" {
#   name                       = "${var.website_prefix}${var.function_app}"
#   resource_group_name        = azurerm_resource_group.website_rg.name
#   location                   = var.website_location
#   app_service_plan_id        = azurerm_app_service_plan.app_service_plan.id
#   storage_account_name       = azurerm_storage_account.app_data_account.name
#   storage_account_access_key = azurerm_storage_account.app_data_account.primary_access_key
#   os_type                    = "linux"
#   version                    = "~3"
#   app_settings = {
#     "BUILD_FLAGS"                                        = "UseExpressBuild"
#     "ENABLE_ORYX_BUILD"                                  = true
#     "FUNCTIONS_WORKER_RUNTIME"                           = "python"
#     "SCM_DO_BUILD_DURING_DEPLOYMENT"                     = 1
#     "NETWORKS_DATA_CONTAINER_READONLY"                   = azurerm_storage_container.networks_data_container.name
#     "NETWORKS_DATA_CONTAINER_READONLY_CONNECTION_STRING" = "${var.networks_data_blobendpoint}"
#   }
#   site_config {
#     http2_enabled             = true
#     linux_fx_version          = "PYTHON|3.8"
#     use_32_bit_worker_process = false
#     pre_warmed_instance_count = 1
#     cors {
#       allowed_origins = ["*"]
#     }
#   }
# }

resource "azurerm_static_site" "static_site" {
  name                = "${var.website_prefix}${var.static_site_name}"
  resource_group_name = azurerm_resource_group.website_rg.name
  location            = var.website_location
}
