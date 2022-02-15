
// API and Frontend Infrastructure
variable "website_location" {
  type        = string
  default     = "westeurope"
  description = "The Azure region to deploy the API and frontend. See https://azure.microsoft.com/en-gb/global-infrastructure/geographies/#choose-your-region"
}

variable "website_prefix" {
  type        = string
  nullable    = false
  description = "A prefix added to the name of all Azure resources"
}

// Network model storage
variable "networks_data_account" {
  type        = string
  default     = "websitedata"
  description = "Suffix to add to name of storage account used to store network models"
}

variable "networks_data_container" {
  type        = string
  default     = "opendssnetworks"
  description = "Suffix to add to name of storage container used to store network models"
}


# variable "networks_data_blobendpoint" {
#   type    = string
#   default = "changeme"
# }


// Static website config
variable "static_site_name" {
  type        = string
  default     = "e4future"
  description = "name of static website"
}


// Variables related to deploying storage for incoming data. Not required to deploy the API or frontend
/*
 variable "incoming_data_location" {
  type    = string
  default = "uksouth"
}

variable "incoming_data_prefix" {
  type    = string
  default = "changeme"
}

variable "incoming_data_account" {
  type    = string
  default = "eonrestricted"
}

variable "incoming_data_container" {
  type    = string
  default = "eonincoming"
}
*/