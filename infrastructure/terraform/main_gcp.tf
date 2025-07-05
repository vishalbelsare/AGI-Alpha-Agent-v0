# Simplified Google Cloud Run deployment
variable "project" { default = "my-project" }
variable "openai_api_key" { default = "" }
variable "openai_api_key_secret_id" { default = "" }
variable "agi_insight_offline" { default = "0" }
variable "agi_insight_bus_port" { default = 6006 }
variable "agi_insight_ledger_path" { default = "./ledger/audit.db" }
variable "container_image" { default = "alpha-demo:latest" }
variable "vpc_connector" { default = "" }

provider "google" { project = var.project }

resource "google_cloud_run_service" "af" {
  name     = "alpha-factory"
  location = "us-central1"

  template {
    spec {
      containers {
        image = var.container_image
        env = concat([
          {
            name  = "RUN_MODE"
            value = "api"
          },
          {
            name  = "AGI_INSIGHT_OFFLINE"
            value = var.agi_insight_offline
          },
          {
            name  = "AGI_INSIGHT_BUS_PORT"
            value = tostring(var.agi_insight_bus_port)
          },
          {
            name  = "AGI_INSIGHT_LEDGER_PATH"
            value = var.agi_insight_ledger_path
          }
        ],
        var.openai_api_key_secret_id == "" ? [
          {
            name  = "OPENAI_API_KEY"
            value = var.openai_api_key
          }
        ] : [
          {
            name  = "AGI_INSIGHT_SECRET_BACKEND"
            value = "gcp"
          },
          {
            name  = "GCP_PROJECT_ID"
            value = var.project
          },
          {
            name  = "OPENAI_API_KEY_SECRET_ID"
            value = var.openai_api_key_secret_id
          }
        ])
        ports { container_port = 8000 }
        ports { container_port = 6006 }
      }
      dynamic "vpc_access" {
        for_each = var.vpc_connector == "" ? [] : [var.vpc_connector]
        content {
          connector = vpc_access.value
        }
      }
    }
  }
}
