# Simplified Google Cloud Run deployment
variable "project" { default = "my-project" }
variable "openai_api_key" { default = "" }

provider "google" { project = var.project }

resource "google_cloud_run_service" "af" {
  name     = "alpha-factory"
  location = "us-central1"

  template {
    spec {
      containers {
        image = "alpha-demo:latest"
        env = [
          {
            name  = "OPENAI_API_KEY"
            value = var.openai_api_key
          },
          {
            name  = "RUN_MODE"
            value = "api"
          }
        ]
        ports { container_port = 8000 }
      }
    }
  }
}
