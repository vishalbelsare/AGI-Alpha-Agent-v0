# Example Google Cloud Run deployment

variable "project" { default = "my-project" }
variable "region" { default = "us-central1" }
variable "image_tag" { default = "latest" }
variable "repository_name" { default = "insight" }
variable "vpc_connector" { default = "" }

provider "google" {
  project = var.project
  region  = var.region
}

resource "google_artifact_registry_repository" "insight" {
  location      = var.region
  repository_id = var.repository_name
  format        = "DOCKER"
}

resource "google_cloud_run_service" "insight" {
  name     = "alpha-insight"
  location = var.region

  template {
    spec {
      containers {
        image = "${google_artifact_registry_repository.insight.repository_url}:${var.image_tag}"
        env = [{ name = "RUN_MODE", value = "api" }]
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
