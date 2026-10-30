# ==============================================================================
# AutoDeal Copilot — Production Terraform Infrastructure as Code (main.tf)
# Provisions Google Cloud APIs, Secret Manager, Firestore, GCS Telemetry Storage,
# and Cloud Run v2 Agent Runtime (`google_cloud_run_v2_service.autodeal_copilot`).
# Generated in alignment with Agent Starter Pack (`agents-cli`) & ADK Cloud Run.
# ==============================================================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.30.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  required_services = toset([
    "run.googleapis.com",
    "aiplatform.googleapis.com",
    "secretmanager.googleapis.com",
    "firestore.googleapis.com",
    "cloudtrace.googleapis.com",
    "dlp.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
  ])
}

resource "google_project_service" "enabled_apis" {
  for_each           = local.required_services
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# ------------------------------------------------------------------------------
# Secret Manager — Zero Hardcoded API Keys (`GOOGLE_API_KEY`)
# ------------------------------------------------------------------------------
resource "google_secret_manager_secret" "gemini_api_key" {
  project   = var.project_id
  secret_id = var.gemini_api_key_secret_id

  labels = {
    app        = var.service_name
    managed_by = "terraform"
  }

  replication {
    auto {}
  }

  depends_on = [google_project_service.enabled_apis]
}

# ------------------------------------------------------------------------------
# Persistent Session & Vehicle Catalog Store — Google Cloud Firestore
# ------------------------------------------------------------------------------
resource "google_firestore_database" "session_store" {
  project                     = var.project_id
  name                        = var.firestore_database_name
  location_id                 = var.region
  type                        = "FIRESTORE_NATIVE"
  concurrency_mode            = "OPTIMISTIC"
  app_engine_integration_mode = "DISABLED"

  depends_on = [google_project_service.enabled_apis]
}

# ------------------------------------------------------------------------------
# Telemetry, Evaluation Artifacts & Context Cache Staging Bucket
# ------------------------------------------------------------------------------
resource "google_storage_bucket" "telemetry_and_eval_artifacts" {
  name                        = "${var.project_id}-${var.service_name}-telemetry"
  project                     = var.project_id
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.enabled_apis]
}

# ------------------------------------------------------------------------------
# Cloud Run v2 Service — AutoDeal Copilot Multi-Agent ADK Runtime
# ------------------------------------------------------------------------------
resource "google_cloud_run_v2_service" "autodeal_copilot" {
  name     = var.service_name
  location = var.region
  project  = var.project_id
  ingress  = "INGRESS_TRAFFIC_ALL"

  labels = {
    app        = var.service_name
    framework  = "google-adk"
    managed_by = "terraform"
  }

  template {
    service_account = google_service_account.app_sa.email
    timeout         = "300s"

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    containers {
      image = var.container_image

      ports {
        name           = "http1"
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
        cpu_idle = true
      }

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }

      env {
        name  = "GOOGLE_CLOUD_LOCATION"
        value = var.region
      }

      env {
        name  = "AUTODEAL_FAST_MODEL"
        value = var.fast_model
      }

      env {
        name  = "AUTODEAL_PRO_MODEL"
        value = var.pro_model
      }

      env {
        name  = "OTEL_TRACES_EXPORTER"
        value = "google_cloud_trace"
      }

      env {
        name  = "OTEL_SERVICE_NAME"
        value = var.service_name
      }

      env {
        name = "GOOGLE_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.gemini_api_key.secret_id
            version = "latest"
          }
        }
      }

      startup_probe {
        http_get {
          path = "/healthz"
          port = 8080
        }
        initial_delay_seconds = 5
        period_seconds        = 10
        failure_threshold     = 3
      }

      liveness_probe {
        http_get {
          path = "/healthz"
          port = 8080
        }
        period_seconds    = 30
        failure_threshold = 3
      }
    }
  }

  depends_on = [
    google_project_service.enabled_apis,
    google_secret_manager_secret_iam_member.app_sa_secret_accessor,
  ]
}
