# ==============================================================================
# AutoDeal Copilot — Least-Privilege IAM & Service Accounts (iam.tf)
# ==============================================================================

resource "google_service_account" "app_sa" {
  project      = var.project_id
  account_id   = "${var.service_name}-runtime-sa"
  display_name = "AutoDeal Copilot Cloud Run Runtime Service Account"
  description  = "Least-privilege identity for AutoDeal Copilot ADK multi-agent runtime."
}

resource "google_service_account" "cicd_runner_sa" {
  project      = var.project_id
  account_id   = "${var.service_name}-cicd-sa"
  display_name = "AutoDeal Copilot CI/CD & Evaluation Runner Service Account"
  description  = "Service account for GitHub Actions / Cloud Build automated evaluation and deployment."
}

# Grant Secret Manager Secret Accessor ONLY on the specific Gemini API key secret
resource "google_secret_manager_secret_iam_member" "app_sa_secret_accessor" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.gemini_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.app_sa.email}"
}

locals {
  app_runtime_roles = toset([
    "roles/aiplatform.user",
    "roles/datastore.user",
    "roles/cloudtrace.agent",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/dlp.user",
  ])

  cicd_runner_roles = toset([
    "roles/run.admin",
    "roles/artifactregistry.writer",
    "roles/iam.serviceAccountUser",
    "roles/cloudbuild.builds.editor",
  ])
}

resource "google_project_iam_member" "app_runtime_permissions" {
  for_each = local.app_runtime_roles
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.app_sa.email}"
}

resource "google_project_iam_member" "cicd_runner_permissions" {
  for_each = local.cicd_runner_roles
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.cicd_runner_sa.email}"
}

resource "google_storage_bucket_iam_member" "app_sa_telemetry_writer" {
  bucket = google_storage_bucket.telemetry_and_eval_artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.app_sa.email}"
}
