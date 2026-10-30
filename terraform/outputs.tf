# ==============================================================================
# AutoDeal Copilot — Terraform Outputs (outputs.tf)
# ==============================================================================

output "cloud_run_service_url" {
  description = "HTTPS endpoint URL of the deployed AutoDeal Copilot Cloud Run v2 service."
  value       = google_cloud_run_v2_service.autodeal_copilot.uri
}

output "runtime_service_account_email" {
  description = "Least-privilege runtime Service Account email for AutoDeal Copilot."
  value       = google_service_account.app_sa.email
}

output "cicd_runner_service_account_email" {
  description = "CI/CD pipeline runner Service Account email."
  value       = google_service_account.cicd_runner_sa.email
}

output "gemini_api_key_secret_name" {
  description = "Secret Manager resource name for GOOGLE_API_KEY."
  value       = google_secret_manager_secret.gemini_api_key.name
}

output "firestore_database_id" {
  description = "Firestore native database ID for persistent session state."
  value       = google_firestore_database.session_store.name
}

output "telemetry_bucket_url" {
  description = "GCS bucket URL for evaluation artifacts and telemetry exports."
  value       = google_storage_bucket.telemetry_and_eval_artifacts.url
}
