# ==============================================================================
# AutoDeal Copilot — Terraform Input Variables (variables.tf)
# ==============================================================================

variable "project_id" {
  description = "Google Cloud Project ID hosting AutoDeal Copilot."
  type        = string
  default     = "autodeal-copilot-gcp"
}

variable "region" {
  description = "Primary Google Cloud region for Cloud Run, Firestore, and Vertex AI."
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Cloud Run v2 service name for AutoDeal Copilot."
  type        = string
  default     = "autodeal-copilot"
}

variable "container_image" {
  description = "Artifact Registry container image URI for AutoDeal Copilot."
  type        = string
  default     = "us-central1-docker.pkg.dev/autodeal-copilot-gcp/autodeal/autodeal-copilot:latest"
}

variable "fast_model" {
  description = "3rd-generation Gemini Flash model for high-throughput lookup and orchestration."
  type        = string
  default     = "gemini-3-flash-preview"
}

variable "pro_model" {
  description = "3rd-generation Gemini Pro model for multi-constraint Top-3 offer ranking."
  type        = string
  default     = "gemini-3.1-pro-preview"
}

variable "gemini_api_key_secret_id" {
  description = "Secret Manager secret identifier storing the Gemini / Google API key."
  type        = string
  default     = "GOOGLE_API_KEY"
}

variable "firestore_database_name" {
  description = "Firestore database name for persistent customer sessions and inventory catalog."
  type        = string
  default     = "autodeal-sessions"
}

variable "min_instances" {
  description = "Minimum warm Cloud Run container instances."
  type        = number
  default     = 1
}

variable "max_instances" {
  description = "Maximum autoscaled Cloud Run container instances."
  type        = number
  default     = 10
}

variable "notification_channels" {
  description = "Optional list of Cloud Monitoring notification channel resource names."
  type        = list(string)
  default     = []
}
