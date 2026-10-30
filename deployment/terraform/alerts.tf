# ==============================================================================
# AutoDeal Copilot — Cloud Monitoring SRE PromQL Alert Policies (alerts.tf)
# Configures 5xx Error Rate and P95 Request Latency alerts for Cloud Run.
# ==============================================================================

resource "google_monitoring_alert_policy" "cloud_run_5xx_error_rate" {
  project      = var.project_id
  display_name = "${var.service_name} — High 5xx Server Error Rate (> 2%)"
  combiner     = "OR"

  user_labels = {
    service                   = var.service_name
    severity                  = "critical"
    created-with-google-skill = "cloud-run-alert-configuration"
  }

  conditions {
    display_name = "Cloud Run 5xx Error Ratio > 2% over 5m"

    condition_prometheus_query_language {
      duration            = "300s"
      evaluation_interval = "60s"
      query               = <<-EOT
        (
          sum(rate(run_googleapis_com:request_count{service_name="${var.service_name}", response_code_class="5xx"}[5m]))
          /
          clamp_min(sum(rate(run_googleapis_com:request_count{service_name="${var.service_name}"}[5m])), 0.001)
        ) > 0.02
      EOT
    }
  }

  alert_strategy {
    auto_close = "1800s"
  }

  notification_channels = var.notification_channels
  depends_on            = [google_project_service.enabled_apis]
}

resource "google_monitoring_alert_policy" "cloud_run_p95_latency" {
  project      = var.project_id
  display_name = "${var.service_name} — High P95 Request Latency (> 8000ms)"
  combiner     = "OR"

  user_labels = {
    service                   = var.service_name
    severity                  = "warning"
    created-with-google-skill = "cloud-run-alert-configuration"
  }

  conditions {
    display_name = "Cloud Run P95 Request Latency > 8000ms over 5m"

    condition_prometheus_query_language {
      duration            = "300s"
      evaluation_interval = "60s"
      query               = <<-EOT
        histogram_quantile(
          0.95,
          sum by (le) (
            rate(run_googleapis_com:request_latencies_bucket{service_name="${var.service_name}"}[5m])
          )
        ) > 8000
      EOT
    }
  }

  alert_strategy {
    auto_close = "1800s"
  }

  notification_channels = var.notification_channels
  depends_on            = [google_project_service.enabled_apis]
}
