variable "project_id" {
  description = "Your Google Cloud project ID"
  type        = string
}

variable "region" {
  description = "The Google Cloud region for the bucket"
  type        = string
  default     = "us-central1"
}