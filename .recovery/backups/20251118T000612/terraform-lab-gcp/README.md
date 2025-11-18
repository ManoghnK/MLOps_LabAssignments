
---

# 🌍 Terraform Lab — GCP Lab 1 (Beginner)

**Original Lab:**
[https://github.com/raminmohammadi/MLOps/tree/main/Labs/Terraform_Labs/GCP/Lab1_Beginner](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Terraform_Labs/GCP/Lab1_Beginner)

This README consolidates the original lab requirements *plus* the enhancements added in this submission:

* Modified VM configuration
* Added a Cloud Storage bucket
* Added an ML training script (`train.py`)
* Uploaded a trained model (`iris_model.pkl`) to GCS
* Included verification screenshots

---

## 🎯 Objective

This lab introduces Terraform fundamentals by creating, modifying, and destroying Google Cloud resources.
You will learn how to:

* Configure Terraform for Google Cloud
* Create infrastructure (VM + GCS bucket) using IaC
* Modify and re-apply configuration
* Destroy Terraform-managed resources
* **(Changes)** Run an Iris ML training script and upload the model artifact to GCS

---

## 📦 Files Included

| File                            | Description                                      |
| ------------------------------- | ------------------------------------------------ |
| `main.tf`                       | Terraform configuration (provider, VM, bucket)   |
| `terraform.tfstate` / `.backup` | Terraform state files                            |
| `train.py`                      | ML training script (Iris dataset → RandomForest) |
| `iris_model.pkl`                | Trained model artifact uploaded to GCS           |
| `*.png`                         | Screenshots documenting each lab step            |

---

## 🆕 Enhancements Added

### ✔ VM Improvements

* Changed machine type → `e2-micro`
* Added labels for organization
* Increased boot disk size to 12GB

### ✔ Storage Bucket

Created via Terraform using:

```hcl
resource "google_storage_bucket" "terraform_lab_bucket" {
  name          = "terraform-lab-bucket-478605"
  location      = "us-central1"
  force_destroy = true
}
```

### ✔ ML Component

* `train.py` trains an Iris model
* Saves `iris_model.pkl`
* Uploads artifact to the bucket created by Terraform

### ✔ Screenshots

All screenshots included to demonstrate:

* VM creation
* Bucket creation
* Model training
* Model upload
* Resource destruction

---

## 🛠️ Prerequisites

* Google Cloud project with billing enabled
* Google Cloud SDK (`gcloud`) installed + authenticated
* Terraform v1.x installed
* Service account with roles:

  * Compute Admin
  * Storage Admin
  * Service Account User

Optional authentication:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
```

---

## 🚀 Part 1 — Terraform Setup

### 1. Check Terraform installation

```bash
terraform --version
```

### 2. Initialize Terraform

```bash
terraform init
```

### 3. Review `main.tf`

```hcl
provider "google" {
  project = "terraform-lab-478605"
  region  = "us-central1"
  zone    = "us-central1-a"
}

resource "google_compute_instance" "vm_instance" {
  name         = "terraform-vm"
  machine_type = "e2-micro"
  zone         = "us-central1-a"

  labels = {
    environment = "development"
    owner       = "team-terraform"
  }

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
      size  = 12
    }
  }

  network_interface {
    network = "default"
    access_config {}
  }
}

resource "google_storage_bucket" "terraform_lab_bucket" {
  name          = "terraform-lab-bucket-478605"
  location      = "us-central1"
  force_destroy = true
}
```

---

## 🟦 Part 2 — Create Infrastructure

### Preview:

```bash
terraform plan -out=tfplan
```

### Apply:

```bash
terraform apply tfplan
```

Expected output:

```
Apply complete! Resources: 2 added, 0 changed, 0 destroyed.
```


---

## 🟧 Part 3 — Modify Resources

Update CPU type, labels, disk size, etc.
Changing Machibe Type to e2
![Changing Machibe Type to e2](./Adding_Configurations_machine.png)
Changing Storage size to 12GB
![Changing Storage size to 12GB](./Adding_configurations_storage.png)

```bash
terraform apply -auto-approve
```

---

## 🟩 Part 4 — Add Cloud Storage Bucket

Bucket creation screenshots:
Before Bucket storage
![Before Bucket storage](./Before_Adding_Storage.png)
After Bucket storage
![After Bucket storage](./After_Adding_Storage.png)

---

## 🧪 Part 5 — ML Training & Model Upload (New)

`train.py` trains a RandomForest model on the Iris dataset and uploads `iris_model.pkl` to your bucket.

### Run Locally:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install scikit-learn pandas joblib google-cloud-storage
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
python3 train.py
```

Expected output:

```
Model accuracy: 0.9+
Model saved as iris_model.pkl
Model uploaded to GCS bucket successfully!
```

Screenshots:
Generated model pkl:
![PKL generated](./pkl_generated.png)
CPU Usage:
![CPU usage](./cpu_utils.png)

---

## 🧨 Part 6 — Destroy All Terraform Resources

```bash
terraform destroy
```

Confirm:

```
yes
```

Destruction screenshot:

![Destroying resources](./Destroying_resources.png)

---

## 🗂 Understanding Terraform Files

* **`terraform.tfstate`** — tracks all managed resources
* **`.terraform/`** — provider plugins & metadata
* Never edit state files manually

---

## 🧑‍💻 Author

Name: Manoghn Kandiraju

---

## 📚 Attribution

Adapted from: **raminmohammadi/MLOps — Terraform Beginner Lab (GCP)**

