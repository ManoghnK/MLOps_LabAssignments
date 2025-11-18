provider "google" {
    project = "manoghn-terraform-lab" # Replace with your project ID
    region  = "us-central1"
    zone    = "us-central1-a"
}

resource "google_compute_instance" "vm_instance" {
    name         = "terraform-vm"
    machine_type = "f1-micro"
    zone         = "us-central1-a"

    boot_disk {
        initialize_params {
            image = "debian-cloud/debian-11"
        }
    }

    network_interface {
        network = "default"
    }
}