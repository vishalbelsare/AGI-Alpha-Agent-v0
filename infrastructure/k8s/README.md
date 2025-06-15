This directory contains Kubernetes manifests for running background jobs.

`spot-gpu-cronjob.yaml` defines a CronJob that periodically dequeues work for
GPU spot instances. The worker container runs `evolution-worker:latest` with one
GPU request. Modify the schedule or image as needed for your cluster.
