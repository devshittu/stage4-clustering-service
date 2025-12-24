#!/usr/bin/env python3
"""
Stage 4 Clustering Service CLI

Command-line interface for interacting with the clustering service.

Usage:
    python cli.py health                          # Check service health
    python cli.py stats                           # Get statistics
    python cli.py cluster-events                  # Cluster events
    python cli.py cluster-documents               # Cluster documents
    python cli.py cluster-entities                # Cluster entities
    python cli.py cluster-storylines              # Cluster storylines
    python cli.py job-status <job_id>             # Check job status
    python cli.py job-list                        # List all jobs
    python cli.py job-cancel <job_id>             # Cancel a job
    python cli.py clusters [--type event]         # List clusters
    python cli.py cluster-info <cluster_id>       # Get cluster details
    python cli.py resources                       # Get resource stats
"""

import sys
import json
import requests
import time
from typing import Optional
import argparse
from datetime import datetime


# Configuration
API_BASE_URL = "http://localhost/api/v1/clustering"
TIMEOUT = 30


class ClusteringCLI:
    """CLI for Stage 4 Clustering Service."""

    def __init__(self, base_url: str = API_BASE_URL):
        """
        Initialize CLI.

        Args:
            base_url: Base URL for API
        """
        self.base_url = base_url.rstrip("/")

    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> dict:
        """
        Make HTTP request to API.

        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request body (for POST/PATCH)
            params: Query parameters

        Returns:
            Response JSON

        Raises:
            SystemExit: On request failure
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            response = requests.request(
                method=method,
                url=url,
                json=data,
                params=params,
                timeout=TIMEOUT,
            )

            if response.status_code >= 400:
                print(f"❌ Error {response.status_code}: {response.text}", file=sys.stderr)
                sys.exit(1)

            return response.json()

        except requests.ConnectionError:
            print(f"❌ Connection failed. Is the service running at {self.base_url}?", file=sys.stderr)
            sys.exit(1)
        except requests.Timeout:
            print(f"❌ Request timeout after {TIMEOUT}s", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"❌ Request failed: {e}", file=sys.stderr)
            sys.exit(1)

    def health(self) -> dict:
        """Check service health."""
        return self._request("GET", "/health")

    def statistics(self) -> dict:
        """Get clustering statistics."""
        return self._request("GET", "/statistics")

    def resources(self) -> dict:
        """Get resource utilization."""
        return self._request("GET", "/resources")

    def cluster(
        self,
        embedding_type: str,
        algorithm: str = "hdbscan",
        params: Optional[dict] = None,
        filters: Optional[dict] = None,
        enable_temporal: bool = True,
    ) -> dict:
        """
        Submit clustering job.

        Args:
            embedding_type: Type of embeddings (document/event/entity/storyline)
            algorithm: Algorithm to use (hdbscan/kmeans/agglomerative)
            params: Algorithm parameters
            filters: Metadata filters
            enable_temporal: Enable temporal clustering

        Returns:
            Job response
        """
        request_data = {
            "embedding_type": embedding_type,
            "algorithm": algorithm,
            "algorithm_params": params or {},
            "filters": filters,
            "enable_temporal_clustering": enable_temporal,
            "checkpoint_interval": 10,
        }

        return self._request("POST", "/batch", data=request_data)

    def job_status(self, job_id: str) -> dict:
        """Get job status."""
        return self._request("GET", f"/jobs/{job_id}")

    def job_list(self, status: Optional[str] = None, limit: int = 100) -> dict:
        """List jobs."""
        params = {"limit": limit}
        if status:
            params["status"] = status

        return self._request("GET", "/jobs", params=params)

    def job_action(self, job_id: str, action: str) -> dict:
        """
        Perform action on job.

        Args:
            job_id: Job identifier
            action: Action (pause/resume/cancel)

        Returns:
            Action response
        """
        return self._request("PATCH", f"/jobs/{job_id}", data={"action": action})

    def job_cancel(self, job_id: str) -> dict:
        """Cancel job."""
        return self._request("DELETE", f"/jobs/{job_id}")

    def list_clusters(
        self,
        embedding_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """List clusters."""
        params = {"limit": limit, "offset": offset}
        if embedding_type:
            params["embedding_type"] = embedding_type

        return self._request("GET", "/clusters", params=params)

    def cluster_info(self, cluster_id: str) -> dict:
        """Get cluster details."""
        return self._request("GET", f"/clusters/{cluster_id}")

    def search_clusters(self, query: dict) -> dict:
        """Search clusters."""
        return self._request("POST", "/clusters/search", data=query)


def print_json(data: dict, indent: int = 2):
    """Pretty print JSON."""
    print(json.dumps(data, indent=indent, default=str))


def print_health(health: dict):
    """Print health check results."""
    status_icon = "✅" if health["status"] == "healthy" else "❌"
    print(f"{status_icon} Service Status: {health['status'].upper()}")
    print(f"   Version: {health.get('version', 'unknown')}")
    print(f"   Redis: {'✅' if health.get('redis_connected') else '❌'}")
    print(f"   PostgreSQL: {'✅' if health.get('postgresql_connected') else '❌'}")
    print(f"   Stage 3: {'✅' if health.get('stage3_available') else '❌'}")
    print(f"   FAISS Loaded: {'✅' if health.get('faiss_loaded') else '❌'}")
    print(f"   Active Jobs: {health.get('active_jobs', 0)}")


def print_stats(stats: dict):
    """Print statistics."""
    print("📊 Clustering Statistics\n")
    print(f"Total Clusters: {stats.get('total_clusters', 0)}")
    print(f"Total Jobs: {stats.get('total_jobs', 0)}")
    print(f"Avg Cluster Size: {stats.get('avg_cluster_size', 0.0):.1f}")
    print(f"Total Outliers: {stats.get('total_outliers', 0)}")

    print("\nClusters by Type:")
    for etype, count in stats.get("clusters_by_type", {}).items():
        print(f"  {etype}: {count}")

    print("\nJobs by Status:")
    for status, count in stats.get("jobs_by_status", {}).items():
        print(f"  {status}: {count}")


def print_job(job: dict):
    """Print job information."""
    status_icons = {
        "queued": "⏳",
        "running": "🔄",
        "paused": "⏸️",
        "completed": "✅",
        "failed": "❌",
        "canceled": "🚫",
    }

    status = job.get("status", "unknown")
    icon = status_icons.get(status, "❓")

    print(f"\n{icon} Job: {job.get('job_id', 'unknown')}")
    print(f"   Status: {status.upper()}")
    print(f"   Embedding Type: {job.get('embedding_type', 'unknown')}")
    print(f"   Algorithm: {job.get('algorithm', 'unknown')}")
    print(f"   Progress: {job.get('progress_percent', 0.0):.1f}%")
    print(f"   Processed: {job.get('processed_items', 0)} / {job.get('total_items', 0)}")
    print(f"   Clusters: {job.get('clusters_created', 0)}")
    print(f"   Outliers: {job.get('outliers', 0)}")
    print(f"   Created: {job.get('created_at', 'unknown')}")

    if job.get("error_message"):
        print(f"   ❌ Error: {job['error_message']}")


def print_resources(resources: dict):
    """Print resource utilization."""
    print("🖥️  Resource Utilization\n")

    cpu = resources.get("cpu", {})
    print(f"CPU: {cpu.get('percent', 0.0):.1f}% ({cpu.get('count', 0)} cores)")

    ram = resources.get("memory", {})
    print(f"RAM: {ram.get('percent', 0.0):.1f}% ({ram.get('used_gb', 0.0):.1f}GB / {ram.get('total_gb', 0.0):.1f}GB)")

    if "gpu" in resources and resources["gpu"]:
        gpu = resources["gpu"]
        print(f"GPU: {gpu.get('utilization', 0.0):.1f}% ({gpu.get('memory_used_mb', 0):.0f}MB / {gpu.get('memory_total_mb', 0):.0f}MB)")

    print(f"\nActive Jobs: {resources.get('active_jobs', 0)}")
    print(f"Idle Mode: {'✅' if resources.get('idle_mode') else '❌'}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Stage 4 Clustering Service CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "command",
        help="Command to execute",
        choices=[
            "health",
            "stats",
            "cluster-events",
            "cluster-documents",
            "cluster-entities",
            "cluster-storylines",
            "job-status",
            "job-list",
            "job-cancel",
            "job-pause",
            "job-resume",
            "clusters",
            "cluster-info",
            "resources",
        ],
    )

    parser.add_argument("args", nargs="*", help="Command arguments")
    parser.add_argument("--algorithm", "-a", default="hdbscan", help="Algorithm (hdbscan/kmeans/agglomerative)")
    parser.add_argument("--min-cluster-size", type=int, help="Min cluster size (HDBSCAN)")
    parser.add_argument("--n-clusters", type=int, help="Number of clusters (K-Means)")
    parser.add_argument("--domain", help="Filter by domain")
    parser.add_argument("--event-type", help="Filter by event type")
    parser.add_argument("--no-temporal", action="store_true", help="Disable temporal clustering")
    parser.add_argument("--type", help="Filter clusters by type")
    parser.add_argument("--status", help="Filter jobs by status")
    parser.add_argument("--limit", type=int, default=50, help="Limit results")
    parser.add_argument("--watch", action="store_true", help="Watch job status (poll)")
    parser.add_argument("--base-url", default=API_BASE_URL, help="API base URL")

    args = parser.parse_args()

    cli = ClusteringCLI(base_url=args.base_url)

    # Execute command
    if args.command == "health":
        health = cli.health()
        print_health(health)

    elif args.command == "stats":
        stats = cli.statistics()
        print_stats(stats)

    elif args.command == "resources":
        resources = cli.resources()
        print_resources(resources)

    elif args.command.startswith("cluster-"):
        # Extract embedding type
        embedding_type = args.command.replace("cluster-", "")

        # Build algorithm params
        params = {}
        if args.min_cluster_size:
            params["min_cluster_size"] = args.min_cluster_size
        if args.n_clusters:
            params["n_clusters"] = args.n_clusters

        # Build filters
        filters = {}
        if args.domain:
            filters["domain"] = args.domain
        if args.event_type:
            filters["event_type"] = args.event_type

        # Submit job
        result = cli.cluster(
            embedding_type=embedding_type,
            algorithm=args.algorithm,
            params=params,
            filters=filters if filters else None,
            enable_temporal=not args.no_temporal,
        )

        print(f"✅ Job submitted: {result.get('job_id')}")
        print(f"   Status: {result.get('status')}")
        print(f"   Message: {result.get('message')}")

        # Watch if requested
        if args.watch:
            job_id = result.get("job_id")
            print(f"\n👀 Watching job {job_id}...")

            while True:
                job = cli.job_status(job_id)
                status = job.get("status")

                print(f"\r   Progress: {job.get('progress_percent', 0):.1f}% | "
                      f"Clusters: {job.get('clusters_created', 0)} | "
                      f"Status: {status.upper()}", end="")

                if status in ["completed", "failed", "canceled"]:
                    print()  # Newline
                    print_job(job)
                    break

                time.sleep(2)

    elif args.command == "job-status":
        if not args.args:
            print("❌ Job ID required", file=sys.stderr)
            sys.exit(1)

        job_id = args.args[0]
        job = cli.job_status(job_id)
        print_job(job)

    elif args.command == "job-list":
        result = cli.job_list(status=args.status, limit=args.limit)
        print(f"📋 Jobs (Total: {result.get('total', 0)})\n")

        for job in result.get("jobs", []):
            print_job(job)

    elif args.command in ["job-cancel", "job-pause", "job-resume"]:
        if not args.args:
            print("❌ Job ID required", file=sys.stderr)
            sys.exit(1)

        job_id = args.args[0]
        action = args.command.replace("job-", "")

        if action == "cancel":
            result = cli.job_cancel(job_id)
        else:
            result = cli.job_action(job_id, action)

        if result.get("success"):
            print(f"✅ {result.get('message')}")
        else:
            print(f"❌ {result.get('message')}", file=sys.stderr)

    elif args.command == "clusters":
        result = cli.list_clusters(
            embedding_type=args.type,
            limit=args.limit,
        )

        print(f"📦 Clusters (Total: {result.get('total', 0)})\n")
        for cluster in result.get("clusters", []):
            print(f"   {cluster}")

    elif args.command == "cluster-info":
        if not args.args:
            print("❌ Cluster ID required", file=sys.stderr)
            sys.exit(1)

        cluster_id = args.args[0]
        cluster = cli.cluster_info(cluster_id)
        print_json(cluster)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
