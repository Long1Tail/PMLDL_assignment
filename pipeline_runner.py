import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import urllib.request
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("PipelineRunner")

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable


def run_stage_1():
    logger.info("STAGE 1: Starting Data Engineering...")
    script = PROJECT_ROOT / "code" / "datasets" / "data_pipeline.py"
    cmd = [PYTHON_EXE, str(script)]
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        logger.error("Stage 1 failed with exit code %d", res.returncode)
        return False
    logger.info("Stage 1 completed successfully.")
    return True


def run_stage_2():
    logger.info("▶ STAGE 2: Starting Model Engineering (MLflow)...")
    script = PROJECT_ROOT / "code" / "models" / "train.py"
    cmd = [PYTHON_EXE, str(script)]
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        logger.error("Stage 2 failed with exit code %d", res.returncode)
        return False
    logger.info("Stage 2 completed successfully.")
    return True


def run_stage_3() -> bool:
    """Execute Stage 3: Deployment via Docker Compose."""
    logger.info("STAGE 3: Deploying Microservices via Docker...")
    compose_file = PROJECT_ROOT / "code" / "deployment" / "docker-compose.yml"
    cmd = ["docker", "compose", "-f", str(compose_file), "up", "-d", "--build"]
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        logger.error("Stage 3 Docker Compose failed with exit code %d", res.returncode)
        return False

    logger.info("Verifying services health...")
    time.sleep(5)
    for attempt in range(6):
        try:
            req = urllib.request.Request("http://localhost:8000/health")
            with urllib.request.urlopen(req, timeout=3) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    logger.info("✔ FastAPI backend is healthy! Run ID: %s", data.get("run_id"))
                    break
        except Exception as e:
            logger.info("Waiting for FastAPI container... attempt %d/6 (%s)", attempt + 1, e)
            time.sleep(3)

    logger.info("Streamlit web app available at: http://localhost:8501")
    logger.info("FastAPI documentation at: http://localhost:8000/docs")
    return True


def execute_full_pipeline():
    start_time = datetime.now()
    logger.info("[%s] Starting Full Pipeline Run...", start_time.strftime("%Y-%m-%d %H:%M:%S"))

    if not run_stage_1():
        return False

    if not run_stage_2():
        return False

    if not run_stage_3():
        return False

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("[%s] Pipeline run finished successfully in %.2f seconds!", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), elapsed)
    return True


def main():
    parser = argparse.ArgumentParser(description="Automated MLOps Pipeline Runner")
    parser.add_argument("--interval", type=int, default=5, help="Schedule interval in minutes (default: 5)")
    parser.add_argument("--once", action="store_true", help="Run pipeline once and exit immediately")
    args = parser.parse_args()

    if args.once:
        success = execute_full_pipeline()
        sys.exit(0 if success else 1)

    interval_sec = args.interval * 60
    logger.info("Automated Scheduler active: Pipeline will run every %d minutes (%d seconds).", args.interval, interval_sec)
    logger.info("Press Ctrl+C to stop.")

    cycle = 1
    while True:
        logger.info("\n>>> Starting Pipeline Execution Cycle #%d <<<", cycle)
        execute_full_pipeline()
        logger.info("Sleeping for %d minutes until next run...", args.interval)
        try:
            time.sleep(interval_sec)
        except KeyboardInterrupt:
            logger.info("Scheduler interrupted by user. Exiting cleanly.")
            break
        cycle += 1


if __name__ == "__main__":
    main()
