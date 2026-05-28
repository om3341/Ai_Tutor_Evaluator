from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from loguru import logger

from backend.config import Settings


class ModelLoadError(RuntimeError):
    """Raised when a model server cannot be reached or started."""


@dataclass(frozen=True)
class ModelProcessConfig:
    label: str
    health_url: str
    start_command: str


class ModelProcessManager:
    """Starts locally configured model server commands and tracks child processes.

    Commands come from `.env` because Qwen/vLLM, Ollama, SSH tunnels, and server
    paths are deployment-specific. The backend never stores passwords; use SSH
    keys or an already-running tunnel for non-interactive startup.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._log_dir = Path("logs/model_processes")
        self._log_dir.mkdir(parents=True, exist_ok=True)

    async def load(self, model_name: str) -> dict[str, object]:
        config = self._config_for_model(model_name)
        if await self._is_healthy(config.health_url):
            return {"loaded": True, "active_model": model_name, "started_process": False, "health_url": config.health_url}

        if not self._settings.model_autostart_enabled:
            raise ModelLoadError(f"{config.label} is not reachable at {config.health_url}. Autostart is disabled.")
        if not config.start_command.strip():
            raise ModelLoadError(
                f"{config.label} is not reachable at {config.health_url}, and no start command is configured."
            )

        process = self._start_process(model_name, config.start_command)
        await self._wait_until_healthy(config, process)
        return {"loaded": True, "active_model": model_name, "started_process": True, "health_url": config.health_url}

    def unload(self, model_name: str | None) -> None:
        if not model_name:
            return
        process = self._processes.pop(model_name, None)
        if process is None or process.poll() is not None:
            return

        logger.info("Stopping managed model process model={} pid={}", model_name, process.pid)
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return

    def _start_process(self, model_name: str, command: str) -> subprocess.Popen[bytes]:
        existing = self._processes.get(model_name)
        if existing is not None and existing.poll() is None:
            logger.info("Stopping stale managed model process before restart model={} pid={}", model_name, existing.pid)
            try:
                os.killpg(existing.pid, signal.SIGTERM)
                existing.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass

        logger.info("Starting model command for {}: {}", model_name, command)
        safe_name = "".join(char if char.isalnum() else "_" for char in model_name.lower())
        stdout_path = self._log_dir / f"{safe_name}.out.log"
        stderr_path = self._log_dir / f"{safe_name}.err.log"
        stdout_handle = stdout_path.open("ab")
        stderr_handle = stderr_path.open("ab")
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
        self._processes[model_name] = process
        return process

    async def _wait_until_healthy(self, config: ModelProcessConfig, process: subprocess.Popen[bytes]) -> None:
        deadline = time.monotonic() + self._settings.model_startup_timeout_seconds
        while time.monotonic() < deadline:
            if await self._is_healthy(config.health_url):
                return
            exit_code = process.poll()
            if exit_code is not None:
                recent_logs = self._recent_logs(config.label)
                raise ModelLoadError(
                    f"{config.label} start command exited with code {exit_code} before becoming healthy. "
                    f"Check logs in {self._log_dir}. {recent_logs}"
                )
            await asyncio.sleep(2.0)
        recent_logs = self._recent_logs(config.label)
        raise ModelLoadError(
            f"{config.label} did not become reachable at {config.health_url} within "
            f"{self._settings.model_startup_timeout_seconds}s. Check logs in {self._log_dir}. {recent_logs}"
        )

    async def _is_healthy(self, url: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
            return response.status_code < 500
        except httpx.HTTPError:
            return False

    def _recent_logs(self, label: str, max_chars: int = 1200) -> str:
        safe_name = "".join(char if char.isalnum() else "_" for char in label.lower())
        matching_logs = sorted(self._log_dir.glob("*.err.log"), key=lambda path: path.stat().st_mtime, reverse=True)
        for path in matching_logs:
            if safe_name.split("_")[0] in path.name:
                text = path.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    return f"Recent error log: {text[-max_chars:]}"
        return "No recent error log output was captured."

    def _config_for_model(self, model_name: str) -> ModelProcessConfig:
        normalized = model_name.casefold()
        if "qwen" in normalized:
            return ModelProcessConfig(
                label="Qwen vLLM server",
                health_url=f"{self._settings.qwen_base_url.rstrip('/')}/models",
                start_command=self._settings.qwen_start_command,
            )
        if "gemma" in normalized:
            return ModelProcessConfig(
                label="Gemma Ollama server",
                health_url=f"{self._settings.gemma_base_url.rstrip('/')}/api/tags",
                start_command=self._settings.gemma_start_command,
            )
        if "llama" in normalized:
            return ModelProcessConfig(
                label="Llama Ollama server",
                health_url=f"{self._settings.llama_base_url.rstrip('/')}/api/tags",
                start_command=self._settings.llama_start_command or self._settings.gemma_start_command,
            )
        raise ModelLoadError(f"No model loader configured for '{model_name}'.")
