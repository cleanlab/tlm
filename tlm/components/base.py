import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


# TODO: consider requiring typed properties rather than a generic "results" dictionary
# similar to Metadata object in existing SaaS TLM
# e.g. reference_completions: list[Completion], reference_answers: list[str], etc.
class ExecutionContext:
    def __init__(self):
        self.results: dict[str, Any] = {}

    def add(self, key: str, value: Any) -> None:
        if key in self.results:
            logger.warning(f"Result {key} already exists, overwriting old value")
        self.results[key] = value

    def get(self, key: str, default: Any | None = None) -> Any:
        return self.results.get(key, default)


class Component(ABC):
    def __init__(self, depends_on: list["Component"] | None = None):
        self.depends_on = depends_on or []
        self.blocking: list["Component"] = []
        self.execution_context = ExecutionContext()

        self._completed_dependencies: int = 0
        self._lock = asyncio.Lock()
        self._ready_event = asyncio.Event()

        for dependency in self.depends_on:
            dependency.blocking.append(self)

        if not self.depends_on:
            self._ready_event.set()

    def merge_context(self, dependency_context: ExecutionContext) -> None:
        self.execution_context.results.update(dependency_context.results)

    async def notify_completion(self, dependency: "Component") -> None:
        """Called upon completion of a dependency."""
        if dependency in self.depends_on:
            async with self._lock:
                self._completed_dependencies += 1
                if self._completed_dependencies >= len(self.depends_on):
                    self._ready_event.set()
                self.merge_context(dependency.execution_context)

    @abstractmethod
    async def execute(self) -> None:
        pass
