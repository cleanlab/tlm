import asyncio
import logging
from typing import Any

from tlm.components import Component

logger = logging.getLogger(__name__)


class InferencePipeline:
    def __init__(self):
        self.components: list[Component] = []

    async def run(self) -> dict[str, Any]:
        self._validate()

        # Create and start tasks for all components
        # They will wait on their ready events before executing
        component_tasks: list[asyncio.Task] = [
            asyncio.create_task(self._execute_component(component)) for component in self.components
        ]

        await asyncio.gather(*component_tasks)

        final_results = {}
        for component in self.components:
            final_results.update(component.execution_context.results)

        return final_results

    async def _execute_component(self, component: Component) -> None:
        """Execute a component after waiting for all dependencies to complete."""
        await component._ready_event.wait()
        await component.execute()

        for child in component.blocking:
            await child.notify_completion(component)

    def _validate(self) -> None:
        # Validate all dependencies are in the pipeline
        component_set = set(self.components)
        for component in self.components:
            for dep in component.depends_on:
                if dep not in component_set:
                    raise ValueError(
                        f"Component dependency {dep} is not in the pipeline. "
                        "All dependencies must be added to the pipeline."
                    )

        # Validate that the dependency graph has no cycles using DFS
        visited: set[Component] = set()
        rec_stack: set[Component] = set()

        for component in self.components:
            if component not in visited:
                if self._has_cycle(component, visited, rec_stack):
                    raise ValueError("Cycle detected in pipeline dependency graph")

    def _has_cycle(self, component: Component, visited: set[Component], rec_stack: set[Component]) -> bool:
        """Check if a cycle exists starting from the given component."""
        if component in rec_stack:
            return True
        if component in visited:
            return False

        visited.add(component)
        rec_stack.add(component)

        for dep in component.depends_on:
            if self._has_cycle(dep, visited, rec_stack):
                return True

        rec_stack.remove(component)
        return False

    def add(self, component: Component) -> Component:
        self.components.append(component)
        return component
