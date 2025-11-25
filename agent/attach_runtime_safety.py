from __future__ import annotations

from typing import Any, Dict, Union

from langchain.tools import BaseTool
from langchain_core.messages import ToolCall
from langchain_core.runnables import Runnable, RunnableConfig

ToolInput = Union[str, Dict[str, Any], ToolCall]


class SafetyLayerError(RuntimeError):
    """Raised when the safety layer returns an invalid or unsupported decision."""
    # Currently unused, but kept for future explicit policy enforcement.


def attach_runtime_safety(
    tool: BaseTool,
    safe_agent: Runnable,
) -> BaseTool:
    """
    Attach a runtime safety hook to a single BaseTool instance in-place.

    After this function is applied, all subsequent calls to `tool.run` / `tool.arun`
    (and, transitively, `tool.invoke` / `tool.ainvoke`) will synchronously or
    asynchronously call `safe_agent` *before* the underlying tool logic executes.

    The original `run` and `arun` implementations are preserved on the tool instance
    as `_unsafe_run` and `_unsafe_arun` to allow internal, explicitly unsafe bypasses
    (e.g. for debugging or controlled internal workflows).

    This function is idempotent: calling it multiple times on the same tool instance
    will only attach the safety layer once.
    """

    # Avoid re-wrapping the same tool instance multiple times.
    if getattr(tool, "_runtime_safety_attached", False):
        return tool

    object.__setattr__(tool, "_runtime_safety_attached", True)

    # Preserve original synchronous / asynchronous execution entrypoints.
    original_run = tool.run
    original_arun = tool.arun
    object.__setattr__(tool, "_unsafe_run", original_run)
    object.__setattr__(tool, "_unsafe_arun", original_arun)

    def safe_run(
        tool_input: ToolInput,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Synchronous execution wrapper for the tool.

        This wrapper invokes the `safe_agent` before delegating to the original
        `tool.run`. It does *not* intercept or re-implement callbacks, tracing,
        or argument parsing; all of those concerns remain handled by `original_run`.

        At this stage, the result returned by `safe_agent` is not interpreted.
        The call is purely observational and the tool execution is always allowed.
        Policy enforcement (e.g. block / rewrite / rate-limit) can be implemented
        here later based on the `decision` object.
        """
        cfg: RunnableConfig | None = kwargs.get("config")

        safety_request = {
            "tool_name": tool.name,
            "tool_description": tool.description,
            "tool_input": tool_input,
        }

        # Synchronous safety probe. The returned object is intentionally ignored
        # for now and can later be used to implement explicit policies.
        decision = safe_agent.invoke(safety_request, config=cfg)
        if decision:
            # Placeholder: inspect `decision` and enforce a policy
            # (e.g. BLOCK / REWRITE / ALLOW) before calling `original_run`.
            pass

        # Delegate to the original implementation to preserve callbacks, tracing,
        # args_schema handling, and response formatting.
        return original_run(tool_input, *args, **kwargs)

    async def safe_arun(
        tool_input: ToolInput,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Asynchronous execution wrapper for the tool.

        Analogous to `safe_run`, but using `safe_agent.ainvoke` and delegating
        to the original `tool.arun`. The safety layer is consulted before the
        underlying tool logic runs, but its output is not yet interpreted.
        """
        cfg: RunnableConfig | None = kwargs.get("config")

        safety_request = {
            "tool_name": tool.name,
            "tool_description": tool.description,
            "tool_input": tool_input,
        }

        decision = await safe_agent.ainvoke(safety_request, config=cfg)
        if decision:
            # Placeholder: inspect `decision` and enforce an asynchronous policy
            # prior to delegating to `original_arun`.
            pass

        return await original_arun(tool_input, *args, **kwargs)

    # Instance-level monkey-patching: from this point on, any code that calls
    # tool.run / tool.arun will transparently go through the safety wrapper.
    object.__setattr__(tool, "run", safe_run)      # type: ignore[assignment]
    object.__setattr__(tool, "arun", safe_arun)    # type: ignore[assignment]

    return tool
