import json
from langchain_core.runnables import RunnableLambda


def build_dummy_safeagent_core(cfg: dict):
    core_cfg = cfg.get("dummy_safeagent_core", {})

    default_middleware_action = core_cfg.get("default_middleware_action", "APPROVE")
    allow_long_term_memory = core_cfg.get("allow_long_term_memory", True)
    default_tool_action = core_cfg.get("default_tool_action", "CALL_BLOCK")
    tool_actions = core_cfg.get("tool_actions", {})

    def _reply(payload: dict):
        return [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]

    def _safeagent_core(request, config=None):
        core_request = request.get("core_request", {})
        hook = core_request.get("hook", "")

        if hook == "tool_wrapper":
            observation = core_request.get("observation", {}) or {}
            tool_name = observation.get("name", "") or ""

            action = tool_actions.get(tool_name, default_tool_action)

            if tool_name in tool_actions:
                reason = f"Tool '{tool_name}' is explicitly configured as {action}."
            else:
                reason = f"Tool '{tool_name}' is not in the whitelist, so {action} is used."

            return _reply({
                "action": action,
                "allow_long_term_memory": allow_long_term_memory,
                "reason": reason
            })

        return _reply({
            "action": default_middleware_action,
            "allow_long_term_memory": allow_long_term_memory,
            "reason": f"Non-tool hook '{hook}' uses default middleware action."
        })

    return RunnableLambda(_safeagent_core)
