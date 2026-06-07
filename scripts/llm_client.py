"""Provider-agnostic LLM text completion (system + user -> text).

Two providers are supported:

  - ``anthropic`` : native Anthropic SDK (Claude). Use an Anthropic API key
                    (``ANTHROPIC_API_KEY`` or ``LLM_API_KEY``). The system prompt
                    is sent as the top-level ``system=`` parameter, per the
                    Anthropic Messages API — not pointed at an OpenAI-compatible
                    shim.
  - ``openai``    : any OpenAI-compatible endpoint (ModelScope / SiliconFlow /
                    Qwen, etc.) via the ``openai`` client and ``--base-url``.

Both expose the same ``complete(system, user, max_tokens) -> str`` method so the
scoring and rubric scripts don't care which provider is in use.
"""

from __future__ import annotations

DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"


class LLMClient:
    """Thin wrapper over the Anthropic or OpenAI-compatible clients."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        if provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        elif provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            raise ValueError(f"Unknown provider: {provider!r} (use 'anthropic' or 'openai')")

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """Return the model's text response to (system, user)."""
        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(
                b.text for b in resp.content if getattr(b, "type", None) == "text"
            ).strip()

        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content.strip()
