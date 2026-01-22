# LLM Adjudicator Refactor Plan

## Overview

Refactor the chart-binder LLM adjudication system from ReAct prompt engineering to proper tool calling and structured output using LangChain's modern APIs. This follows patterns proven in questfoundry.

## Current State (chart-binder)

**Location:** `src/chart_binder/llm/react_adjudicator.py`

**Issues with ReAct prompt approach:**
1. **Fragile parsing** - Regex-based extraction of `Action:`, `Action Input:`, `Observation:` patterns
2. **No type safety** - JSON parsing with multiple fallbacks, error-prone
3. **No validation** - Final answer JSON parsed without Pydantic validation
4. **Manual retry** - Ad-hoc retry logic when parsing fails
5. **Provider coupling** - Direct instantiation of `ChatOllama`/`ChatOpenAI` without abstraction

## Target State (from questfoundry patterns)

**Key improvements:**
1. **Native tool calling** via LangChain's `create_agent()` and `@tool` decorator
2. **Structured output** via `with_structured_output()` with Pydantic schemas
3. **Validation/repair loop** - Automatic retry with error feedback
4. **Provider factory** - Clean abstraction over Ollama/OpenAI/Anthropic
5. **Consistent JSON responses** - Tools return standardized JSON format

---

## Implementation Plan

### Phase 1: Provider Factory

**New file:** `src/chart_binder/llm/providers.py`

Create a provider factory following questfoundry's pattern:

```python
from langchain_core.language_models import BaseChatModel

PROVIDER_DEFAULTS = {
    "ollama": None,  # Require explicit model
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-20250514",
}

def create_chat_model(
    provider_name: str,
    model: str,
    **kwargs,
) -> BaseChatModel:
    """Create provider-agnostic chat model."""
    ...
```

**Features:**
- Ollama: Use `OLLAMA_HOST` env var, default 32k context
- OpenAI: Use `OPENAI_API_KEY`, handle reasoning models (o1/o3)
- Anthropic: Use `ANTHROPIC_API_KEY` (optional future addition)
- Temperature, seed, top_p handling per provider

### Phase 2: Tool Definitions

**New file:** `src/chart_binder/llm/tools.py`

Convert existing tool methods to LangChain `@tool` functions:

```python
from langchain_core.tools import tool

@tool
def search_artist(query: str) -> str:
    """Search MusicBrainz for artists by name.

    Args:
        query: Artist name to search

    Returns:
        JSON with artist results including MBIDs, countries, disambiguation
    """
    ...

@tool
def get_release_group(mbid: str) -> str:
    """Get release group details by MBID.

    Args:
        mbid: MusicBrainz release group ID

    Returns:
        JSON with release group details
    """
    ...
```

**Tools to implement:**
1. `search_artist(query: str)` - Search artists
2. `get_artist(mbid: str)` - Lookup artist by MBID
3. `search_recording(title: str, artist: str | None)` - Search recordings
4. `get_recording(mbid: str)` - Lookup recording
5. `search_release_group(title: str, artist: str | None)` - Search RGs
6. `get_release_group(mbid: str)` - Lookup RG
7. `get_releases_in_group(rg_mbid: str)` - List releases in RG
8. `web_search(query: str)` - Web search via SearxNG
9. `web_fetch(url: str)` - Fetch URL content

**JSON response format (ADR-008 standard from questfoundry):**
```json
{
  "result": "success|no_results|error",
  "content": "...",
  "action": "guidance on next steps"
}
```

### Phase 3: Structured Output Schema

**Update:** `src/chart_binder/llm/adjudicator.py`

Define Pydantic schema for adjudication result:

```python
from pydantic import BaseModel, Field

class AdjudicationResponse(BaseModel):
    """Structured response from LLM adjudicator."""

    crg_mbid: str | None = Field(
        default=None,
        description="Canonical Release Group MBID"
    )
    rr_mbid: str | None = Field(
        default=None,
        description="Representative Release MBID within the CRG"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score (0.0-1.0)"
    )
    rationale: str = Field(
        description="One-line explanation of the decision"
    )
```

### Phase 4: Structured Output Strategy

**New file:** `src/chart_binder/llm/structured_output.py`

Implement strategy selection (from questfoundry):

```python
from enum import Enum
from langchain_core.runnables import Runnable

class StructuredOutputStrategy(str, Enum):
    TOOL = "tool"        # Use function/tool calling
    JSON_MODE = "json_mode"  # Use native JSON schema
    AUTO = "auto"        # Auto-select per provider

def with_structured_output(
    model: BaseChatModel,
    schema: type[BaseModel],
    strategy: StructuredOutputStrategy | None = None,
    provider_name: str | None = None,
) -> Runnable:
    """Wrap model with structured output enforcement."""
    ...
```

**Key insight from questfoundry:** Default to `JSON_MODE` for all providers. The `TOOL` strategy (function calling) returns `None` with complex nested schemas on Ollama.

### Phase 5: Agent Adjudicator

**New file:** `src/chart_binder/llm/agent_adjudicator.py`

Replace ReAct prompt approach with native agent:

```python
from langchain.agents import create_agent

class AgentAdjudicator:
    """LLM adjudicator using native tool calling."""

    def __init__(
        self,
        config: LLMConfig,
        db: MusicGraphDB | None = None,
        ...
    ):
        # Create provider-agnostic model
        self.model = create_chat_model(
            config.provider,
            config.model_id,
            temperature=config.temperature,
        )

        # Initialize tools
        self.tools = self._create_tools(db, ...)

        # Create agent with structured output
        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=ADJUDICATION_SYSTEM_PROMPT,
            response_format=AdjudicationResponse,
        )

    async def adjudicate(
        self,
        evidence_bundle: dict[str, Any],
    ) -> AdjudicationResult:
        """Run adjudication agent."""
        result = await self.agent.ainvoke({
            "messages": [{"role": "user", "content": self._build_prompt(evidence_bundle)}]
        })

        # Extract structured response
        response = result.get("structured_response")
        ...
```

### Phase 6: Validation/Repair Loop

Implement serialize-style validation with retry:

```python
async def adjudicate_with_retry(
    model: BaseChatModel,
    evidence_bundle: dict[str, Any],
    max_retries: int = 3,
) -> tuple[AdjudicationResponse, int]:
    """Adjudicate with validation/repair loop."""

    structured_model = with_structured_output(
        model, AdjudicationResponse, strategy=StructuredOutputStrategy.JSON_MODE
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_evidence_prompt(evidence_bundle)),
    ]

    total_tokens = 0

    for attempt in range(1, max_retries + 1):
        result = await structured_model.ainvoke(messages)
        total_tokens += extract_tokens(result)

        if isinstance(result, AdjudicationResponse):
            # Validate response
            if result.crg_mbid and result.confidence > 0:
                return result, total_tokens

            # Invalid response - add feedback
            if attempt < max_retries:
                messages.append(HumanMessage(content=format_validation_errors(result)))

    raise AdjudicationError(f"Failed after {max_retries} attempts")
```

---

## Migration Strategy

### Step 1: Create Provider Factory (Low Risk)
- Add `providers.py` alongside existing code
- No changes to existing functionality
- Unit test each provider

### Step 2: Create Tool Wrappers (Low Risk)
- Add `tools.py` with `@tool` decorated functions
- Wrap existing `SearchTool` methods
- Unit test each tool

### Step 3: Create Agent Adjudicator (Parallel to Existing)
- Add `agent_adjudicator.py` as alternative to `react_adjudicator.py`
- Feature flag in `LLMConfig` to select adjudicator type
- Integration test both paths

### Step 4: Add Structured Output (Enhancement)
- Add `structured_output.py` with strategy selection
- Add `AdjudicationResponse` Pydantic model
- Unit test structured output parsing

### Step 5: Wire Up with Validation Loop
- Update `AgentAdjudicator.adjudicate()` to use validation/repair
- Integration test full flow
- Compare results with ReAct adjudicator

### Step 6: Deprecate ReAct Adjudicator
- After validation, make agent adjudicator the default
- Keep ReAct as fallback option
- Eventually remove

---

## File Structure

```
src/chart_binder/llm/
├── __init__.py
├── adjudicator.py           # Base types, AdjudicationResult
├── agent_adjudicator.py     # NEW: Native tool calling agent
├── providers.py             # NEW: Provider factory
├── react_adjudicator.py     # EXISTING: ReAct prompt approach
├── search_tool.py           # EXISTING: MusicBrainz search
├── searxng.py               # EXISTING: Web search
├── structured_output.py     # NEW: Structured output utilities
└── tools.py                 # NEW: @tool decorated functions
```

---

## Configuration Changes

**Update:** `src/chart_binder/config.py`

```python
@dataclass
class LLMConfig:
    enabled: bool = False
    provider: str = "ollama"
    model_id: str = "qwen3:8b"
    temperature: float = 0.1

    # NEW: Adjudicator type selection
    adjudicator_type: str = "agent"  # "agent" or "react"

    # NEW: Structured output strategy
    structured_output_strategy: str = "auto"  # "auto", "json_mode", "tool"

    # Existing thresholds
    auto_accept_threshold: float = 0.85
    review_threshold: float = 0.60
```

---

## Dependencies

**Add to pyproject.toml:**
```toml
[project.optional-dependencies]
llm = [
    "langchain>=1.0.0",      # Core LangChain v1.0+
    "langchain-core>=0.3",
    "langchain-ollama>=0.3",
    "langchain-openai>=0.3",
    # Optional for Anthropic support
    "langchain-anthropic>=0.3",
]
```

---

## Testing Plan

### Unit Tests
1. Provider factory creates correct model types
2. Each tool returns valid JSON with correct schema
3. Structured output validates Pydantic models
4. Validation loop retries on errors

### Integration Tests
1. Full adjudication flow with mock evidence
2. Compare agent vs ReAct results on same inputs
3. Test with Ollama and OpenAI providers

### Regression Tests
1. Run existing QA pack through both adjudicators
2. Verify identical outcomes
3. Check performance (tokens, latency)

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Tool calling not available on all Ollama models | Use JSON_MODE strategy, test with qwen3:8b |
| Structured output returns None | Validation loop with error feedback |
| Breaking existing workflows | Feature flag, parallel implementations |
| Different results from ReAct | Extensive testing, gradual rollout |

---

## Timeline

1. **Phase 1-2:** Provider factory + tools (2-3 hours)
2. **Phase 3-4:** Structured output schema + strategy (2 hours)
3. **Phase 5:** Agent adjudicator (3-4 hours)
4. **Phase 6:** Validation loop + testing (2-3 hours)
5. **Migration:** Integration + deprecation (ongoing)

**Total estimated effort:** 10-15 hours

---

## References

- questfoundry implementation: `/mnt/code/questfoundry/src/questfoundry/`
  - `providers/factory.py` - Provider factory
  - `agents/discuss.py` - Agent creation
  - `agents/serialize.py` - Validation/repair loop
  - `tools/langchain_tools.py` - Tool definitions
  - `providers/structured_output.py` - Structured output strategy
- LangChain docs: `create_agent`, `with_structured_output`, `@tool` decorator
