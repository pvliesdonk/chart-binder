# ReAct Pattern for LLM Tool Calling

## Overview

Chart-binder uses the **ReAct (Reasoning + Acting) pattern** for LLM tool calling instead of API-based tool binding. This approach uses prompt engineering to instruct the LLM on how to request and use tools.

## Why ReAct vs bind_tools?

### Problems with bind_tools (API-based tool calling):

1. **Fragile**: Each LLM provider implements tool calling differently
2. **Opaque**: Tool calls hidden in API responses, hard to debug
3. **Version-dependent**: API changes can break functionality
4. **Provider-specific**: Code needs modifications for different providers

### Benefits of ReAct (Prompt-based):

1. **Transparent**: All reasoning and tool calls visible as text
2. **Debuggable**: Easy to trace what the LLM is thinking
3. **Portable**: Works across any LLM that can follow instructions
4. **Robust**: Less fragile than relying on specific APIs
5. **Customizable**: Easy to modify format and behavior

## How ReAct Works

### 1. System Prompt Structure

The system prompt includes:
- Tool descriptions (name, arguments, return type)
- ReAct format instructions
- Examples of Thought/Action/Observation pattern

```
You have access to these tools:

**Tool: web_search**
Arguments:
  - query (string): Search query
Returns: Search results with titles and URLs

## How to Use Tools

When you need information, use this format:

Thought: [Explain what you're thinking]
Action: tool_name
Action Input: {"arg": "value"}
Observation: [Tool result will be filled here]

When ready, provide:
Final Answer:
```json
{...}
```
```

### 2. LLM Response Parsing

The implementation parses LLM text output for:

```python
# Example LLM output
"""
Thought: I need to verify the release date for this single
Action: web_search
Action Input: {"query": "Queen Killer Queen single release 1974"}
"""

# Parsed as:
{
  "name": "web_search",
  "args": {"query": "Queen Killer Queen single release 1974"}
}
```

### 3. Tool Execution

When a tool call is detected:

```python
# Execute tool
observation = execute_tool("web_search", {"query": "..."})

# Append to conversation
conversation += f"\nObservation: {observation}"

# Continue loop (LLM sees the observation and can use it)
```

### 4. Iteration Loop

```python
for iteration in range(MAX_ITERATIONS):
    response = llm.invoke(conversation)

    if "Final Answer:" in response:
        return parse_final_answer(response)

    tool_call = parse_tool_call(response)
    if tool_call:
        observation = execute_tool(tool_call)
        conversation += f"\n{response}\nObservation: {observation}"
    else:
        # No tool call, try to parse as final answer
        return parse_final_answer(response)
```

## Example Interaction

### Turn 1: Initial Request

**User:**
```
# Recording Evidence
## Artist: Queen
## Recording Title: Killer Queen

## Release Group Candidates
1. Killer Queen (Single, 1974)
2. Sheer Heart Attack (Album, 1974)

Determine the Canonical Release Group.
```

**LLM Response:**
```
Thought: I need to verify which release came first and whether the single
was a lead single for the album. Let me search for release dates.

Action: web_search
Action Input: {"query": "Queen Killer Queen single release date 1974"}
```

**System:** Executes `web_search(...)` → Returns results

### Turn 2: After Tool Use

**Observation:**
```
Web search results for 'Queen Killer Queen single release date 1974':

1. Killer Queen - Wikipedia
   Released as a single on October 11, 1974, three weeks before the album
   URL: https://en.wikipedia.org/wiki/Killer_Queen

2. Sheer Heart Attack - Wikipedia
   Released November 8, 1974
   URL: https://en.wikipedia.org/wiki/Sheer_Heart_Attack
```

**LLM Response:**
```
Thought: The single was released 3 weeks before the album. This is within
the 90-day lead single window, so I should prefer the album as the CRG
according to the canonicalization rules.

Final Answer:
```json
{
  "crg_mbid": "album-mbid-here",
  "rr_mbid": "release-mbid-here",
  "confidence": 0.9,
  "rationale": "Single released 3 weeks before album (within 90-day window)"
}
```
```

**System:** Parses final answer → Returns AdjudicationResult

## Implementation Details

### Tool Call Parsing

The regex patterns used:

```python
# Action pattern
action_match = re.search(r"Action:\s*(\w+)", text, re.IGNORECASE)

# Action Input (JSON or plain text)
input_match = re.search(
    r"Action Input:\s*({.*?}|\{[\s\S]*?\}|[^\n]+)",
    text,
    re.IGNORECASE | re.DOTALL
)
```

### Final Answer Parsing

Looks for:
1. "Final Answer:" marker
2. JSON code block: ````json ... ````
3. JSON object: `{...}`

Extracts and parses:
- `crg_mbid`
- `rr_mbid`
- `confidence`
- `rationale`

### Error Handling

1. **Max iterations**: Prevents infinite loops (default: 5)
2. **Malformed responses**: Attempts to extract partial JSON
3. **Tool errors**: Returns error message as observation
4. **Parse failures**: Returns ERROR outcome with details

## Configuration

Enable in `config.toml`:

```toml
[llm]
enabled = true
provider = "ollama"
model_id = "llama3.2"
temperature = 0.0

[llm.searxng]
enabled = true
url = "http://localhost:8080"
```

## Available Tools

### web_search

```python
def web_search(query: str) -> str:
    """Search the web for music release information.

    Returns formatted search results with titles, snippets, and URLs.
    """
```

### web_fetch

```python
def web_fetch(url: str) -> str:
    """Fetch and extract text content from a URL.

    Returns extracted text (HTML tags stripped, truncated to 2000 chars).
    """
```

## Debugging

Enable debug logging to see full ReAct loop:

```bash
uv run canon -vv decide audio.mp3
```

Output shows:
```
ReAct LLM Adjudication:
======================================================================
SYSTEM:
[Full system prompt with tool descriptions]
======================================================================
USER:
[Evidence bundle formatted as markdown]
======================================================================
ReAct iteration 1/5
LLM Response:
Thought: I need to search...
Action: web_search
Action Input: {"query": "..."}
======================================================================
Tool execution: web_search('...')
Observation: [Search results]
======================================================================
ReAct iteration 2/5
...
```

## Extending with New Tools

To add a new tool:

1. **Add to system prompt:**

```python
REACT_SYSTEM_PROMPT = """
...
**Tool: my_new_tool**
Description: What this tool does
Arguments:
  - arg1 (type): Description
Returns: What it returns
...
"""
```

2. **Add execution handler:**

```python
def _execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str:
    if tool_name == "web_search":
        return self._tool_web_search(tool_args.get("query", ""))
    elif tool_name == "web_fetch":
        return self._tool_web_fetch(tool_args.get("url", ""))
    elif tool_name == "my_new_tool":
        return self._tool_my_new_tool(tool_args)
    else:
        return f"Error: Unknown tool '{tool_name}'"

def _tool_my_new_tool(self, args: dict[str, Any]) -> str:
    # Implementation here
    pass
```

## Testing

Test with a simple case:

```python
from chart_binder.llm.react_adjudicator import ReActAdjudicator
from chart_binder.config import LLMConfig

config = LLMConfig(
    enabled=True,
    provider="ollama",
    model_id="llama3.2"
)

adjudicator = ReActAdjudicator(config)
result = adjudicator.adjudicate(evidence_bundle)

print(f"Outcome: {result.outcome}")
print(f"CRG: {result.crg_mbid}")
print(f"Confidence: {result.confidence}")
print(f"Reasoning: {result.rationale}")
```

## Comparison with Old Implementation

| Feature | bind_tools (Old) | ReAct (New) |
|---------|------------------|-------------|
| **Transparency** | Opaque API responses | Full text output visible |
| **Portability** | Provider-specific | Works with any LLM |
| **Debugging** | Hard (need API logs) | Easy (read the text) |
| **Robustness** | Fragile to API changes | Robust to model updates |
| **Control** | Limited by API | Full control via prompts |
| **Dependencies** | LangChain tool system | Just prompt engineering |

## Best Practices

1. **Keep tool descriptions clear**: LLMs work better with simple, concise tool docs
2. **Show examples**: Include example tool usage in system prompt
3. **Set max iterations**: Prevent runaway loops (5 is usually enough)
4. **Log everything**: Debug mode helps understand LLM behavior
5. **Handle errors gracefully**: Always return useful error messages
6. **Test with multiple models**: Verify portability across providers

## References

- Original ReAct paper: https://arxiv.org/abs/2210.03629
- LangChain docs on ReAct: https://python.langchain.com/docs/modules/agents/agent_types/react
- Our implementation: `src/chart_binder/llm/react_adjudicator.py`
