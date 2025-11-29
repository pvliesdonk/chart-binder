# Batch LLM Adjudication Design

## Overview

Batch adjudication processes all INDETERMINATE decisions using the LLM adjudicator, with rate limiting, progress tracking, and resume capability.

## Use Cases

1. **Initial processing**: Adjudicate all INDETERMINATE after running resolver on a large library
2. **Incremental updates**: Adjudicate new INDETERMINATE decisions after adding more charts
3. **Re-adjudication**: Retry failed/low-confidence adjudications with better prompts

## Architecture

### Command: `charts llm batch-adjudicate`

```bash
# Basic usage
charts llm batch-adjudicate

# With options
charts llm batch-adjudicate \
  --limit 100 \
  --min-confidence 0.85 \
  --rate-limit 10 \
  --dry-run \
  --resume session_abc123
```

### Options

- `--limit N`: Process at most N decisions (default: all)
- `--min-confidence FLOAT`: Only accept adjudications with confidence >= threshold (default: 0.85)
- `--rate-limit N`: Max requests per minute (default: from config or provider limits)
- `--dry-run`: Show what would be done without making API calls
- `--resume SESSION_ID`: Resume a previous batch session
- `--force`: Process even if LLM is disabled
- `--skip-reviewed`: Skip decisions already in review queue
- `--auto-accept-threshold FLOAT`: Auto-accept above this confidence (default: from config)
- `--review-threshold FLOAT`: Add to review queue above this confidence (default: from config)

### Workflow

1. **Query INDETERMINATE decisions**
   - Use `DecisionsDB.get_stale_decisions()`
   - Filter for `state = 'indeterminate'`
   - Skip if already in review queue (optional)

2. **Create batch session**
   - Generate session ID
   - Store session metadata (start time, total, config)
   - Track progress in separate table

3. **Process each decision**
   - Load evidence bundle
   - Call LLM adjudicator
   - Handle rate limiting (exponential backoff)
   - Store result based on confidence:
     - High (>= auto_accept): Update decision directly
     - Medium (>= review): Add to ReviewQueue
     - Low (< review): Keep as INDETERMINATE, log
     - Error: Log, optionally retry

4. **Progress tracking**
   - Show progress bar
   - Display running stats: accepted/reviewed/rejected/errors
   - Allow Ctrl-C to gracefully stop (save session state)

5. **Resume capability**
   - Track which file_ids have been processed
   - Allow resume from last checkpoint
   - Session expires after 7 days

### Database Schema

Add to `decisions.sqlite`:

```sql
CREATE TABLE IF NOT EXISTS llm_batch_session (
    session_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    completed_at REAL,
    total_count INTEGER NOT NULL,
    processed_count INTEGER NOT NULL,
    accepted_count INTEGER NOT NULL,
    reviewed_count INTEGER NOT NULL,
    rejected_count INTEGER NOT NULL,
    error_count INTEGER NOT NULL,
    config_snapshot_json TEXT NOT NULL,
    state TEXT NOT NULL  -- 'running', 'completed', 'cancelled', 'error'
);

CREATE TABLE IF NOT EXISTS llm_batch_result (
    session_id TEXT NOT NULL,
    file_id TEXT NOT NULL,
    processed_at REAL NOT NULL,
    outcome TEXT NOT NULL,  -- 'accepted', 'review', 'rejected', 'error'
    crg_mbid TEXT,
    rr_mbid TEXT,
    confidence REAL,
    rationale TEXT,
    model_id TEXT,
    error_message TEXT,
    adjudication_id TEXT,
    PRIMARY KEY (session_id, file_id),
    FOREIGN KEY (session_id) REFERENCES llm_batch_session(session_id)
);

CREATE INDEX IF NOT EXISTS idx_batch_result_session ON llm_batch_result(session_id);
CREATE INDEX IF NOT EXISTS idx_batch_result_outcome ON llm_batch_result(outcome);
```

### Rate Limiting

1. **Provider-specific limits**:
   - OpenAI: 500 RPM (requests per minute) for tier 1, 3000 for tier 2
   - Ollama: Local, no limit but respect system resources

2. **Implementation**:
   - Token bucket algorithm
   - Sleep between requests
   - Exponential backoff on 429 (rate limit) errors
   - Configurable via `--rate-limit`

3. **Cost estimation**:
   - Before starting, estimate cost based on:
     - Number of decisions
     - Average prompt size (~2K tokens)
     - Model pricing (gpt-4o-mini: $0.15/1M input tokens)
   - Show estimate and ask for confirmation

### Error Handling

1. **Retryable errors**:
   - Network errors: Retry with exponential backoff (3 attempts)
   - Rate limit (429): Wait and retry
   - Timeout: Retry with longer timeout

2. **Non-retryable errors**:
   - Invalid credentials: Stop immediately
   - Quota exceeded: Stop immediately
   - Invalid response: Log and skip

3. **Graceful shutdown**:
   - Catch SIGINT (Ctrl-C)
   - Save current session state
   - Display resume command

### Output Actions

Based on confidence thresholds:

1. **Auto-accept** (confidence >= auto_accept_threshold, default 0.85):
   - Update `decision` table with CRG/RR from LLM
   - Update state to `decided`
   - Store adjudication metadata

2. **Add to review queue** (review_threshold <= confidence < auto_accept_threshold):
   - Create `ReviewItem` with source=`LLM_REVIEW`
   - Include LLM suggestion in `llm_suggestion_json`
   - Keep decision state as `indeterminate`

3. **Reject** (confidence < review_threshold, default 0.60):
   - Keep state as `indeterminate`
   - Log in batch results
   - Don't update decision

4. **Error**:
   - Keep state as `indeterminate`
   - Log error in batch results
   - Optionally retry

### CLI Output

```
Batch LLM Adjudication
======================

Querying INDETERMINATE decisions... 234 found
Estimating cost: ~$2.34 (234 decisions × ~2K tokens × $0.15/1M)

Continue? [y/N]: y

Creating batch session: batch_20250115_abc123

Processing decisions:
[████████████░░░░░░░░] 150/234 (64%)

Statistics:
  ✓ Auto-accepted:  87 (58%)
  ⚠ Needs review:   42 (28%)
  ✗ Rejected:       18 (12%)
  ⨯ Errors:         3 (2%)

Rate limit: 10 req/min | Elapsed: 15:23 | ETA: 8:24

^C Stopping gracefully...

Session saved. Resume with:
  charts llm batch-adjudicate --resume batch_20250115_abc123
```

### Implementation Phases

**Phase 1: Core batch processing**
- Query indeterminate decisions
- Process with rate limiting
- Store results in batch tables
- Basic progress display

**Phase 2: Enhanced features**
- Resume capability
- Auto-accept/review/reject logic
- Integration with DecisionsDB and ReviewQueue
- Cost estimation

**Phase 3: Advanced features**
- Parallel processing (multiple workers)
- Smart batching (group similar decisions)
- Quality metrics (confidence distribution)
- Export batch results to CSV/JSON

## Testing Strategy

1. **Unit tests**:
   - Rate limiter
   - Session management
   - Result processing

2. **Integration tests**:
   - Mock LLM provider
   - Test with small dataset (10 decisions)
   - Test resume from checkpoint

3. **Manual testing**:
   - Run on real dataset
   - Test Ctrl-C handling
   - Verify cost estimates

## Example Usage

```bash
# Process all INDETERMINATE, auto-accept high confidence
charts llm batch-adjudicate --auto-accept-threshold 0.90

# Process 100 decisions with lower rate limit
charts llm batch-adjudicate --limit 100 --rate-limit 5

# Dry run to see what would happen
charts llm batch-adjudicate --dry-run

# Resume interrupted session
charts llm batch-adjudicate --resume batch_20250115_abc123

# Process but don't auto-accept anything (all go to review)
charts llm batch-adjudicate --auto-accept-threshold 1.0

# View batch session results
charts llm batch-results batch_20250115_abc123
```

## Benefits

1. **Efficiency**: Process hundreds/thousands of decisions automatically
2. **Cost control**: Estimate and limit costs
3. **Quality control**: Configurable confidence thresholds
4. **Transparency**: Full audit trail of all adjudications
5. **Resume capability**: Survive interruptions, retry failures
6. **Human oversight**: Medium-confidence go to review queue

## Considerations

1. **API costs**: OpenAI charges per token, can add up
2. **Time**: Processing thousands takes hours with rate limits
3. **Quality**: LLM not perfect, may need human review
4. **Stale evidence**: Evidence bundle might be outdated
