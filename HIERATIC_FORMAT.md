# Hieratic Format Specification v1.0

## Introduction

**Hieratic** is a structured, LLM-parseable prompt compression format designed to dramatically reduce token usage while preserving semantic integrity. Named after ancient Egypt's cursive writing system (a practical simplification of hieroglyphics), Hieratic enables 70-90% token reduction for LLM prompts.

## Design Philosophy

### Historical Inspiration

Around 3000 BCE, Egyptian scribes developed Hieratic script as a faster, more efficient version of hieroglyphics for everyday documents. Similarly, Hieratic format compresses verbose prompts into a compact structure that LLMs can understand natively.

### Core Principles

1. **LLM-Native**: Uses syntax patterns familiar to LLMs (markdown-like, @-directives)
2. **Semantic Preservation**: Maintains meaning while reducing tokens
3. **Self-Documenting**: Format explains itself through structure
4. **Human-Readable**: Developers can easily read and edit compressed prompts
5. **Backward Compatible**: Falls back gracefully if LLM doesn't understand

## Format Specification

### Basic Structure

```hieratic
@HIERATIC v1.0
@CONTEXT: contexts.toml

@ROLE[role_id]
"Role description"

@EXAMPLES[examples_id]
- Example 1
- Example 2

@TASK
Main instruction

@FOCUS: key areas
@STYLE: response preferences
```

### Directives

#### Required Directives

**`@HIERATIC v1.0`**
- Must be first line
- Declares format version
- Enables LLMs to recognize compressed format

**`@TASK`**
- Core instruction or question
- Can contain multi-line content
- Supports code blocks, lists, and formatting

#### Optional Directives

**`@CONTEXT: <path>`**
- References external context library
- Path to TOML file with reusable patterns
- Must be a relative or absolute file path
- Omit for inline-only compression
- Example: `@CONTEXT: contexts.toml` or `@CONTEXT: ./libs/prompts.toml`

**`@ROLE[id]`**
- System role or persona
- Can reference context library or be inline
- Inline format: `@ROLE[inline] "description"`
- Referenced format: `@ROLE[role_001]`

**`@EXAMPLES[id]`**
- Example inputs/outputs
- Supports bullet lists or numbered items
- Can reference context library

**`@CONSTRAINTS[id]`**
- Limitations or requirements
- Format guidelines
- Output restrictions

**`@FOCUS: <areas>`**
- Comma-separated focus areas
- Guides LLM attention
- Example: `@FOCUS: performance, security, maintainability`

**`@STYLE: <preferences>`**
- Response style preferences
- Example: `@STYLE: concise, actionable, technical`

**`@FORMAT: <structure>`**
- Desired output structure
- Example: `@FORMAT: numbered list, 3-5 items`

**`@ANCHOR[id]`** (Incremental Compression)
- Compression anchor for incremental mode
- Marks a point where content has been compressed
- Contains a compressed summary of content up to that point
- Used internally for multi-turn conversations
- Format: `@ANCHOR[uuid]` followed by summary content

**`@RECENT`** (Incremental Compression)
- Marks recent, uncompressed content in incremental mode
- Contains the most recent tokens that haven't been compressed yet
- Used to maintain context while avoiding re-compression

### Inline vs Referenced Modes

#### Inline Mode

All content embedded directly in the file:

```hieratic
@HIERATIC v1.0

@ROLE[inline]
"Expert programmer: 10y Python, clean code, SOLID principles"

@EXAMPLES[inline]
1. Bug fix: auth bypass → HMAC signing
2. Performance: DB query 2.3s → 0.1s

@TASK
Review this code for security issues
```

#### Referenced Mode

Content stored in external context library:

```hieratic
@HIERATIC v1.0
@CONTEXT: contexts.toml

@ROLE[system_role_001]
@EXAMPLES[examples_002]
@CONSTRAINTS[output_format_003]

@TASK
Review this code for security issues
```

### Scoring Modes

Hieratic compression supports different scoring modes for determining sentence importance:

**Heuristic** (default)
- Keyword-based scoring
- Position-based importance
- Fast, no external dependencies
- Works without embedding models

**Semantic** (requires `compression-embeddings` feature)
- Embedding-based similarity scoring
- Uses ONNX models (all-MiniLM-L6-v2)
- Better understanding of semantic importance
- Requires model setup: `tokuin setup models`

**Hybrid** (requires `compression-embeddings` feature)
- Combines semantic (70%) and heuristic (30%) scoring
- Best of both worlds
- Recommended for optimal quality
- Requires model setup: `tokuin setup models`

**Usage:**
```bash
# Heuristic scoring (default)
tokuin compress prompt.txt --level medium

# Semantic scoring
tokuin compress prompt.txt --scoring semantic --level medium

# Hybrid scoring (recommended)
tokuin compress prompt.txt --scoring hybrid --level medium --quality
```

### Compression Techniques

#### 1. Abbreviations
- Technical terms kept exact
- Common words abbreviated
- Example: "years" → "y", "experience" → "exp"

#### 2. Arrow Notation
- `→` replaces "then", "leads to", "results in"
- Example: "Input is processed then validated" → "Input → processed → validated"

#### 3. Compact Lists
- Remove connecting prose
- Use bullets or numbers only
- Example: "The first example shows..." → "1. Example..."

#### 4. Structured Format
- Replace verbose instructions with @-directives
- Example: "Please provide a concise response" → `@STYLE: concise`

#### 5. Context References
- Extract repeated patterns to context library
- Replace with `@ROLE[id]` or `@EXAMPLES[id]`

#### 6. Structured Document Mode
- Preserves JSON document structure
- Keeps HTML tables intact
- Detects and consolidates repetitive instruction patterns
- Segments by logical sections (definitions, examples, formats)
- Structure-aware importance scoring

**When to use `--structured`:**
- LLM extraction/parsing instructions with JSON documents
- Prompts with HTML tables or code blocks
- Technical specifications with repeated formatting rules
- Documents with clear sections (Definition:, Location:, Response Format:)

**Usage:**
```bash
tokuin compress technical-prompt.txt --structured --level medium
```

## Examples

### Example 1: Code Review Prompt

**Original (850 tokens):**

```
You are an expert software engineer with 15 years of experience in full-stack development. You specialize in Python, JavaScript, and Go. You have deep knowledge of software architecture, design patterns, SOLID principles, and performance optimization. You always focus on writing clean, maintainable code that follows industry best practices.

Here are some examples of your past work:

Example 1: Bug Fix in Authentication System
Problem: Users were able to bypass the login system by manipulating session tokens
Solution: Implemented HMAC-based token signing with rotating keys and added rate limiting
Impact: Eliminated the security vulnerability and reduced bot traffic by 94%

Example 2: Database Performance Optimization
Problem: Database queries were taking an average of 2.3 seconds to complete
Solution: Added connection pooling, implemented query result caching, and optimized indexes
Impact: Reduced average query time to 0.1 seconds and increased system capacity by 10x

Example 3: Microservices Refactoring
Problem: Monolithic application was becoming difficult to maintain and deploy
Solution: Extracted bounded contexts into separate microservices with event-driven architecture
Impact: Development velocity increased by 3x, deployment frequency increased by 5x

Now, please analyze the following code and provide specific recommendations for improvement:

```python
def calculate_total(items):
    total = 0
    for item in items:
        total = total + item.price
    return total
```

Please provide:
- A detailed analysis of the code
- 3-5 specific improvement suggestions
- Focus on performance, security, and maintainability
- Keep your response concise and actionable
```

**Hieratic (285 tokens, 66.5% reduction):**

```hieratic
@HIERATIC v1.0

@ROLE[inline]
"Expert engineer: 15y full-stack, Python/JS/Go, architecture, patterns, SOLID, perf optimization"

@EXAMPLES[inline]
1. Auth bug: session token bypass → HMAC signing+rate limit → 94% bot reduction
2. DB perf: 2.3s queries → pooling+cache+indexes → 0.1s, 10x capacity
3. Refactor: monolith → event-driven µservices → 3x velocity, 5x deploys

@TASK
Analyze code, provide recommendations:

```python
def calculate_total(items):
    total = 0
    for item in items:
        total = total + item.price
    return total
```

@FORMAT: detailed analysis, 3-5 suggestions
@FOCUS: performance, security, maintainability
@STYLE: concise, actionable
```

### Example 2: System Architecture Review

**Original (1,200 tokens):**

```
You are an expert software architect assistant, specialized in analyzing system design documents, identifying architectural patterns, potential issues, and suggesting refinements.

Context:
We have an AI platform service that handles ingestion of zipped bundles of quotes and documents. The pipeline is as follows:

1. A frontend triggers an API, providing a request payload including requestId, issueId, individualId, organizationId, employerId.

2. The Java service receives the payload, triggers an asynchronous call to a Python service via a Feign client. That service fetches files from S3, imprints pages, creates a zip bundle for both quotes and documents, and then returns a reference to the Java service.

3. The Java service passes the S3 reference to the downstream offer-level logic; the system persists the reference in MongoDB in the lumity-production-organizations database, collections employers and organizations.

4. Separately the system supports plan comparison: Plan JSON objects are generated by the orchestration service, then passed to the Python service for semantic comparison. The result is sent back to orchestration and surfaced in the UI.

5. The system uses FastAPI + Gunicorn: max requests 250, jitter 30s, five workers, keep-alive 5s, graceful-timeout 120s.

6. The development pipeline includes separate branches (develop, release, master), environments (develop, test, stage, production). Developers currently lack exclusive dev environment access; QA controls deployments. Dynamic environments once used are torn down after testing to save resources.

7. The system aims to support multi-pod safe FIFO-compliant SQS consumers, distributing messages based on MessageGroupId.

8. One of the current pain points: large context windows (both in terms of pipeline state and LLM prompt content) are increasing cost and latency; you are exploring a prompt-compression tool that takes large prompts (10,000 tokens) and reduces them to ~1,000 tokens while preserving clarity and semantics.

Instruction:
Given the above context, produce:
- A summary of the architectural strengths and design patterns currently used.
- A list of 5 key architectural risks or bottlenecks you observe (with brief explanations).
- For each risk, propose a concrete mitigation or improvement.
- Then, provide suggestions for how the prompt-compression tool (for large LLM context) could be integrated into this architecture: where it should live, how the data flow should change, what metrics you would track, and how you would validate its effectiveness.
- Finally, outline a minimal viable pilot plan (with phases, deliverables, success criteria) for building and testing this prompt-compression tool in your system.

Prompt:
"Please analyze the system described in the context and provide the requested outputs."
```

**Hieratic (390 tokens, 67.5% reduction):**

```hieratic
@HIERATIC v1.0

@ROLE[inline]
"Expert software architect: system design analysis, pattern recognition, risk assessment, optimization"

AI Platform Context:
- Flow: Frontend API → Java → Python (async, Feign) → S3 (zip bundles) → MongoDB
- IDs: requestId/issueId/individualId/organizationId/employerId
- DB: MongoDB `lumity-production-organizations` (employers, organizations)
- Plan comparison: Orchestration → Python semantic → UI
- Stack: FastAPI+Gunicorn (5 workers, max 250 req, 30s jitter, 120s timeout)
- Envs: develop/test/stage/prod; QA-controlled, dynamic teardown
- SQS: Multi-pod FIFO, MessageGroupId distribution
- Pain: Large contexts (10K tokens) → cost/latency → need 1K compression

@TASK
Analyze system, provide:
1. Architecture strengths + patterns
2. 5 risks/bottlenecks (with explanations)
3. Mitigation for each risk
4. Prompt compression integration:
   - Placement
   - Data flow changes
   - Metrics
   - Validation
5. MVP pilot plan: phases, deliverables, success criteria

@STYLE: structured, actionable, technical
@FORMAT: numbered sections
```

### Example 3: With Context Library

**Context Library (contexts.toml):**

```toml
[metadata]
version = "1.0"
format = "hieratic"
created_at = "2024-11-15T10:00:00Z"

[[patterns]]
id = "code_reviewer_001"
content = "Expert code reviewer: 10+ years, multiple languages, security-focused, performance optimization, best practices"
category = "role"
avg_tokens = 45

[[patterns]]
id = "review_examples_002"
content = "1. SQL injection fix: parameterized queries → 100% vuln elimination\n2. Memory leak: weak refs + profiling → 95% mem reduction\n3. Race condition: mutex + atomic ops → zero conflicts"
category = "examples"
avg_tokens = 85
```

**Hieratic Prompt:**

```hieratic
@HIERATIC v1.0
@CONTEXT: contexts.toml

@ROLE[code_reviewer_001]
@EXAMPLES[review_examples_002]

@TASK
Review this authentication middleware for security issues:

```javascript
function authMiddleware(req, res, next) {
  const token = req.headers.authorization;
  if (token == process.env.ADMIN_TOKEN) {
    req.user = { role: 'admin' };
    next();
  } else {
    res.status(401).send('Unauthorized');
  }
}
```

@FOCUS: security vulnerabilities, timing attacks, token handling
@FORMAT: issue list, severity ratings, fix recommendations
@STYLE: precise, security-focused
```

## File Extensions

- `.hieratic` - Primary extension for compressed prompts
- `.hrt` - Short alternative extension

## Context Library Format

Context libraries use TOML format:

```toml
[metadata]
version = "1.0"
format = "hieratic"
created_at = "2024-11-15T10:00:00Z"
source_directory = "./prompts"
total_patterns = 5

[[patterns]]
id = "unique_identifier"
content = "Full text content of the pattern"
frequency = 10  # Number of prompts using this pattern
avg_tokens = 150
category = "role|examples|constraints|instructions"
tags = ["tag1", "tag2"]
```

## LLM Compatibility

### Why LLMs Understand Hieratic

1. **Familiar Syntax**: Uses markdown-like structure seen in training data
2. **@-Directives**: Similar to Python decorators, mentions, annotations
3. **Structured Sections**: Clear semantic boundaries
4. **Natural Language**: Core content remains readable
5. **Self-Explanatory**: Format explains its purpose

### Tested With

- ✅ GPT-4 / GPT-4 Turbo
- ✅ Claude 3 (Opus, Sonnet, Haiku)
- ✅ Gemini Pro
- ⏳ Other models (testing ongoing)

## Best Practices

### When to Use Hieratic

✅ **Good Use Cases:**
- High-frequency prompts with stable structure
- Prompts with repeated role descriptions
- Multi-example prompts
- Prompts sent thousands of times per month
- Cost-sensitive applications

❌ **Not Ideal For:**
- One-off prompts
- Prompts < 500 tokens (minimal savings)
- Highly dynamic content
- Prompts where every word matters legally

### Compression Levels

**Light (30-50% reduction)**
- Minimal structural changes
- Abbreviations and compact lists
- Suitable for sensitive content

**Medium (50-70% reduction)**
- Structural compression with @-directives
- Context references where applicable
- Recommended for most use cases

**Aggressive (70-90% reduction)**
- Maximum use of references
- Extensive abbreviation
- Extractive compression
- Best for high-volume scenarios

### Validation

Always validate compressed prompts:

1. **Syntax Check**: Ensure valid Hieratic format
2. **Reference Resolution**: Verify context library references exist
3. **Token Count**: Confirm expected compression ratio
4. **Output Quality**: Use `--quality` flag to calculate quality metrics (semantic similarity, critical instruction preservation, information retention, structural integrity)

### Incremental Compression

For multi-turn conversations or continuously growing documents, Hieratic supports incremental compression:

**How it works:**
- Creates **anchors** (`@ANCHOR[id]`) at regular intervals
- Each anchor stores a compressed summary of content up to that point
- Recent content is marked with `@RECENT` and kept uncompressed
- Only new content (delta) is compressed in subsequent runs

**Example:**
```hieratic
@HIERATIC v1.0

@ANCHOR[550e8400-e29b-41d4-a716-446655440000]
[Compressed summary of previous conversation turns]

@RECENT
[Most recent, uncompressed content]
```

**Usage:**
```bash
# First compression (creates state file)
tokuin compress conversation-turn1.txt --incremental

# Subsequent turns (only compresses new content)
tokuin compress conversation-turn2.txt --incremental
tokuin compress conversation-turn3.txt --incremental
```

**Benefits:**
- ✅ Much faster for long conversations (no re-compression)
- ✅ Lower cost per compression operation
- ✅ Maintains context across multiple turns
- ✅ Ideal for agent workflows and chat sessions

## Version History

### v1.0 (Current)
- Initial specification
- Core @-directives (@HIERATIC, @ROLE, @EXAMPLES, @CONSTRAINTS, @TASK, @FOCUS, @STYLE, @FORMAT, @CONTEXT)
- Context library support
- Inline and referenced modes
- Incremental compression with @ANCHOR and @RECENT directives
- Semantic scoring support (heuristic, semantic, hybrid)
- Quality metrics integration

### Future Versions

**v1.1 (Planned)**
- `@LANG` directive for multi-language support
- `@INHERIT` for context inheritance
- `@OVERRIDE` for selective overrides

**v2.0 (Proposed)**
- Binary encoding option
- Compression streaming
- Multi-model targeting

## Contributing

The Hieratic format is designed to be community-driven. Suggestions for improvements, new directives, or use cases are welcome.

## License

The Hieratic format specification is released under MIT OR Apache-2.0 dual license, consistent with the Tokuin project.

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-02  
**Maintainer**: Tokuin Project  
**Repository**: https://github.com/nooscraft/tokuin

