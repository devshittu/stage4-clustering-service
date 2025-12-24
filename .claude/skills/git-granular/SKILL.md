# Granular Git Committer

Commits each changed file individually with comprehensive, detailed messages. Ensures self-explanatory Git history without bulk operations.

## Usage

/git-granular [optional: feature/fix description]

## Core Workflow

Begin by running 'git status' to list all changes, then proceed file by file.

For each file:

1. Identify the file path and name.
2. Use 'git diff' or equivalent to analyze the exact changes made to that file (e.g., added lines, removed lines, modified sections).
3. Provide a detailed, comprehensive explanation of what specifically changed in that file: describe the before-and-after state line-by-line or section-by-section where relevant, including code snippets if helpful for clarity.
4. Explain the reasons why those changes were made: reference the problem being solved, improvements in functionality, bug fixes, refactoring for better modularity/DRY/SOLID principles, performance optimizations, adherence to best practices (e.g., Python 3 standards, flake8/black formatting), or any other context from the project history.
5. Stage only that single file using 'git add [file_path]'.
6. Commit that file individually using 'git commit -m "[detailed commit message]"', where the commit message incorporates the full explanation from steps 3 and 4. Make the message structured, starting with a summary line (e.g., "Refactored [file] for improved modularity"), followed by a blank line, then the detailed body describing changes and reasons. Be as verbose and comprehensive as possible to ensure the commit history is self-explanatory.
7. After committing, confirm the commit hash and show the updated 'git status' to verify only that file was handled.

Process all changed files in sequence, one after another. If there are no changes, state that clearly and stop.

## Anti-Hallucination & Quality Safeguards

To avoid common LLM pitfalls such as hallucinations, overconfidence, inconsistency, verbosity, forgetting context, or introducing unintended biases/errors:

- **Always base explanations solely on actual 'git diff' output** or verifiable file changes—do not invent or assume changes that aren't present.
- **Maintain strict consistency with Git best practices**: use conventional commit message formats (e.g., starting with types like feat, fix, refactor), ensure messages are factual and neutral without subjective opinions or biases.
- **Be concise yet comprehensive** in explanations: limit to relevant details without unnecessary elaboration, but cover all changes thoroughly.
- **Cross-reference the full conversation history** and any provided context to ensure explanations align with project goals—do not forget or ignore prior instructions.
- **If uncertain about a change's reason** (e.g., no clear context provided), explicitly state "Reason inferred from code changes: [brief inference]" and avoid overconfident claims; prompt for user clarification if needed before committing.
- **Handle edge cases robustly**: for no changes, stop immediately; for large diffs, summarize key sections without omitting critical details; for binary files, describe metadata changes only.
- **Prevent chain-of-thought errors**: double-check each step internally before outputting (e.g., verify 'git add' succeeds, ensure commit message accurately reflects diff).
- **Do not introduce new code, files, or modifications**—only commit existing changes as-is.

## Critical Rules

- **Never bulk commit**: No `git add .` or `git commit -a` - process one file at a time
- **No watermarks or signatures**: Do not introduce any watermarks, signatures, or extraneous notes like "Claude" in your output—keep responses purely functional and code-terminal-like
- **Base on actual diffs only**: No assumptions about changes not visible in diff
- **One file = one commit**: Sequential processing only
- **Verify each step**: Show commit hash and git status after each commit

## Commit Message Format

```
type(scope): Brief summary (50 chars max)

- What changed (specific lines/functions/logic)
- Why it changed (bug fix/feature/refactor/optimization)
- How it changed (before → after state)
- Impact (performance/UX/architecture/maintainability)
- Related context (issue numbers, design decisions)
```

**Types**: feat, fix, refactor, docs, style, test, chore, perf

## Example Execution

```bash
# Step 1: Check status
git status

# Step 2: For modified src/services/resource_manager.py
git diff src/services/resource_manager.py

# Step 3: Detailed analysis
# Changed lines 45-67: Added context manager for Redis connections
# Changed lines 89-102: Implemented graceful shutdown with resource cleanup
# Added imports: contextlib.asynccontextmanager

# Step 4: Stage and commit
git add src/services/resource_manager.py
git commit -m "feat(resource-manager): Add async context manager for Redis lifecycle

- Implemented asynccontextmanager decorator for Redis connection handling
- Added graceful shutdown logic in __aexit__ to release connections properly
- Prevents resource leaks when service terminates unexpectedly
- Follows Python 3.10+ async best practices with proper typing
- Ensures zero connection pool exhaustion in production environments

Changes align with DRY principle by centralizing connection lifecycle.
Improves system reliability by guaranteeing cleanup on process termination."

# Step 5: Verify
# [commit hash displayed]
git status
# Shows remaining files to process

# Repeat for next file...
```

## Edge Cases

- **No changes**: Report "Working tree clean" and stop immediately
- **Binary files**: Describe metadata only (e.g., "Updated logo.png - 45KB → 52KB")
- **Large diffs (>200 lines)**: Summarize by major sections/functions without omitting critical logic changes
- **Merge conflicts**: Alert user and request manual resolution before proceeding
- **Untracked files**: Ask user if they should be added to .gitignore or staged

## Context Awareness

- Reference user's coding standards (SOLID, DRY, type hints, async patterns)
- Align commit messages with project architecture (microservices, pipelines, etc.)
- Mention performance impacts (10-30x improvements, caching benefits)
- Connect to broader system goals (production-ready, zero regressions)
