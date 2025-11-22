# Update .claude/AGENTS.md with PR Description File Workflow

## Summary

Adds workflow instructions to `.claude/AGENTS.md` for managing PR description files. This solves the copy-paste difficulty when using Claude Code in the web interface.

## Changes

### New Section: "Pull Request Description Workflow"

**When creating a PR:**
1. Create `EPIC{N}_PR.md` at repo root with full PR description
2. Commit and push with feature branch
3. Provide GitHub links to user:
   - PR creation: `https://github.com/{owner}/{repo}/compare/main...{branch}?expand=1`
   - Raw file: `https://github.com/{owner}/{repo}/blob/{branch}/EPIC{N}_PR.md`
4. User clicks "Raw" button on GitHub to easily copy description

**After PR merged:**
1. Remove `EPIC{N}_PR.md` file
2. Keeps repo clean long-term
3. PR description preserved in GitHub PR

### Benefits

✅ Solves copy-paste difficulty in web-based Claude Code interface
✅ Description viewable directly on GitHub
✅ Easy access via "Raw" button
✅ Version controlled with branch
✅ No clutter in repo after merge

### Example Workflow

```bash
# Creating PR
echo "PR content..." > EPIC4_PR.md
git add EPIC4_PR.md
git commit -m "docs: add Epic 4 PR description"
git push

# After PR merged
git rm EPIC4_PR.md
git commit -m "docs: remove PR description file after merge"
git push
```

## Implementation Notes

- Placed after "Pull Request Format" section in AGENTS.md
- Clear step-by-step instructions for both creation and cleanup
- Explains the "why" behind the pattern
- Provides concrete examples with proper file naming

## Related

- Addresses usability issue with Claude Code web interface
- Establishes pattern for future Epic PRs
- Complements existing PR format guidelines
