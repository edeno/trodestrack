I'm working on the trodestrack project.

Start now by reading the files and telling me which task you'll work on first.

Your workflow MUST be:

1. First, read these files IN ORDER:
   - CLAUDE.md (implementation guide)
   - PRD.md (product requirements)
   - TASKS.md (current tasks)

2. Find the FIRST unchecked [ ] task in TASKS.md

3. For EVERY feature, follow TDD:
   a. Create the TEST file first
   b. Run the test and verify it FAILS
   c. Only then create the implementation
   d. Run test until it PASSES
   e. Apply review agents (code-reviewer, ux-reviewer)
   f. Refactor for clarity and efficiency based on feedback
   g. Add/Update docstrings and types.

4. Update TASKS.md checkboxes as you complete items.

5. Update SCRATCHPAD.md with notes and CHANGELOG.md with changes.

6. Commit frequently with messages like "feat(F24): implement error handling"

Do not change tests or skip tests to match broken code. Ask permission to change requirements if needed.
