# Public Work Trail

This repository uses three complementary records of ongoing work:

1. Small, meaningful commits show when implementation changed.
2. `WORKLOG.md` explains the evidence, limitation, and next question behind each milestone.
3. GitHub Actions runs the focused test suite after every push to `main` and on every pull request.

## End of session checklist

1. Run `python -m unittest discover -s tests -v`.
2. Add a dated entry to `WORKLOG.md` for meaningful coding, data, training, or evaluation work.
3. Review `git diff` for secrets, raw replay captures, model binaries, and unrelated files.
4. Commit with a concise description of the result.
5. Push the commit so the activity and verification run appear on GitHub.

Public history should reflect real work. Avoid empty commits or artificial activity solely to fill a contribution graph.
