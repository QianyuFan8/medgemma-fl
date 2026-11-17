# Contribution rules

- All documentation committed to this repository must be written in English.
- Chinese documentation may be kept locally, but must remain untracked and ignored. Do not force-add it.
- Never commit patient data, credentials, model weights, private handoff bundles, or generated experiment results.
- After staging changes, run `python3 scripts/check_english_docs.py` and inspect the staged file list before committing or pushing.
- GitHub Actions checks English-only documentation on pushes and pull requests. This guard does not replace content review.

Enable the local pre-commit guard after cloning:

```bash
git config core.hooksPath .githooks
```

The configured local hook checks staged document contents before creating a commit.
