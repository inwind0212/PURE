# Release validation

Validation date: 2026-10-01

The following checks were run against the main-text release validated against the manuscript:

- Python syntax compilation for every file under code/: passed.
- SHA256 manifest validation before final packaging: passed for all 65 listed files.
- Figure 1 regeneration from the released country-seed table: passed.
- Figure 2 regeneration from the four released Figure 2 tables: passed.
- Figure 3 regeneration from the released transfer tables and illustration: passed.
- Scan of released code for local absolute paths: no matches.
- Scan for obsolete 13-city, 384-dimensional text and 100-epoch main-training descriptions: no manuscript-protocol conflicts found.

Figure 1 validation used the Natural Earth boundary file documented by the
project. Figure 3 validation used the pinned pure Python environment, including
statsmodels. All generated validation outputs were written outside the release
directory.

The final manifest was regenerated after documentation and release cleanup. A software license remains a rights-holder decision and is not inferred
by this package.
