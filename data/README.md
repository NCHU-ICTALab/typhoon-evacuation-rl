# Local data

This directory intentionally contains no dataset in Git.

Expected local file: `data/ua1008l.sqlite`. Commands also accept an explicit `--db` path. The loader
opens SQLite with `mode=ro`; training variations are generated in memory and are not appended here.

Before publishing any replacement dataset, document its source, license, fields, anonymization status,
and checksum. Do not commit operational VTS lists or personally/commercially sensitive records.
