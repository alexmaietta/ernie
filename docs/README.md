# Docs

The schema reference pages and the scene files' JSON Schema are generated from the schema registry by a build of `ernie`, before the site is built:

```sh
ernie schema --json-schema docs/schema/ernie-scene.schema.json --markdown docs/reference
python -m pip install -r docs/requirements.txt
mkdocs serve # live preview at http://127.0.0.1:8000
mkdocs build # static site in site/
```
