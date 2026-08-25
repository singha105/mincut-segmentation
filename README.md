# mincut-segmentation

Built an interactive image segmentation tool using max-flow/min-cut graph algorithms, achieving segmentation on user-marked images in under 2s.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Test

```bash
pytest
```

## Project layout

- `maxflow/` — Edmonds-Karp max-flow implementation and image-to-graph construction
- `segmentation/` — orchestration layer that turns a graph min-cut into a foreground/background mask
- `app.py` — Streamlit UI for marking foreground/background and viewing results
- `sample_images/` — small synthetic images for manual testing and demos
- `tests/` — automated tests
