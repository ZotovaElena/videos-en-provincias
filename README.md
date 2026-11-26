# vmap

Map of the most-viewed official music video filmed in each Spanish province / island (61 units).

```bash
conda activate my_analysis
pip install -e . --no-deps      # once
cp .env.example .env            # fill YOUTUBE_KEY, CONTACT_EMAIL
vmap check
pytest
```
