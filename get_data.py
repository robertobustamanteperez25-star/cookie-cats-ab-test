"""Download the Cookie Cats A/B test dataset into data/raw/."""
from pathlib import Path
from urllib.request import urlretrieve

URL = ("https://raw.githubusercontent.com/ryanschaub/"
       "Mobile-Games-A-B-Testing-with-Cookie-Cats/master/cookie_cats.csv")
out = Path(__file__).resolve().parent / "data" / "raw" / "cookie_cats.csv"
out.parent.mkdir(parents=True, exist_ok=True)
urlretrieve(URL, out)
print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
