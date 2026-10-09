import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]

def download_data():
    source = json.loads((ROOT / 'data/source.json').read_text(encoding='utf-8'))
    path = ROOT / 'data/raw' / source['filename']
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = Request(source['url'], headers={'User-Agent': 'ML-portfolio-data-loader'})
        with urlopen(request, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != source['sha256']:
            raise ValueError('Dataset snapshot changed; review the source before updating its hash.')
        path.write_bytes(payload)
    if hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
        raise ValueError('Cached data does not match the recorded SHA-256.')
    return path
