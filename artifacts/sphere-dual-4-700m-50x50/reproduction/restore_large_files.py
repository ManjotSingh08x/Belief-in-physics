"""Restore byte-identical ZIPs and standalone HTML from the repository's <=8 MiB parts."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path


def within(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('path escapes packet directory')
    return path


def restore(root):
    entries = json.loads((root / 'LARGE_FILES.json').read_text())
    for name, entry in entries.items():
        target = within(root, name)
        size = entry['bytes']
        if not 0 < size < 100_000_000:
            raise ValueError('unexpected restored file size')
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError(f'existing file differs; preserving {name}')
            continue
        blocks = []
        for part, expected in entry['parts'].items():
            raw = within(root, part).read_bytes()
            if len(raw) > 8 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError(f'part integrity mismatch: {part}')
            blocks.append(raw)
        raw = b''.join(blocks)
        if entry['encoding'] == 'gzip':
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
                raw = stream.read(size + 1)
        elif entry['encoding'] != 'raw':
            raise ValueError('unknown encoding')
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError(f'restored file integrity mismatch: {name}')
        target.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation preserves an existing user file even if it appears during restoration.
        with target.open('xb') as output:
            output.write(raw)
        print('verified', name)
    print(f'{len(entries)} original files restored or already verified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    restore(parser.parse_args().root)
