"""Import verified arXiv metadata and retain the abstracts used for review."""

import json
import time
import unicodedata
from pathlib import Path

import requests
from bs4 import BeautifulSoup


PAPERS = [
    ('h2o', '2306.14048', 'Heavy-Hitter'),
    ('scissorhands', '2305.17118', 'Scissorhands'),
    ('snapkv', '2404.14469', 'SnapKV'),
    ('pyramidkv', '2406.02069', 'PyramidKV'),
    ('quest', '2406.10774', 'Quest'),
    ('kivi', '2402.02750', 'KIVI'),
    ('kvquant', '2401.18079', 'KVQuant'),
    ('eigenattention', '2408.05646', 'Eigen'),
    ('palu', '2407.21118', 'Palu'),
    ('puzzlekv', '2608.23843', 'PuzzleKV'),
    ('streamingllm', '2309.17453', 'Streaming'),
    ('duoattention', '2410.10819', 'DuoAttention'),
    ('gear', '2403.05527', 'GEAR'),
    ('minicache', '2405.14366', 'MiniCache'),
    ('shadowkv', '2410.21465', 'ShadowKV'),
    ('xkv', '2503.18893', 'xKV'),
    ('transformer', '1706.03762', 'Attention Is All'),
    ('rope', '2104.09864', 'RoFormer'),
    ('mqa', '1911.02150', 'Fast Transformer Decoding'),
    ('gqa', '2305.13245', 'GQA'),
    ('deepseekv2', '2405.04434', 'DeepSeek-V2'),
    ('pagedattention', '2309.06180', 'PagedAttention'),
    ('flashattention', '2205.14135', 'FlashAttention'),
    ('flashattention2', '2307.08691', 'FlashAttention-2'),
    ('ruler', '2404.06654', 'RULER'),
    ('longbench', '2308.14508', 'LongBench'),
    ('adakv', '2407.11550', 'Ada-KV'),
    ('think', '2407.21018', 'ThinK'),
    ('asvd', '2312.05821', 'ASVD'),
    ('svdllm', '2403.07378', 'SVD-LLM'),
]


def latex(value):
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode()
    return value.replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')


def main():
    root = Path(__file__).resolve().parent.parent
    cache_path = root / 'reference_sources.json'
    records = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for key, identifier, expected in PAPERS:
        if key not in records:
            url = 'https://arxiv.org/abs/' + identifier
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.select_one('meta[name=citation_title]')['content']
            assert expected.lower() in title.lower(), (key, title)
            records[key] = {
                'title': title,
                'authors': [m['content'] for m in soup.select('meta[name=citation_author]')],
                'date': soup.select_one('meta[name=citation_date]')['content'],
                'arxiv': identifier,
                'source': url,
                'abstract': soup.select_one('blockquote.abstract').get_text(' ', strip=True),
            }
            cache_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + '\n')
            time.sleep(0.3)
        print(key, records[key]['title'], flush=True)
    entries = []
    for key, _, _ in PAPERS:
        record = records[key]
        authors = record['authors']
        if len(authors) > 15:
            authors = authors[:12] + ['others']
        entries.append(
            '@article{' + key + ',\n'
            '  title = {{' + latex(record['title']) + '}},\n'
            '  author = {' + ' and '.join(latex(a) for a in authors) + '},\n'
            '  journal = {arXiv preprint arXiv:' + record['arxiv'] + '},\n'
            '  year = {' + record['date'][:4] + '},\n'
            '  doi = {10.48550/arXiv.' + record['arxiv'] + '}\n}\n'
        )
    (root / 'related_work.bib').write_text('\n'.join(entries))


if __name__ == '__main__':
    main()
