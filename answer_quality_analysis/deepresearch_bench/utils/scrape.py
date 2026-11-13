import multiprocessing
import json
import os
import time
import argparse
import platform
from tqdm import tqdm
from .api import scrape_url
from .io_utils import load_jsonl


def scrape(citation_url):
    retries = 0
    while retries < 3:
        result = scrape_url(citation_url)
        retries += 1
        if 'error' in result:
            time.sleep(1)
        else:
            break

    url_content = None
    if 'error' not in result:
        title = result.get('title', '')
        content = result.get('content', '')
        description = result.get('description', '')

        url_content = f"{title}\n\n{description}\n\n{content}"
    else:
        url_content = f"scrape failed: {result.get('error', 'unknown error')}"
    # print(f"url_content: {url_content}")
    return {
        'url': citation_url,
        'url_content': url_content
    } 


if __name__ == '__main__':
    if platform.system() == 'Darwin':  
        try:
            multiprocessing.set_start_method('spawn')
        except RuntimeError:
            pass
    else: 
        try:
            multiprocessing.set_start_method('fork')
        except RuntimeError:
            pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--raw_data_path", type=str, required=True)
    parser.add_argument("--n_total_process", type=int, default=1)
    args = parser.parse_args()
    
    output_path = args.output_path
    
    # initialize variables
    raw_data = []
    data_to_process = []
    processed = []
    
    try:
        raw_data = load_jsonl(args.raw_data_path)
        
        if os.path.exists(output_path):
            processed = [d['id'] for d in load_jsonl(output_path)]
            data_to_process = [d for d in raw_data if d['id'] not in processed]
        else:
            data_to_process = raw_data
    except:
        import sys
        print(f"cannot process file {args.raw_data_path}")
        sys.exit(f'{args.raw_data_path} has not been processed yet...')
    
    print(f"processing {len(data_to_process)} instances...")

    # Scrape every unique URL across the whole batch through a single pool
    # instead of re-spinning a pool per article: per-article batches left
    # most of n_total_process idle whenever an article had few citations,
    # and one slow/dead URL stalled that article's whole (small) batch.
    url_cache_path = output_path + ".url_cache.jsonl"
    url_content_map = {}
    if os.path.exists(url_cache_path):
        for row in load_jsonl(url_cache_path):
            url_content_map[row['url']] = row['url_content']

    all_urls = set()
    for d in data_to_process:
        for k, v in d['citations_deduped'].items():
            if ('url_content' not in v or not v['url_content']) and k not in url_content_map:
                all_urls.add(k)
    all_urls = list(all_urls)

    print(f"scraping {len(all_urls)} unique urls ({len(url_content_map)} already cached)...")

    cache_file = open(url_cache_path, 'a+', encoding='utf-8')
    try:
        n_total_process = min(args.n_total_process, len(all_urls)) if all_urls else 0
        if n_total_process == 1:
            result_iter = (scrape(url) for url in all_urls)
        elif n_total_process > 1:
            pool = multiprocessing.Pool(processes=n_total_process)
            result_iter = pool.imap_unordered(scrape, all_urls)
        else:
            result_iter = iter([])

        for res in tqdm(result_iter, total=len(all_urls)):
            url_content_map[res['url']] = res['url_content']
            cache_file.write(json.dumps(res, ensure_ascii=False) + "\n")
            cache_file.flush()

        if n_total_process > 1:
            pool.close()
            pool.join()
    finally:
        cache_file.close()

    for d in data_to_process:
        for k, v in d['citations_deduped'].items():
            if k in url_content_map:
                v['url_content'] = url_content_map[k]

        # write the updated data to the output file
        with open(output_path, 'a+', encoding='utf-8') as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
