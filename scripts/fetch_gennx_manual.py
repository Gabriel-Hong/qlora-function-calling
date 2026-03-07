"""
Fetch all GEN NX Online Manual articles from Zendesk and save as JSON files.
- _index.json: mapping of feature name -> article ID
- {article_id}.json: individual article content
"""

import urllib.request
import json
import re
import os
import time
import html
import sys

OUTPUT_DIR = r"C:\Users\hjm0830\OneDrive - MIDAS\바탕 화면\MIDAS\01_Study\PyTorch\API_Data\GENNX_Feature"
MANUAL_ARTICLE_ID = "49909210848537"
BASE_API = "https://support.midasuser.com/api/v2/help_center/en-us/articles"


def fetch_article(article_id):
    """Fetch a single article from Zendesk API."""
    url = f"{BASE_API}/{article_id}.json"
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))["article"]


def html_to_text(html_body):
    """Convert HTML to clean text, preserving structure."""
    text = html_body
    # Replace common block elements with newlines
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'</(p|div|li|tr|h[1-6])>', '\n', text)
    text = re.sub(r'<(p|div|li|tr|h[1-6])[^>]*>', '\n', text)
    # Remove all remaining tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode HTML entities
    text = html.unescape(text)
    # Clean whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n[ \t]+', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_sections(text):
    """Try to extract structured sections from the text."""
    sections = {}
    current_section = "overview"
    current_lines = []

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            current_lines.append("")
            continue
        # Detect section headers (common patterns in Zendesk articles)
        if line in ("Function", "Call", "Input", "Output", "Note", "Example",
                     "Examples", "Description", "Parameters", "Related Topics"):
            if current_lines:
                sections[current_section] = '\n'.join(current_lines).strip()
            current_section = line.lower()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_section] = '\n'.join(current_lines).strip()

    return sections


def extract_links_from_manual(body):
    """Extract all feature links from the manual index page."""
    # Find all links
    links = re.findall(
        r'<a[^>]*href=["\']([^"\']+support\.midasuser\.com/hc/en-us/articles/(\d+)[^"\']*)["\'][^>]*>(.*?)</a>',
        body, re.DOTALL
    )

    features = {}
    seen_ids = set()
    for full_url, article_id, link_text in links:
        text = re.sub(r'<[^>]+>', '', link_text).strip()
        # Remove trailing emoji/icons
        text = re.sub(r'\s*[\U0001f300-\U0001f9ff\u2600-\u27bf\ufffd]+\s*$', '', text).strip()
        text = re.sub(r'\s*$', '', text).strip()

        if not text or article_id == MANUAL_ARTICLE_ID:
            continue

        # Handle duplicates - append number
        key = text
        if article_id in seen_ids:
            continue
        if key in features and features[key] != article_id:
            key = f"{text} (2)"
        features[key] = article_id
        seen_ids.add(article_id)

    return features


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Fetch the main manual index page
    print("=" * 60)
    print("Step 1: Fetching GEN NX Online Manual index page...")
    print("=" * 60)
    manual = fetch_article(MANUAL_ARTICLE_ID)
    body = manual["body"]

    # Step 2: Extract all feature links
    features = extract_links_from_manual(body)
    print(f"Found {len(features)} unique features")

    # Save _index.json
    index_path = os.path.join(OUTPUT_DIR, "_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)
    print(f"Saved _index.json ({len(features)} entries)")

    # Step 3: Fetch each article
    print("\n" + "=" * 60)
    print("Step 2: Fetching individual articles...")
    print("=" * 60)

    total = len(features)
    success = 0
    errors = []

    for i, (name, article_id) in enumerate(features.items()):
        json_path = os.path.join(OUTPUT_DIR, f"{article_id}.json")

        # Skip if already fetched
        if os.path.exists(json_path):
            success += 1
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{total}] (skipped, already exists) {name}")
            continue

        try:
            article = fetch_article(article_id)
            article_body = article.get("body", "")
            text_content = html_to_text(article_body)
            sections = extract_sections(text_content)

            result = {
                "article_id": article_id,
                "title": article.get("title", name),
                "feature_name": name,
                "url": f"https://support.midasuser.com/hc/en-us/articles/{article_id}",
                "labels": article.get("label_names", []),
                "sections": sections,
                "full_text": text_content,
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            success += 1
            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"  [{i+1}/{total}] {name}")

            # Rate limiting - be gentle with the API
            time.sleep(0.3)

        except Exception as e:
            errors.append((name, article_id, str(e)))
            print(f"  [{i+1}/{total}] ERROR: {name} - {e}")
            time.sleep(1)

    # Summary
    print("\n" + "=" * 60)
    print(f"DONE: {success}/{total} articles fetched successfully")
    if errors:
        print(f"ERRORS: {len(errors)} articles failed:")
        for name, aid, err in errors:
            print(f"  - {name} ({aid}): {err}")
    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
