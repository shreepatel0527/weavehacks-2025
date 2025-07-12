#pip install feedparser

import feedparser

def search_arxiv(query="nanoparticle", max_results=10):
    base_url = "http://export.arxiv.org/api/query?"
    search_query = f"search_query=all:{query}&start=0&max_results={max_results}"
    query_url = base_url + search_query

    print(f"Querying arXiv API with URL:\n{query_url}\n")
    feed = feedparser.parse(query_url)

    print(f"Found {len(feed.entries)} results.\n")
    for i, entry in enumerate(feed.entries):
        print(f"{i+1}. Title: {entry.title}")
        print(f"   Authors: {', '.join(author.name for author in entry.authors)}")
        print(f"   Published: {entry.published}")
        print(f"   Summary: {entry.summary[:200]}...")  # Truncated summary
        print(f"   PDF Link: {entry.link.replace('abs', 'pdf')}\n")

# Run the search
search_arxiv("nanoparticle")
