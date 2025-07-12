import requests
import csv

def search_google_scholar(query="nanoparticle", api_key="apikey", num_results=10):
    url = "https://serpapi.com/search"
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": api_key,
        "num": num_results
    }

    print(f"Querying Google Scholar via SerpAPI for: '{query}'...")
    response = requests.get(url, params=params)
    data = response.json()

    results = data.get("organic_results", [])
    print(f"Found {len(results)} results.\n")

    # Prepare CSV export
    csv_file = "google_scholar_nanoparticle.csv"
    with open(csv_file, mode="w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Title", "Authors/Info", "Snippet", "Link", "Citation Link"])

        for i, result in enumerate(results):
            title = result.get("title", "")
            link = result.get("link", "")
            snippet = result.get("snippet", "")
            authors = result.get("publication_info", {}).get("summary", "")
            citation_link = result.get("inline_links", {}).get("cited_by", {}).get("link", "")

            writer.writerow([title, authors, snippet, link, citation_link])

            # Print preview
            print(f"{i+1}. Title: {title}")
            print(f"   Authors/Info: {authors}")
            print(f"   Snippet: {snippet[:200]}...")
            print(f"   Link: {link}")
            print(f"   Citation Link: {citation_link}\n")

    print(f"\n✅ Results saved to '{csv_file}'.")

# Run the search and export
search_google_scholar("nanoparticle")
