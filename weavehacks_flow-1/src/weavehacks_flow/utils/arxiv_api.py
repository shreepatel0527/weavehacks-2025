#!/usr/bin/env python3
"""
Enhanced arXiv API client for scientific paper searches
Provides robust search functionality with error handling and data validation
"""

import feedparser
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import re
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ArxivPaper:
    """Data class representing an arXiv paper"""
    title: str
    authors: List[str]
    published: datetime
    summary: str
    pdf_link: str
    arxiv_id: str
    categories: List[str]
    doi: Optional[str] = None
    
    def __post_init__(self):
        """Clean and validate data after initialization"""
        self.title = self._clean_text(self.title)
        self.summary = self._clean_text(self.summary)
        
    def _clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace and newlines"""
        return re.sub(r'\s+', ' ', text.strip())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'title': self.title,
            'authors': self.authors,
            'published': self.published.isoformat(),
            'summary': self.summary,
            'pdf_link': self.pdf_link,
            'arxiv_id': self.arxiv_id,
            'categories': self.categories,
            'doi': self.doi
        }


class ArxivAPI:
    """Enhanced arXiv API client with robust error handling"""
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def __init__(self, timeout: int = 30):
        """Initialize the API client
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        
    def search(self, 
               query: str, 
               max_results: int = 10,
               start: int = 0,
               sort_by: str = "relevance",
               sort_order: str = "descending") -> List[ArxivPaper]:
        """Search arXiv for papers
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            start: Starting index for pagination
            sort_by: Sort criteria ('relevance', 'lastUpdatedDate', 'submittedDate')
            sort_order: Sort order ('ascending', 'descending')
            
        Returns:
            List of ArxivPaper objects
            
        Raises:
            ValueError: If parameters are invalid
            requests.RequestException: If API request fails
        """
        # Validate parameters
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if max_results <= 0 or max_results > 2000:
            raise ValueError("max_results must be between 1 and 2000")
            
        if start < 0:
            raise ValueError("start must be non-negative")
            
        valid_sort_by = ["relevance", "lastUpdatedDate", "submittedDate"]
        if sort_by not in valid_sort_by:
            raise ValueError(f"sort_by must be one of: {valid_sort_by}")
            
        valid_sort_order = ["ascending", "descending"]
        if sort_order not in valid_sort_order:
            raise ValueError(f"sort_order must be one of: {valid_sort_order}")
        
        # Build query URL
        params = {
            'search_query': f"all:{query.strip()}",
            'start': start,
            'max_results': max_results,
            'sortBy': sort_by,
            'sortOrder': sort_order
        }
        
        query_url = self._build_url(params)
        logger.info(f"Querying arXiv API: {query_url}")
        
        try:
            # Parse feed with error handling
            feed = feedparser.parse(query_url)
            
            # Check for feed errors
            if hasattr(feed, 'status'):
                try:
                    if isinstance(feed.status, int) and feed.status >= 400:
                        raise requests.RequestException(f"API returned status {feed.status}")
                except (TypeError, AttributeError):
                    # Skip status check if status is not a proper integer
                    pass
                
            if not feed.entries:
                logger.warning("No results found for query")
                return []
                
            # Convert entries to ArxivPaper objects
            papers = []
            for entry in feed.entries:
                try:
                    paper = self._parse_entry(entry)
                    papers.append(paper)
                except Exception as e:
                    logger.warning(f"Failed to parse entry: {e}")
                    continue
                    
            logger.info(f"Successfully parsed {len(papers)} papers")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            raise
    
    def search_by_category(self, category: str, max_results: int = 10) -> List[ArxivPaper]:
        """Search arXiv by subject category
        
        Args:
            category: arXiv category (e.g., 'physics.chem-ph', 'cond-mat.mtrl-sci')
            max_results: Maximum number of results
            
        Returns:
            List of ArxivPaper objects
        """
        query = f"cat:{category}"
        return self.search(query, max_results=max_results)
    
    def search_by_author(self, author: str, max_results: int = 10) -> List[ArxivPaper]:
        """Search arXiv by author name
        
        Args:
            author: Author name
            max_results: Maximum number of results
            
        Returns:
            List of ArxivPaper objects
        """
        query = f"au:{author}"
        return self.search(query, max_results=max_results)
    
    def _build_url(self, params: Dict[str, Any]) -> str:
        """Build query URL from parameters"""
        param_string = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self.BASE_URL}?{param_string}"
    
    def _parse_entry(self, entry: Any) -> ArxivPaper:
        """Parse a feed entry into an ArxivPaper object"""
        # Extract authors
        authors = []
        if hasattr(entry, 'authors'):
            authors = [author.name for author in entry.authors]
        elif hasattr(entry, 'author'):
            authors = [entry.author]
            
        # Extract publication date
        published = datetime.now()
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            published = datetime(*entry.published_parsed[:6])
        elif hasattr(entry, 'published'):
            try:
                published = datetime.fromisoformat(entry.published.replace('Z', '+00:00'))
            except:
                pass
                
        # Extract arXiv ID
        arxiv_id = ""
        if hasattr(entry, 'id'):
            arxiv_id = entry.id.split('/')[-1]
            
        # Extract categories
        categories = []
        if hasattr(entry, 'tags'):
            categories = [tag.term for tag in entry.tags]
            
        # Extract DOI if available
        doi = None
        if hasattr(entry, 'arxiv_doi'):
            doi = entry.arxiv_doi
            
        # Generate PDF link
        pdf_link = ""
        if hasattr(entry, 'link'):
            pdf_link = entry.link.replace('/abs/', '/pdf/') + '.pdf'
            
        return ArxivPaper(
            title=getattr(entry, 'title', ''),
            authors=authors,
            published=published,
            summary=getattr(entry, 'summary', ''),
            pdf_link=pdf_link,
            arxiv_id=arxiv_id,
            categories=categories,
            doi=doi
        )


def search_nanoparticle_papers(max_results: int = 10) -> List[ArxivPaper]:
    """Convenience function to search for nanoparticle-related papers
    
    Args:
        max_results: Maximum number of results
        
    Returns:
        List of ArxivPaper objects
    """
    client = ArxivAPI()
    query = "nanoparticle OR nanomaterial OR nanotechnology"
    return client.search(query, max_results=max_results)


def search_chemistry_papers(topic: str, max_results: int = 10) -> List[ArxivPaper]:
    """Search for chemistry papers on a specific topic
    
    Args:
        topic: Chemistry topic to search for
        max_results: Maximum number of results
        
    Returns:
        List of ArxivPaper objects
    """
    client = ArxivAPI()
    # Search in chemistry categories
    categories = ["physics.chem-ph", "cond-mat.mtrl-sci", "q-bio.BM"]
    
    papers = []
    for category in categories:
        try:
            results = client.search(f"cat:{category} AND {topic}", max_results=max_results//len(categories))
            papers.extend(results)
        except Exception as e:
            logger.warning(f"Failed to search category {category}: {e}")
            
    return papers[:max_results]


# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Search arXiv for scientific papers")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--max-results", type=int, default=10, help="Maximum results")
    parser.add_argument("--category", help="Search by category")
    parser.add_argument("--author", help="Search by author")
    parser.add_argument("--output", help="Output file (JSON format)")
    
    args = parser.parse_args()
    
    client = ArxivAPI()
    
    try:
        if args.category:
            papers = client.search_by_category(args.category, args.max_results)
        elif args.author:
            papers = client.search_by_author(args.author, args.max_results)
        else:
            papers = client.search(args.query, args.max_results)
            
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump([paper.to_dict() for paper in papers], f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            for i, paper in enumerate(papers, 1):
                print(f"{i}. {paper.title}")
                print(f"   Authors: {', '.join(paper.authors)}")
                print(f"   Published: {paper.published.strftime('%Y-%m-%d')}")
                print(f"   Categories: {', '.join(paper.categories)}")
                print(f"   PDF: {paper.pdf_link}")
                print(f"   Summary: {paper.summary[:200]}...")
                print()
                
    except Exception as e:
        logger.error(f"Search failed: {e}")
        exit(1)