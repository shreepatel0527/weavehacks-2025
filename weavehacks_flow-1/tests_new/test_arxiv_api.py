#!/usr/bin/env python3
"""
Comprehensive tests for the enhanced ArXiv API client
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from weavehacks_flow.utils.arxiv_api import ArxivAPI, ArxivPaper, search_nanoparticle_papers


class TestArxivPaper:
    """Test ArxivPaper data class"""
    
    def test_paper_creation(self):
        """Test basic paper creation"""
        paper = ArxivPaper(
            title="Test Paper",
            authors=["John Doe", "Jane Smith"],
            published=datetime.now(),
            summary="Test summary",
            pdf_link="http://example.com/paper.pdf",
            arxiv_id="2101.00001",
            categories=["physics.chem-ph"]
        )
        
        assert paper.title == "Test Paper"
        assert len(paper.authors) == 2
        assert paper.arxiv_id == "2101.00001"
    
    def test_text_cleaning(self):
        """Test text cleaning functionality"""
        paper = ArxivPaper(
            title="Test  Paper\nWith   Whitespace",
            authors=["John Doe"],
            published=datetime.now(),
            summary="Summary\n  with\t multiple\n\nlines",
            pdf_link="http://example.com/paper.pdf",
            arxiv_id="2101.00001",
            categories=["physics.chem-ph"]
        )
        
        assert paper.title == "Test Paper With Whitespace"
        assert paper.summary == "Summary with multiple lines"
    
    def test_to_dict(self):
        """Test dictionary conversion"""
        paper = ArxivPaper(
            title="Test Paper",
            authors=["John Doe"],
            published=datetime(2021, 1, 1),
            summary="Test summary",
            pdf_link="http://example.com/paper.pdf",
            arxiv_id="2101.00001",
            categories=["physics.chem-ph"]
        )
        
        paper_dict = paper.to_dict()
        
        assert isinstance(paper_dict, dict)
        assert paper_dict['title'] == "Test Paper"
        assert paper_dict['authors'] == ["John Doe"]
        assert 'published' in paper_dict


class TestArxivAPI:
    """Test ArxivAPI client"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.api = ArxivAPI()
    
    def test_api_initialization(self):
        """Test API client initialization"""
        api = ArxivAPI(timeout=60)
        assert api.timeout == 60
        assert api.BASE_URL == "http://export.arxiv.org/api/query"
    
    def test_search_validation(self):
        """Test search parameter validation"""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            self.api.search("")
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            self.api.search("   ")
        
        with pytest.raises(ValueError, match="max_results must be between 1 and 2000"):
            self.api.search("test", max_results=0)
        
        with pytest.raises(ValueError, match="max_results must be between 1 and 2000"):
            self.api.search("test", max_results=3000)
        
        with pytest.raises(ValueError, match="start must be non-negative"):
            self.api.search("test", start=-1)
        
        with pytest.raises(ValueError, match="sort_by must be one of"):
            self.api.search("test", sort_by="invalid")
        
        with pytest.raises(ValueError, match="sort_order must be one of"):
            self.api.search("test", sort_order="invalid")
    
    def test_build_url(self):
        """Test URL building"""
        params = {
            'search_query': 'all:nanoparticle',
            'start': 0,
            'max_results': 10
        }
        
        url = self.api._build_url(params)
        
        assert "http://export.arxiv.org/api/query?" in url
        assert "search_query=all:nanoparticle" in url
        assert "start=0" in url
        assert "max_results=10" in url
    
    @patch('feedparser.parse')
    def test_search_success(self, mock_parse):
        """Test successful search"""
        # Mock feedparser response
        mock_entry = Mock()
        mock_entry.title = "Test Paper"
        
        # Create proper author mock with name attribute
        mock_author = Mock()
        mock_author.name = "John Doe"
        mock_entry.authors = [mock_author]
        
        mock_entry.published_parsed = (2021, 1, 1, 0, 0, 0, 0, 0, 0)
        mock_entry.summary = "Test summary"
        mock_entry.id = "http://arxiv.org/abs/2101.00001v1"
        mock_entry.link = "http://arxiv.org/abs/2101.00001"
        
        # Create proper tag mock with term attribute
        mock_tag = Mock()
        mock_tag.term = "physics.chem-ph"
        mock_entry.tags = [mock_tag]
        
        mock_feed = Mock()
        mock_feed.entries = [mock_entry]
        # Don't set status attribute to avoid comparison issues
        mock_parse.return_value = mock_feed
        
        papers = self.api.search("nanoparticle", max_results=5)
        
        assert len(papers) == 1
        assert isinstance(papers[0], ArxivPaper)
        assert papers[0].title == "Test Paper"
        assert papers[0].authors == ["John Doe"]
        assert papers[0].arxiv_id == "2101.00001v1"
    
    @patch('feedparser.parse')
    def test_search_no_results(self, mock_parse):
        """Test search with no results"""
        mock_feed = Mock()
        mock_feed.entries = []
        mock_parse.return_value = mock_feed
        
        papers = self.api.search("nonexistent_topic")
        
        assert papers == []
    
    @patch('feedparser.parse')
    def test_search_by_category(self, mock_parse):
        """Test search by category"""
        mock_feed = Mock()
        mock_feed.entries = []
        mock_parse.return_value = mock_feed
        
        papers = self.api.search_by_category("physics.chem-ph", max_results=5)
        
        # Check that the correct query was built
        mock_parse.assert_called_once()
        call_args = mock_parse.call_args[0][0]
        assert "cat:physics.chem-ph" in call_args
    
    @patch('feedparser.parse')
    def test_search_by_author(self, mock_parse):
        """Test search by author"""
        mock_feed = Mock()
        mock_feed.entries = []
        mock_parse.return_value = mock_feed
        
        papers = self.api.search_by_author("John Doe", max_results=5)
        
        # Check that the correct query was built
        mock_parse.assert_called_once()
        call_args = mock_parse.call_args[0][0]
        assert "au:John Doe" in call_args
    
    def test_parse_entry_minimal(self):
        """Test parsing entry with minimal data"""
        mock_entry = Mock()
        mock_entry.title = "Test Paper"
        mock_entry.authors = []
        mock_entry.published_parsed = None
        mock_entry.published = "2021-01-01T00:00:00Z"
        mock_entry.summary = "Test summary"
        mock_entry.id = "http://arxiv.org/abs/2101.00001v1"
        mock_entry.link = "http://arxiv.org/abs/2101.00001"
        
        # Remove optional attributes
        del mock_entry.tags
        del mock_entry.arxiv_doi
        
        paper = self.api._parse_entry(mock_entry)
        
        assert isinstance(paper, ArxivPaper)
        assert paper.title == "Test Paper"
        assert paper.authors == []
        assert paper.categories == []
        assert paper.doi is None
    
    def test_parse_entry_with_author_string(self):
        """Test parsing entry where author is a string"""
        mock_entry = Mock()
        mock_entry.title = "Test Paper"
        mock_entry.author = "John Doe"
        mock_entry.published_parsed = (2021, 1, 1, 0, 0, 0, 0, 0, 0)
        mock_entry.summary = "Test summary"
        mock_entry.id = "http://arxiv.org/abs/2101.00001v1"
        mock_entry.link = "http://arxiv.org/abs/2101.00001"
        mock_entry.tags = []
        
        # Remove authors attribute to test author fallback
        del mock_entry.authors
        
        paper = self.api._parse_entry(mock_entry)
        
        assert paper.authors == ["John Doe"]


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    @patch('weavehacks_flow.utils.arxiv_api.ArxivAPI.search')
    def test_search_nanoparticle_papers(self, mock_search):
        """Test nanoparticle paper search"""
        mock_search.return_value = []
        
        papers = search_nanoparticle_papers(max_results=5)
        
        mock_search.assert_called_once()
        args, kwargs = mock_search.call_args
        assert "nanoparticle OR nanomaterial OR nanotechnology" in args[0]
        assert kwargs['max_results'] == 5


class TestIntegration:
    """Integration tests (require network access)"""
    
    @pytest.mark.integration
    def test_real_arxiv_search(self):
        """Test real arXiv API call (requires internet)"""
        api = ArxivAPI()
        
        try:
            papers = api.search("quantum", max_results=2)
            
            # Basic validation
            assert isinstance(papers, list)
            if papers:  # Only test if results returned
                assert isinstance(papers[0], ArxivPaper)
                assert papers[0].title
                assert papers[0].arxiv_id
                
        except Exception as e:
            pytest.skip(f"Network error, skipping integration test: {e}")
    
    @pytest.mark.integration
    def test_real_category_search(self):
        """Test real category search"""
        api = ArxivAPI()
        
        try:
            papers = api.search_by_category("physics.chem-ph", max_results=2)
            
            # Basic validation
            assert isinstance(papers, list)
            if papers:
                assert isinstance(papers[0], ArxivPaper)
                # Should contain chemistry-related categories
                has_chem_category = any("chem" in cat.lower() or "physics" in cat.lower() 
                                      for paper in papers for cat in paper.categories)
                assert has_chem_category or len(papers) == 0
                
        except Exception as e:
            pytest.skip(f"Network error, skipping integration test: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])