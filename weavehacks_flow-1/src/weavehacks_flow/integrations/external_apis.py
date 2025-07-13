"""
Integration module for external scientific APIs
"""
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(parent_dir))

import weave
import wandb

class ExternalAPIIntegration:
    """Wrapper for external scientific API integrations"""
    
    def __init__(self):
        self.arxiv_api = None
        self.esm_api = None
        self.gs_api = None
        self._load_apis()
    
    def _load_apis(self):
        """Load external API modules"""
        try:
            import arxiv_api
            self.arxiv_api = arxiv_api
        except ImportError:
            print("Warning: arxiv_api not available")
        
        try:
            import esm_api
            self.esm_api = esm_api
        except ImportError:
            print("Warning: esm_api not available")
        
        try:
            import gs_api
            self.gs_api = gs_api
        except ImportError:
            print("Warning: gs_api not available")
    
    @weave.op()
    def search_arxiv(self, query, max_results=5):
        """Search arXiv for scientific papers"""
        if not self.arxiv_api:
            return {"error": "arXiv API not available"}
        
        try:
            # Assuming the arxiv_api has a search function
            results = self.arxiv_api.search(query, max_results=max_results)
            wandb.log({
                'api_call': {
                    'api': 'arxiv',
                    'query': query,
                    'results_count': len(results) if results else 0
                }
            })
            return results
        except Exception as e:
            return {"error": str(e)}
    
    @weave.op()
    def get_protein_embedding(self, sequence):
        """Get protein embeddings using ESM API"""
        if not self.esm_api:
            return {"error": "ESM API not available"}
        
        try:
            # Assuming the esm_api has an embedding function
            embedding = self.esm_api.get_embedding(sequence)
            wandb.log({
                'api_call': {
                    'api': 'esm',
                    'sequence_length': len(sequence),
                    'embedding_generated': embedding is not None
                }
            })
            return embedding
        except Exception as e:
            return {"error": str(e)}
    
    @weave.op()
    def search_google_scholar(self, query, num_results=10):
        """Search Google Scholar for academic papers"""
        if not self.gs_api:
            return {"error": "Google Scholar API not available"}
        
        try:
            # Assuming the gs_api has a search function
            results = self.gs_api.search(query, num_results=num_results)
            wandb.log({
                'api_call': {
                    'api': 'google_scholar',
                    'query': query,
                    'results_count': len(results) if results else 0
                }
            })
            return results
        except Exception as e:
            return {"error": str(e)}
    
    @weave.op()
    def search_nanoparticle_literature(self, material="gold", application="cancer"):
        """Specialized search for nanoparticle literature"""
        queries = [
            f"{material} nanoparticles {application} therapy",
            f"Au25 clusters {application}",
            f"{material} nanoparticle synthesis protocol"
        ]
        
        all_results = {
            'arxiv': [],
            'google_scholar': []
        }
        
        for query in queries:
            # Search arXiv
            arxiv_results = self.search_arxiv(query, max_results=3)
            if isinstance(arxiv_results, list):
                all_results['arxiv'].extend(arxiv_results)
            
            # Search Google Scholar
            gs_results = self.search_google_scholar(query, num_results=5)
            if isinstance(gs_results, list):
                all_results['google_scholar'].extend(gs_results)
        
        return all_results
    
    @weave.op()
    def get_safety_data(self, chemical_name):
        """Get safety data for a chemical (mock implementation)"""
        # This would integrate with a real chemical safety database
        safety_data = {
            "HAuCl4": {
                "hazards": ["Corrosive", "Oxidizing"],
                "handling": "Use in fume hood with appropriate PPE",
                "disposal": "Collect for precious metal recovery"
            },
            "NaBH4": {
                "hazards": ["Flammable", "Water reactive"],
                "handling": "Keep dry, use in well-ventilated area",
                "disposal": "Neutralize with dilute acid before disposal"
            },
            "PhCH2CH2SH": {
                "hazards": ["Malodorous", "Irritant"],
                "handling": "Use in fume hood",
                "disposal": "Oxidize before disposal"
            },
            "toluene": {
                "hazards": ["Flammable", "Toxic"],
                "handling": "Use in fume hood, avoid skin contact",
                "disposal": "Collect as organic waste"
            }
        }
        
        chemical_key = chemical_name.replace("₄", "4").replace("₂", "2")
        
        if chemical_key in safety_data:
            wandb.log({
                'safety_lookup': {
                    'chemical': chemical_name,
                    'found': True
                }
            })
            return safety_data[chemical_key]
        else:
            wandb.log({
                'safety_lookup': {
                    'chemical': chemical_name,
                    'found': False
                }
            })
            return {
                "hazards": ["Unknown - use caution"],
                "handling": "Consult MSDS before use",
                "disposal": "Consult safety officer"
            }