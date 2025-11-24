import unittest
import logging
from src.keyword_code.rag.chunking import SentenceChunker

# Configure logging to see debug output
logging.basicConfig(level=logging.INFO)

class TestChunkingStructure(unittest.TestCase):
    def test_chunking_splits_on_article(self):
        # Use a small chunk size to ensure grouping would happen if not for the split
        chunker = SentenceChunker(sentences_per_chunk=10)
        
        # Text with embedded Article V
        # Note: spaCy needs punctuation to detect sentences well.
        text = (
            "Section 4.03. Conditions for IFC Benefit. "
            "The conditions in Section 4.01 are for the benefit of IFC and may be waived only by IFC in its sole discretion. "
            "ARTICLE V PARTICULAR COVENANTS. "
            "Section 5.01. Affirmative Covenants. "
            "Unless IFC otherwise agrees, the Borrower shall maintain its corporate existence."
        )
        
        chunks = chunker.create_chunks(text)
        
        print(f"\nGenerated {len(chunks)} chunks:")
        found_article_start = False
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i}: {chunk['text']}")
            if chunk['text'].strip().startswith("ARTICLE V"):
                found_article_start = True
                
        self.assertTrue(found_article_start, "ARTICLE V should start a new chunk")

if __name__ == '__main__':
    unittest.main()
