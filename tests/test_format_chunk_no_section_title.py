import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from src.keyword_code.rag.retrieval import format_chunk_with_metadata


def test_excludes_section_title():
    chunk = {
        'text': 'This is the body text of the chunk.',
        'metadata': {
            'article_title': 'PARTICULAR COVENANTS',
            'article_number': 'V',
            'section_number': '5.01',
            'section_title': 'Affirmative Covenants'
        }
    }
    out = format_chunk_with_metadata(chunk)
    print('Formatted:', out)
    assert 'Affirmative Covenants' not in out
    assert 'Section 5.01' in out

if __name__ == '__main__':
    test_excludes_section_title()
    print('OK')
