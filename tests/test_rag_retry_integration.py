import base64
import importlib.util
import sys
import types
import unittest

import streamlit as st


class TestRagRetryIntegration(unittest.TestCase):
    def setUp(self):
        # Reset session state
        st.session_state.clear()

    def _load_pages_module(self):
        # Load pages/1_📄_CNT_space.py as a module dynamically
        pages_path = 'pages/1_📄_CNT_space.py'
        spec = importlib.util.spec_from_file_location('cnt_space_page', pages_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules['cnt_space_page'] = module
        spec.loader.exec_module(module)  # type: ignore
        return module

    def test_retry_updates_session_state_with_new_results(self):
        # Arrange: minimal analysis result and preprocessed data
        filename = 'dummy.pdf'
        dummy_pdf_bytes = b'%PDF-1.4 dummy'
        annotated_pdf_b64 = base64.b64encode(dummy_pdf_bytes).decode('utf-8')

        st.session_state.analysis_results = [
            {
                'filename': filename,
                'annotated_pdf': annotated_pdf_b64,
                'ai_analysis': '{"analysis_sections": {"section_1_test": {"Analysis": "Test analysis", "Supporting_Phrases": ["foo"]}}}'
            }
        ]
        st.session_state.preprocessed_data = {filename: {"chunks": [{"text": "foo", "page_num": 1}], "valid_chunk_indices": []}}

        section_key = 'section_1_test'
        st.session_state.rag_retry_requests = {
            section_key: {
                'status': 'requested',
                'section_data': {
                    'Analysis': 'MT599 Swift inquiry',
                    'prompt': 'MT599 Swift inquiry',
                    'valid_chunk_indices': [],
                },
                'result': {'filename': filename},
            }
        }

        # Monkeypatch RAGRetryTool to avoid real LLM calls
        import src.keyword_code.agents.rag_agent as rag_agent_mod

        class DummyAnalysis(rag_agent_mod.RAGAnalysis):
            pass

        class DummyOptimizationAgent:
            def __init__(self, *args, **kwargs):
                pass

        class DummyRetryTool:
            def __init__(self, *args, **kwargs):
                pass

            async def retry_with_optimization(self, context):
                return [
                    {"text": "Optimized result A", "score": 0.91, "page_num": 3},
                    {"text": "Optimized result B", "score": 0.88, "page_num": 5},
                ], DummyAnalysis(
                    query_type='MT599_Swift',
                    current_quality_score=0.7,
                    issues_identified=[],
                    recommended_bm25_weight=0.8,
                    recommended_semantic_weight=0.2,
                    recommended_top_k=5,
                    reasoning='Heavily weight BM25 for MT599 Swift.'
                )

        original_retry_tool = rag_agent_mod.RAGRetryTool
        original_opt_agent = rag_agent_mod.RAGOptimizationAgent
        rag_agent_mod.RAGRetryTool = DummyRetryTool
        rag_agent_mod.RAGOptimizationAgent = DummyOptimizationAgent
        try:
            # Act
            page_mod = self._load_pages_module()
            updated_results, processed = page_mod.process_rag_requests(st.session_state.analysis_results)

            # Assert
            self.assertTrue(processed)
            self.assertIn('rag_retry_results', st.session_state)
            self.assertIn(section_key, st.session_state.rag_retry_results)
            rr = st.session_state.rag_retry_results[section_key]
            self.assertTrue(rr['new_results'])
            self.assertEqual(rr['new_results'][0]['text'], 'Optimized result A')
            self.assertEqual(rr.get('filename'), filename)
        finally:
            # Restore patch
            rag_agent_mod.RAGRetryTool = original_retry_tool


if __name__ == '__main__':
    unittest.main()

