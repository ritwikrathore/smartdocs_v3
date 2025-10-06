# CNT SmartDocs 
## Project Diff Since 5/28

```
.
├── src/
│   └── keyword_code/               
│       ├── ai/                     
│       │   ├── analyzer.py         
│       │   ├── chat.py             
│       │   ├── databricks_llm.py   
│       │   ├── decomposition.py    
│       ├── assets/                 
│       ├── models/                 
│       │   ├── databricks_embedding.py  # Updated to remove sentence-transformers
│       │   ├── databricks_reranker.py  # Updated to remove torch
│       │   └── embedding.py        # # Updated to remove torch
│       ├── processors/             
│       │   ├── pdf_processor.py    
│       │   └── word_processor.py   
│       ├── rag/                    
│       │   ├── retrieval.py        # Updated to remove torch
│       │   └── chunking.py         
│       ├── utils/                  
│       │   ├── async_utils.py      
│       │   ├── display.py          # Minor update
│       │   ├── file_manager.py    
│       │   ├── helpers.py          
│       │   ├── interaction_logger.py 
│       │   ├── memory_monitor.py   
│       │   ├── spacy_utils.py      
│       │   └── ui_helpers.py       # Minor update
│       ├── app.py                  # Updated to remove torch
│       └── config.py               # Updated to remove sentence-transformers
├── models/                         
│   └── spacy/                      
├── tmp/                            
├── requirements.txt                # Updated to remove torch and sentnece-transformers
└── README.md                       
```
