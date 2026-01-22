export OPENAI_API_KEY="your_openai_api_key_here"
epxort OPENAI_BASE_URL="your_openai_base_url_here"

python build_graph.py --domains Long \
                      --source_texts narrative_qa \
                      --llm_model_name gpt-4o


python graph_run.py --domains Long \
                     --dataset narrative_qa \
                     --graph_mode single \
                     --rag_mode HGMem \
                     --source_type chunks100 \
                     --question_type traditional \
                     --llm_model_name gpt-4o
