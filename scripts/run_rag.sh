python graph_run.py --domains Financial,Governmental \
                     --dataset longbench_v2_qa \
                     --graph_mode single \
                     --rag_mode debug \
                     --source_type chunks100 \
                     --question_type sampled_difficult \
                     --llm_model_name gpt-4o-mini

python build_graph.py --domains Financial \
                      --source_texts longbench_v2_qa
