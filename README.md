# MMIR
An Evaluation of Cross-Modal Text-to-Image Retrieval Pipelines

## Overall Aim
The goal of this project is to benchmark and evaluate various cross-modal retrieval pipelines, specifically focusing on text-to-image retrieval. It involves generating embeddings using state-of-the-art models (CLIP, FLAVA, UniIR, PreFLMR), building vector stores (FAISS, ColBERT), and serving retrieval requests through dedicated services.

## Documentation
- [Embedding Generation](Benchmark/code/embeddingGeneration/README.md)
- [FAISS Retrieval Service](Benchmark/code/retrievalService/faiss_base_service/README.md)
- [FAISS Two-Stage Testing](Benchmark/code/retrievalService/faiss_base_service/2_stage_testing_faiss/README.md)
- [FLMR Retrieval Service](Benchmark/code/retrievalService/flmr/README.md)
- [FAISS Index Building](Benchmark/code/vectorStore/faiss/ingestDataIntoFAISS/README.md)
- [FLMR Index Building](Benchmark/code/vectorStore/flmr/README.md)
- [Configuration](Benchmark/config/README.md)
- [Evaluation Scripts](Benchmark/code/evaluation/README.md)
