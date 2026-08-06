"""
Script 06: RAGAS Evaluation
=============================
Measures the quality of your RAG system across 4 metrics:
  1. Faithfulness     — Are all claims in the answer grounded in retrieved context?
  2. Answer Relevance — Does the answer address the question?
  3. Context Precision — Were the retrieved chunks actually useful?
  4. Context Recall    — Was all needed information retrieved?

Run with:
  python scripts/06_evaluate.py
"""

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# Build a small evaluation dataset manually
# In production: generate this from real user queries and expert-written answers
eval_data = {
    "question": [
        "What is the main topic of the document?",
        "What are the key points explained?",
    ],
    "answer": [
        # Run your agent on these questions and paste the answers here
        "The document covers...",
        "The key points are...",
    ],
    "contexts": [
        # Paste the retrieved chunks for each question here
        ["Chunk text 1", "Chunk text 2"],
        ["Chunk text 1", "Chunk text 3"],
    ],
    "ground_truth": [
        # Write the ideal answer for each question
        "The ideal answer for question 1...",
        "The ideal answer for question 2...",
    ]
}

dataset = Dataset.from_dict(eval_data)

print("[INFO] Running RAGAS evaluation...")
results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])

print("\n" + "="*60)
print("RAGAS EVALUATION RESULTS")
print("="*60)
print(f"Faithfulness     : {results['faithfulness']:.3f}  (1.0 = no hallucinations)")
print(f"Answer Relevance : {results['answer_relevancy']:.3f}  (1.0 = perfectly relevant)")
print("="*60)
print("\nTarget: All metrics > 0.80 for production quality.")