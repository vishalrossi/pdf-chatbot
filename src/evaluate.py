from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

def run_eval(dataset, chain):
    return evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=chain.llm
    )