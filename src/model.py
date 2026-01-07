import mlflow
import mlflow.pyfunc
from rag_chain import load_chain
import sys

class EpChatbot(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.chain = load_chain(context.artifacts["env"])

    def predict(self, context, model_input):
        return self.chain.run(model_input["question"])

if "--register" in sys.argv:
    env = sys.argv[sys.argv.index("--env") + 1]

    mlflow.set_experiment(f"/chatbot-task/pdf/{env}")

    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=EpChatbot(),
            artifacts={"env": env}
        )