from __future__ import annotations

import argparse
import time

import sagemaker
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--raw-s3", required=True)
    parser.add_argument("--output-s3", required=True)
    parser.add_argument("--pipeline-name", default="GranEscalaDockerPipeline")
    parser.add_argument("--max-wape", type=float, default=0.75)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()

    pipeline_session = PipelineSession()

    raw_input_s3 = ParameterString("RawInputS3", default_value=args.raw_s3.rstrip("/") + "/")
    output_s3 = ParameterString("OutputS3", default_value=args.output_s3.rstrip("/"))
    max_wape = ParameterFloat("MaxWape", default_value=args.max_wape)
    run_id = ParameterString("RunId", default_value=f"run_{int(time.time())}")

    processor = Processor(
        image_uri=args.image_uri,
        role=args.role_arn,
        instance_count=1,
        instance_type="ml.m5.large",
        command=["python", "-m", "src.cloud_pipeline.run_step"],
        sagemaker_session=pipeline_session,
    )

    preprocess_args = processor.run(
        inputs=[
            ProcessingInput(
                source=raw_input_s3,
                destination="/opt/ml/processing/input/raw",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="prep",
                source="/opt/ml/processing/output/prep",
                destination=Join(on="/", values=[output_s3, "processed"]),
            )
        ],
        arguments=["--step", "preprocess"],
    )
    step_preprocess = ProcessingStep(name="Preprocess", step_args=preprocess_args)

    train_args = processor.run(
        inputs=[
            ProcessingInput(
                source=step_preprocess.properties.ProcessingOutputConfig.Outputs["prep"].S3Output.S3Uri,
                destination="/opt/ml/processing/input/prep",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="model",
                source="/opt/ml/processing/output/model",
                destination=Join(on="/", values=[output_s3, "model"]),
            )
        ],
        arguments=["--step", "train"],
    )
    step_train = ProcessingStep(name="Train", step_args=train_args)

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation_report.json",
    )

    evaluate_args = processor.run(
        inputs=[
            ProcessingInput(
                source=step_preprocess.properties.ProcessingOutputConfig.Outputs["prep"].S3Output.S3Uri,
                destination="/opt/ml/processing/input/prep",
            ),
            ProcessingInput(
                source=step_train.properties.ProcessingOutputConfig.Outputs["model"].S3Output.S3Uri,
                destination="/opt/ml/processing/input/model",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/output/evaluation",
                destination=Join(on="/", values=[output_s3, "evaluation"]),
            )
        ],
        arguments=["--step", "evaluate", "--run-id", run_id],
    )
    step_evaluate = ProcessingStep(
        name="Evaluate",
        step_args=evaluate_args,
        property_files=[evaluation_report],
    )

    batch_predict_args = processor.run(
        inputs=[
            ProcessingInput(
                source=step_preprocess.properties.ProcessingOutputConfig.Outputs["prep"].S3Output.S3Uri,
                destination="/opt/ml/processing/input/prep",
            ),
            ProcessingInput(
                source=step_train.properties.ProcessingOutputConfig.Outputs["model"].S3Output.S3Uri,
                destination="/opt/ml/processing/input/model",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="curated",
                source="/opt/ml/processing/output/curated",
                destination=Join(on="/", values=[output_s3, "curated"]),
            )
        ],
        arguments=["--step", "batch_predict", "--run-id", run_id],
    )
    step_batch_predict = ProcessingStep(name="BatchPredict", step_args=batch_predict_args)

    quality_gate = ConditionStep(
        name="CheckModelQuality",
        conditions=[
            ConditionLessThanOrEqualTo(
                left=JsonGet(
                    step_name=step_evaluate.name,
                    property_file=evaluation_report,
                    json_path="metrics.wape",
                ),
                right=max_wape,
            )
        ],
        if_steps=[step_batch_predict],
        else_steps=[],
    )

    pipeline = Pipeline(
        name=args.pipeline_name,
        parameters=[raw_input_s3, output_s3, max_wape, run_id],
        steps=[step_preprocess, step_train, step_evaluate, quality_gate],
        sagemaker_session=pipeline_session,
    )

    pipeline.upsert(role_arn=args.role_arn)
    print(f"Pipeline creado/actualizado: {args.pipeline_name}")

    if args.start:
        execution = pipeline.start()
        print(f"Ejecución iniciada: {execution.arn}")


if __name__ == "__main__":
    main()
