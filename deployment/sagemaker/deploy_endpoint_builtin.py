import os, sagemaker
from sagemaker.sklearn.model import SKLearnModel
role=os.environ['SAGEMAKER_ROLE_ARN']; model_s3=os.environ['MODEL_S3']; endpoint=os.environ.get('ENDPOINT_NAME','granescala-ridge-endpoint')
model=SKLearnModel(model_data=model_s3, role=role, entry_point='inference.py', framework_version='1.2-1', py_version='py3', sagemaker_session=sagemaker.Session())
predictor=model.deploy(initial_instance_count=1, instance_type='ml.m5.large', endpoint_name=endpoint)
print(predictor.endpoint_name)
