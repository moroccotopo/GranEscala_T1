CREATE EXTERNAL TABLE IF NOT EXISTS {{GLUE_DB}}.forecast_predictions (
  run_id string,
  forecast_month string,
  shop_id bigint,
  item_id bigint,
  item_category_id bigint,
  prediction double
)
STORED AS PARQUET
LOCATION 's3://{{BUCKET_NAME}}/{{PROJECT_PREFIX}}/pipeline-runs/curated/forecast_predictions/';

CREATE EXTERNAL TABLE IF NOT EXISTS {{GLUE_DB}}.forecast_evaluation (
  run_id string,
  date_block_num bigint,
  shop_id bigint,
  item_id bigint,
  item_category_id bigint,
  y_true double,
  y_pred double,
  error double,
  abs_error double
)
STORED AS PARQUET
LOCATION 's3://{{BUCKET_NAME}}/{{PROJECT_PREFIX}}/pipeline-runs/evaluation/forecast_evaluation/';

CREATE EXTERNAL TABLE IF NOT EXISTS {{GLUE_DB}}.forecast_metrics (
  run_id string,
  level string,
  group_id string,
  mae double,
  rmse double,
  wape double,
  bias double,
  n_obs bigint,
  item_category_id bigint,
  item_id bigint
)
STORED AS PARQUET
LOCATION 's3://{{BUCKET_NAME}}/{{PROJECT_PREFIX}}/pipeline-runs/evaluation/forecast_metrics/';
