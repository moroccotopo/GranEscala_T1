import json, os, logging
from datetime import datetime, timezone
import boto3, pandas as pd, streamlit as st
from pyathena import connect
try:
    from src.feedback.postgres_schema import add_feedback, init_db, list_feedback
except Exception:
    add_feedback=init_db=list_feedback=None
logging.basicConfig(level=os.getenv('LOG_LEVEL','INFO'), format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger=logging.getLogger('granescala_app')
AWS_REGION=os.getenv('AWS_REGION','us-east-1'); BUCKET=os.getenv('FORECAST_BUCKET', os.getenv('BUCKET_NAME','')); PREFIX=os.getenv('PROJECT_PREFIX','granescala'); DB=os.getenv('ATHENA_DATABASE','granescala_mvp'); OUT=os.getenv('ATHENA_OUTPUT', f's3://{BUCKET}/{PREFIX}/athena-results/'); FEEDBACK=os.getenv('FEEDBACK_BACKEND','s3'); ENDPOINT=os.getenv('ENDPOINT_NAME','')
st.set_page_config(page_title='GranEscala Forecast MVP', layout='wide'); st.title('GranEscala Forecast MVP'); st.caption('Docker + ECS Fargate + SageMaker Pipeline + Athena + RDS')
@st.cache_resource
def conn(): return connect(s3_staging_dir=OUT, region_name=AWS_REGION, schema_name=DB)
@st.cache_data(ttl=300)
def q(sql):
    logger.info('athena_query_start sql=%s', sql.replace('\n',' ')[:300])
    df = pd.read_sql(sql, conn())
    logger.info('athena_query_end rows=%s', len(df))
    return df
def save_json_s3(row):
    key=f"{PREFIX}/feedback/feedback_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}.json"; boto3.client('s3',region_name=AWS_REGION).put_object(Bucket=BUCKET,Key=key,Body=json.dumps(row,ensure_ascii=False).encode(),ContentType='application/json'); return f's3://{BUCKET}/{key}'
def save_export(df, filename):
    logger.info('export_start filename=%s rows=%s', filename, len(df))
    key=f'{PREFIX}/exports/{filename}'; boto3.client('s3',region_name=AWS_REGION).put_object(Bucket=BUCKET,Key=key,Body=df.to_csv(index=False).encode(),ContentType='text/csv'); return f's3://{BUCKET}/{key}'
def list_feedback_s3():
    s3=boto3.client('s3',region_name=AWS_REGION); resp=s3.list_objects_v2(Bucket=BUCKET,Prefix=f'{PREFIX}/feedback/',MaxKeys=100); rows=[]
    for obj in resp.get('Contents',[]):
        try: rows.append(json.loads(s3.get_object(Bucket=BUCKET,Key=obj['Key'])['Body'].read().decode()))
        except Exception: pass
    return pd.DataFrame(rows)
page=st.sidebar.radio('Sección',['Forecast Explorer','Evaluación','Exportación CFO','Feedback','Endpoint opcional','Estado'])
if page=='Forecast Explorer':
    try: runs=q('SELECT DISTINCT run_id FROM forecast_predictions ORDER BY run_id')['run_id'].tolist()
    except Exception as e: st.error(f'Error Athena: {e}'); st.stop()
    run=st.selectbox('Corrida',runs,index=len(runs)-1 if runs else 0); cats=q(f"SELECT DISTINCT item_category_id FROM forecast_predictions WHERE run_id='{run}' ORDER BY item_category_id")['item_category_id'].dropna().astype(int).tolist(); cat=st.selectbox('Categoría',['Todas']+cats)
    where=f"run_id='{run}'" + ('' if cat=='Todas' else f' AND item_category_id={cat}')
    s=q(f'SELECT SUM(prediction) total, COUNT(DISTINCT item_id) products, COUNT(*) rows FROM forecast_predictions WHERE {where}').iloc[0]
    c1,c2,c3=st.columns(3); c1.metric('Forecast total',f"{s.total:,.2f}"); c2.metric('Productos',f"{int(s.products):,}"); c3.metric('Registros',f"{int(s.rows):,}")
    st.dataframe(q(f'SELECT item_id,item_category_id,SUM(prediction) prediction FROM forecast_predictions WHERE {where} GROUP BY item_id,item_category_id ORDER BY prediction DESC LIMIT 50'), use_container_width=True)
elif page=='Evaluación':
    m=q('SELECT * FROM forecast_metrics'); st.subheader('Global'); st.dataframe(m[m.level=='global'].sort_values('run_id'), use_container_width=True); st.subheader('Errores por categoría'); st.dataframe(m[m.level=='category'].sort_values('wape',ascending=False).head(100), use_container_width=True); st.subheader('Ground truth vs predicción'); st.dataframe(q('SELECT * FROM forecast_evaluation LIMIT 1000'), use_container_width=True)
elif page=='Exportación CFO':
    runs=q('SELECT DISTINCT run_id FROM forecast_predictions ORDER BY run_id')['run_id'].tolist(); run=st.selectbox('Corrida',runs,index=len(runs)-1); 
    if st.button('Generar archivo CFO'):
        df=q(f"SELECT * FROM forecast_predictions WHERE run_id='{run}'"); fn=f"forecast_cfo_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"; path=save_export(df,fn); st.success(path); st.download_button('Descargar',df.to_csv(index=False),fn,'text/csv')
elif page=='Feedback':
    st.write('Backend:',FEEDBACK); item=st.number_input('item_id',0,value=1001); shop=st.number_input('shop_id',0,value=1); cat=st.number_input('item_category_id',-1,value=20); sev=st.selectbox('Severidad',['baja','media','alta','crítica']); user=st.text_input('Usuario','analista'); comment=st.text_area('Comentario')
    if st.button('Guardar') and comment.strip():
        row={'created_at':datetime.now(timezone.utc).isoformat(),'item_id':int(item),'shop_id':int(shop),'item_category_id':int(cat),'severity':sev,'created_by':user,'comment':comment,'status':'open'}
        if FEEDBACK=='postgres' and add_feedback: st.success(f"Postgres id={add_feedback(**row)}")
        else: st.success(save_json_s3(row))
    st.dataframe(pd.DataFrame(list_feedback(100) if FEEDBACK=='postgres' and list_feedback else list_feedback_s3()), use_container_width=True)
elif page=='Endpoint opcional':
    import json as _j
    payload={'month':st.number_input('month',0,11,5),'lag1_cnt':st.number_input('lag1_cnt',value=10.0),'lag12_cnt':st.number_input('lag12_cnt',value=8.0),'avg_price':st.number_input('avg_price',value=399.0)}; st.json(payload)
    if st.button('Invocar endpoint'):
        rt=boto3.client('sagemaker-runtime',region_name=AWS_REGION); resp=rt.invoke_endpoint(EndpointName=ENDPOINT,ContentType='application/json',Accept='application/json',Body=_j.dumps(payload)); st.json(_j.loads(resp['Body'].read()))
else: st.json({'AWS_REGION':AWS_REGION,'BUCKET':BUCKET,'PREFIX':PREFIX,'ATHENA_DATABASE':DB,'ATHENA_OUTPUT':OUT,'FEEDBACK_BACKEND':FEEDBACK,'ENDPOINT_NAME':ENDPOINT})
