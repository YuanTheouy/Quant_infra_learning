import tushare as ts
import pandas as pd
import numpy as np
import time
import math
from datetime import datetime, timedelta
from tqdm import tqdm

from joblib import Parallel, delayed
import os
from . import db_utils

# Tushare配置，请确保在环境变量中设置了TB_TOKEN（相关流程请自行ai，或在此处直接输入token）。
token = os.getenv('TB_TOKEN')
BASIC_INFO_PATH = 'Data/Metadata'
def _get_pro_client():
    if not token:
        raise RuntimeError('未找到TB_TOKEN环境变量，请先配置后再运行。')
    # 显式传入 token，避免 tushare 在导入阶段进行 token 连接
    return ts.pro_api(token)

# ====================  数据获取函数  ====================
def get_ins(index_code):
    """获取指数成分股并保存，tushare接口只能获取最近一个月的成分股数据，所以这里自动计算上个月的日期范围"""
    today = datetime.now()
    if today.month == 1:
        first_day_of_last_month = today.replace(year=today.year - 1, month=12, day=1)
    else:
        first_day_of_last_month = today.replace(month=today.month - 1, day=1)
    last_day_of_last_month = today.replace(day=1) - timedelta(days=1)
    
    start = first_day_of_last_month.strftime('%Y%m%d')
    end = last_day_of_last_month.strftime('%Y%m%d')
    
    try:
        pro = _get_pro_client()
        ins = pro.index_weight(index_code=index_code, start_date=start, end_date=end)
        if ins is None or ins.empty:
            print(f"警告: 未获取到指数 {index_code} 的成分股数据")
            return
        ins_codes = ins['con_code'].unique().tolist()

        pd.DataFrame({'con_code': ins_codes}).to_csv(f'{BASIC_INFO_PATH}/{index_code}_ins.csv', index=False)

        return set(ins_codes)
    except Exception as e:
        print(f"获取指数成分股失败: {e}")

def get_trade(start_date, end_date):
    """获取交易日历"""
    pro = _get_pro_client()
    df = pro.trade_cal(exchange='SSE', is_open='1', start_date=start_date, end_date=end_date, fields='cal_date')
    df.to_csv(f'{BASIC_INFO_PATH}/trade_day.csv', index=False)
    return df

def fetch_bar_by_single_date(date):  ## 增加重试机制，避免偶尔的网络问题导致数据缺失
    for i in range(3):  # 最多重试3次
        try:
            pro = _get_pro_client()
            df = pro.daily(trade_date=date)
            df = df[df['pct_chg'].abs() < 35]  # 过滤掉涨跌幅超过35%的异常数据
            if df is not None:
                time.sleep(0.8) # 基础积分建议增加睡眠时间
                return (date, df)
        except Exception as e:
            if "最多访问" in str(e): # 如果是频率限制，多睡一会儿
                time.sleep(10)
            else:
                print(f"日期 {date} 第 {i+1} 次尝试失败: {e}")
                time.sleep(2)
    return (date, None)
def fetch_basic_by_single_date(date):  ## 增加重试机制，避免偶尔的网络问题导致数据缺失
    for i in range(3):  # 最多重试3次
        try:
            pro = _get_pro_client()
            df = pro.daily_basic(trade_date=date)
            if df is not None:
                time.sleep(1.5) # 基础积分建议增加睡眠时间
                return (date, df)
        except Exception as e:
            if "最多访问" in str(e): # 如果是频率限制，多睡一会儿
                time.sleep(10)
            else:
                print(f"日期 {date} 第 {i+1} 次尝试失败: {e}")
                time.sleep(2)
    return (date, None)
def get_data_by_date(single_function, table_name):
    """
    按交易日多线程循环获取股票日频数据，自动保存数据，返回为空
    
    Args:
        single_function: 下载单日数据的函数
        table_name: DuckDB表名
    """    
    # 过滤出需要下载的日期
    dates_to_download = get_dates_todo(table_name)
    
    if not dates_to_download:
        print("数据已是最新")
        return 

    print(f"正在下载从{(dates_to_download)[0]}到{(dates_to_download)[-1]}的数据")
    
    results = Parallel(n_jobs=-1)(
        delayed(single_function)(date) for date in tqdm(dates_to_download, desc="下载进度"))
    
    # 过滤掉None结果
    all_df_list = [df for date, df in results if df is not None and not df.empty]
    
    if all_df_list:
        new_data = pd.concat(all_df_list, ignore_index=True)
        new_data.sort_values(by=['trade_date', 'ts_code'], ascending=False, inplace=True)
        db_utils.write_to_db(new_data, table_name, save_mode='append')
def get_stock_data_by_date():
    """获取股票日频数据"""
    get_data_by_date(fetch_bar_by_single_date, table_name='stock_bar')

def get_daily_basic():
    """获取每日指标，如换手率、量比、PB、PE、PS、股息率、总股本、总市值"""
    get_data_by_date(fetch_basic_by_single_date, table_name='daily_basic')

def get_index_data(index_code):
    """获取指数日频数据"""
    # 首先尝试从本地 DuckDB 中读取，避免每次运行都向 Tushare 发起网络请求并重复写入
    #获取下载日期范围
    dates_to_download = get_dates_todo('index_data', ts_code=index_code)
    if not dates_to_download:
        return db_utils.read_sql(f"SELECT * FROM index_data WHERE ts_code='{index_code}'")
    
    else:
        pro = _get_pro_client()
        df = pro.index_daily(ts_code=index_code, start_date=dates_to_download[0], end_date=dates_to_download[-1])
        if df is not None and not df.empty:
            db_utils.write_to_db(df, 'index_data', save_mode='append')
        index_df = db_utils.read_sql(f"SELECT * FROM index_data WHERE ts_code='{index_code}'")
        return index_df

def get_dates_todo(table_name,ts_code=None):
    """
    获取需要下载更新数据的时间范围
    max_trade_day 为数据库中已有数据的最大交易日
    latest_date_str 为当前最新的交易日（根据当前时间自动计算）
    table_max_date 为数据库表中已有数据的最大日期记录
    trade_dates 为交易日历
    Args:
        table_name: 数据库表名
        ts_code: 股票代码（可选）

    Returns:
        dates_to_download: 需要更新的日期列表，若无须更新则返回None
    """
    now = datetime.now()
    
    # 匹配tushare数据入库时间：大于等于18点算当日，否则停在昨天
    if now.hour >= 18:
        latest_date = now.date()
    else:
        latest_date = now.date() - timedelta(days=1)
        
    # 若为周末，调整至周五
    if latest_date.weekday() >= 5:
        latest_date -= timedelta(days=latest_date.weekday() - 4)
        
    latest_date_str = latest_date.strftime('%Y%m%d')
    trade_day_path = f'{BASIC_INFO_PATH}/trade_day.csv'
    # 初始化读取交易日历并比较
    if os.path.exists(trade_day_path):
        trade_dates = pd.read_csv(trade_day_path)
        max_trade_day = trade_dates['cal_date'].astype(str).max()
    else:
        max_trade_day = '0'
        
    # 当前最新日期超出日历已知最大日期，更新max_trade_day
    if latest_date_str > max_trade_day:
        trade_dates = get_trade('20160101', latest_date_str)
        max_trade_day = trade_dates['cal_date'].astype(str).max()
        
    # 获取表中最新的日期记录
    try:
        if ts_code is not None:
            query = f"SELECT MAX(trade_date) as max_date FROM {table_name} WHERE ts_code='{ts_code}'"
        else:
            query = f"SELECT MAX(trade_date) as max_date FROM {table_name}"
        result = db_utils.read_sql(query)
        table_max_date = str(result.iloc[0, 0])
    except Exception:
        table_max_date = '0'
    
    # 若表数据落后于最新交易日，则需要下载最新交易数据
    if table_max_date < max_trade_day:
        dates_to_download_df =trade_dates[trade_dates['cal_date'].astype(str) > table_max_date]
        dates_to_download = dates_to_download_df['cal_date'].astype(str).tolist()
        return sorted(dates_to_download)
             
