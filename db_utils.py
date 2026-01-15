# -*- coding: utf-8 -*-
"""
DuckDB数据库工具模块 - 替代大CSV文件的读写操作
"""
import duckdb
import pandas as pd
import os
from pathlib import Path

# 数据库路径常量
DB_PATH = './Data/data.db'

def init_db():
    """初始化数据库连接"""
    os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else '.', exist_ok=True)
    return duckdb.connect(DB_PATH)
def csv_to_db(csv_path, table_name, if_exists='replace'):
    """
    将CSV文件导入DuckDB表
    
    Args:
        csv_path: CSV文件路径
        table_name: 目标表名
        if_exists: 'replace'|'append'
    """
    if not os.path.exists(csv_path):
        print(f"⚠ 文件不存在: {csv_path}")
        return False
    
    conn = init_db()
    try:
        print(f"正在导入 {csv_path} 到表 {table_name}...")
        
        # 读取CSV并写入数据库
        df = pd.read_csv(csv_path, dtype_backend='numpy_nullable')
        
        if if_exists == 'replace':
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        
        # 使用DuckDB的insert语句而不是to_sql以获得更好的性能
        conn.register('temp_df', df)
        if if_exists == 'replace':
            conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM temp_df")
        else:  # append
            conn.execute(f"INSERT INTO {table_name} SELECT * FROM temp_df")
        
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchall()[0][0]
        print(f"✓ 成功导入 {row_count:,} 行数据到表 {table_name}")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False
    finally:
        conn.close()

def read_sql(query):
    """
    执行SQL查询并返回DataFrame
    
    Args:
        query: SQL查询语句
        
    Returns:
        DataFrame
    """
    conn = init_db()
    try:
        result = conn.execute(query).fetch_df()
        return result
    finally:
        conn.close()

def write_to_db(df, table_name, if_exists='replace'):
    """
    将DataFrame写入数据库
    
    Args:
        df: DataFrame
        table_name: 目标表名
        if_exists: 'replace'|'append'
    """
    conn = init_db()
    try:
        conn.register('temp_df', df)
        if if_exists == 'replace':
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM temp_df")
        else:  # append
            conn.execute(f"INSERT INTO {table_name} SELECT * FROM temp_df")
        
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchall()[0][0]
        print(f"✓ 成功写入 {row_count:,} 行数据到表 {table_name}")
    finally:
        conn.close()

def query_stock_bar(start_date=None, end_date=None, ts_codes=None, columns=None):
    """
    查询股票日线数据
    
    Args:
        start_date: 开始日期 (YYYYMMDD格式)
        end_date: 结束日期 (YYYYMMDD格式)
        ts_codes: 股票代码列表
        columns: 要查询的列名列表，默认为None（查询所有列）
        
    Returns:
        DataFrame
    """
    conn = init_db()
    try:
        # 构建SELECT子句
        if columns:
            select_clause = ", ".join(columns)
        else:
            select_clause = "*"
        
        query = f"SELECT {select_clause} FROM stock_bar WHERE 1=1"
        
        if start_date:
            query += f" AND trade_date >= '{start_date}'"
        if end_date:
            query += f" AND trade_date <= '{end_date}'"
        if ts_codes:
            codes_str = "','".join(ts_codes)
            query += f" AND ts_code IN ('{codes_str}')"
        
        query += " ORDER BY ts_code, trade_date"
        result = conn.execute(query).fetch_df()
        return result
    finally:
        conn.close()

def query_daily_basic(start_date=None, end_date=None, ts_codes=None, columns=None):
    """
    查询每日基本信息
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        ts_codes: 股票代码列表
        columns: 要查询的列名列表，默认为None（查询所有列）
        
    Returns:
        DataFrame
    """
    conn = init_db()
    try:
        # 构建SELECT子句
        if columns:
            select_clause = ", ".join(columns)
        else:
            select_clause = "*"
        
        query = f"SELECT {select_clause} FROM daily_basic WHERE 1=1"
        
        if start_date:
            query += f" AND trade_date >= '{start_date}'"
        if end_date:
            query += f" AND trade_date <= '{end_date}'"
        if ts_codes:
            codes_str = "','".join(ts_codes)
            query += f" AND ts_code IN ('{codes_str}')"
        
        query += " ORDER BY ts_code, trade_date"
        result = conn.execute(query).fetch_df()
        return result
    finally:
        conn.close()

def query_index_data(ts_code=None, start_date=None, end_date=None, columns=None):
    """
    查询指数数据
    
    Args:
        ts_code: 指数代码
        start_date: 开始日期
        end_date: 结束日期
        columns: 要查询的列名列表，默认为None（查询所有列）
        
    Returns:
        DataFrame
    """
    conn = init_db()
    try:
        # 构建SELECT子句
        if columns:
            select_clause = ", ".join(columns)
        else:
            select_clause = "*"
        
        query = f"SELECT {select_clause} FROM index_data WHERE 1=1"
        
        if ts_code:
            query += f" AND ts_code = '{ts_code}'"
        if start_date:
            query += f" AND trade_date >= '{start_date}'"
        if end_date:
            query += f" AND trade_date <= '{end_date}'"
        
        query += " ORDER BY trade_date"
        result = conn.execute(query).fetch_df()
        return result
    finally:
        conn.close()

def append_to_db(csv_path, table_name):
    """
    追加CSV数据到现有表
    
    Args:
        csv_path: CSV文件路径
        table_name: 目标表名
    """
    return csv_to_db(csv_path, table_name, if_exists='append')

def get_table_stats(table_name):
    """获取表的统计信息"""
    conn = init_db()
    try:
        result = conn.execute(f"""
            SELECT 
                '{table_name}' as table_name,
                COUNT(*) as row_count,
                COUNT(DISTINCT ts_code) as unique_codes,
                MIN(trade_date) as min_date,
                MAX(trade_date) as max_date
            FROM {table_name}
        """).fetch_df()
        return result
    finally:
        conn.close()

def export_db_to_csv(table_name, output_path):
    """
    将数据库表导出为CSV
    
    Args:
        table_name: 表名
        output_path: 输出CSV路径
    """
    conn = init_db()
    try:
        result = conn.execute(f"SELECT * FROM {table_name}").fetch_df()
        result.to_csv(output_path, index=False)
        print(f"✓ 已导出 {len(result):,} 行到 {output_path}")
    finally:
        conn.close()

if __name__ == '__main__':
    conn = init_db()
    conn.sql("SHOW TABLES").show()
    conn.sql("DESCRIBE stock_bar").show()
    # 方式 A：直接打印到控制台（非常漂亮的可视化表格）
    # conn.table('pricing_factors').show()

    # 方式 B：获取列名列表
    columns = conn.execute("PRAGMA table_info('week_factor')").fetchall()
    print([col[1] for col in columns]) # 只打印列名
