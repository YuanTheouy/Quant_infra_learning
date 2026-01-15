# -*- coding: utf-8 -*-
"""
因子分析模块 - IC计算、分组分析、回测
支持使用DuckDB数据库而不是直接读取CSV文件
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import db_utils


def evaluate_factor(factor_table='week_factor', index_code='000852.SH', output_path='./Output', n_groups=10):
    """
    因子评估主流程（月度调仓）
    
    算法逻辑：
    1. 读取月度因子数据（从DuckDB表）
    2. 读取中证1000成分股（从metadata）
    3. 计算股票月度收益率
    4. 月度调仓：根据每月月底的因子值给股票分组，统计下一个月的多空收益
    5. 计算IC、IR、Sharpe等指标
    
    Args:
        factor_table: 因子表名（DuckDB中的表名，如'week_factor'）
        index_code: 指数代码（默认'000852.SH'，中证1000）
        output_path: 输出目录路径
        n_groups: 分组数（默认10）
        
    Returns:
        包含分组收益、多空收益、IC、摘要的字典
    """    
    # 1. 从DuckDB读取月度因子
    print('读取月度因子数据...')
    try:
        fac = db_utils.read_sql(f"SELECT * FROM {factor_table}")
        if 'raw_factor' in fac.columns:
            fac = fac.rename(columns={'raw_factor': 'factor'})
        # trade_date是整数，需要转为字符串后再转为日期
        fac["trade_date"] = fac["trade_date"].astype(str)
        # 从year_month字符串生成Period对象，或者从trade_date生成
        if 'year_month' in fac.columns:
            # year_month是字符串格式如'2024-01'，转换为Period
            fac["year_month"] = pd.PeriodIndex(fac["year_month"].astype(str), freq='M')
        else:
            # 如果没有year_month字段，从trade_date生成
            fac["date"] = pd.to_datetime(fac["trade_date"], format="%Y%m%d", errors="coerce")
            fac["year_month"] = fac["date"].dt.to_period("M")
        print(f'  因子数据: {len(fac)} 条记录，{fac["ts_code"].nunique()} 只股票')
    except Exception as e:
        print(f'错误：无法读取因子表 {factor_table}: {e}')
        return None
    
    # 2. 读取中证1000成分股
    print('读取指数成分股...')
    constituent_stocks = None
    try:
        constituent_path = f'./Data/Metadata/{index_code}_ins.csv'
        constituents = pd.read_csv(constituent_path)
        constituent_stocks = set(constituents['con_code'].unique())
        print(f'  成分股数量: {len(constituent_stocks)}')
        # 只保留成分股的因子
        fac = fac[fac['ts_code'].isin(constituent_stocks)].copy()
        print(f'  过滤后因子数据: {len(fac)} 条记录，{fac["ts_code"].nunique()} 只股票')
    except Exception as e:
        print(f'警告：无法读取成分股文件 {constituent_path}: {e}')
        print('  将使用全部股票')
    
    # 3. 计算股票月度收益率（只保留成分股）
    print('计算月度收益率...')
    stk = db_utils.query_stock_bar(columns=["ts_code", "trade_date", "pct_chg"])
    # 过滤只保留成分股
    if constituent_stocks is not None:
        stk = stk[stk['ts_code'].isin(constituent_stocks)].copy()
        print(f'  过滤后股票数据: {len(stk)} 条记录，{stk["ts_code"].nunique()} 只股票')
    stk["date"] = pd.to_datetime(stk["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    stk["year_month"] = stk["date"].dt.to_period("M")
    stk["ret"] = stk["pct_chg"] / 100.0
    
    # 按股票和月份聚合收益
    stk_m = (
        stk.dropna(subset=["year_month"])
           .groupby(["ts_code", "year_month"], as_index=False)
           .agg(ret_month=("ret", "sum"), trade_date=("trade_date", "last"))
    )
    stk_m["trade_date"] = stk_m["trade_date"].astype(str)
    stk_m = stk_m.sort_values(["ts_code", "year_month"])
    # 下一个月的收益（用于回测）
    stk_m["ret_next"] = stk_m.groupby("ts_code")["ret_month"].shift(-1)
    print(f'  月度收益数据: {len(stk_m)} 条记录')
    
    # 4. 读取指数月度收益（中证1000）
    print('读取指数收益数据...')
    idx = db_utils.read_sql(f"SELECT trade_date, pct_chg FROM index_data WHERE ts_code = '{index_code}'")
    idx["date"] = pd.to_datetime(idx["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    idx["year_month"] = idx["date"].dt.to_period("M")
    idx["ret"] = idx["pct_chg"] / 100.0
    idx_m = (
        idx.dropna(subset=["year_month"])
           .groupby(["year_month"], as_index=False)
           .agg(ret_month=("ret", "sum"), trade_date=("trade_date", "last"))
    )
    idx_m["trade_date"] = idx_m["trade_date"].astype(str)
    idx_m = idx_m.sort_values("year_month")
    idx_m["ret_next"] = idx_m["ret_month"].shift(-1)
    print(f'  指数月度收益: {len(idx_m)} 条记录')
    
    # 5. 合并因子和收益数据（月度调仓逻辑）
    print('合并因子和收益数据...')
    # 下月收益（使用year_month合并）
    stk_m_next = stk_m[["ts_code", "year_month", "ret_next"]].rename(
        columns={"ret_next": "ret_stock_next"}
    )
    # 指数下月收益（使用year_month合并）
    idx_m_next = idx_m[["year_month", "ret_next"]].rename(
        columns={"ret_next": "ret_index_next"}
    )
    
    # 合并：因子 + 下月收益 + 指数下月收益（使用year_month作为键）
    df = (
        fac[["ts_code", "year_month", "trade_date", "factor"]]
           .merge(stk_m_next, on=["ts_code", "year_month"], how="inner")
           .merge(idx_m_next, on="year_month", how="inner")
           .dropna(subset=["factor", "ret_stock_next", "ret_index_next"])
           .reset_index(drop=True)
    )
    
    print(f'  合并后数据: {len(df)} 条记录')
    if len(df) == 0:
        print("警告：没有有效的因子-收益数据可以评估")
        return None

    def _cut(x, q=n_groups):
        if x.notna().sum() < 2 or x.nunique() < 2:
            return pd.Series([np.nan] * len(x), index=x.index)
        return pd.qcut(x, q, labels=range(1, q + 1), duplicates="drop")

    df = df.sort_values(["trade_date", "ts_code"])
    df["group"] = df.groupby("trade_date")["factor"].apply(_cut, q=n_groups).reset_index(level=0, drop=True)

    grp_ret = (
        df.dropna(subset=["group"])
          .groupby(["trade_date", "group"], as_index=False)
          .agg(group_ret=("ret_stock_next", "mean"),
               benchmark=("ret_index_next", "first"))
    )

    long_series = grp_ret[grp_ret["group"] == n_groups][["trade_date", "group_ret"]].rename(columns={"group_ret": "long"})
    short_series = grp_ret[grp_ret["group"] == 1][["trade_date", "group_ret"]].rename(columns={"group_ret": "short"})
    ls = (
        long_series.merge(short_series, on="trade_date", how="inner")
                   .merge(grp_ret[["trade_date", "benchmark"]].drop_duplicates(), on="trade_date", how="left")
    )
    ls["ls"] = ls["long"] - ls["short"]
    ls = ls.sort_values("trade_date").reset_index(drop=True)

    # IC & IR
    print('计算IC序列...')
    # 计算每个交易日的Rank IC（Spearman相关系数）
    ic_series = df.groupby('trade_date').apply(lambda x: x['factor'].corr(x['ret_stock_next'], method='spearman'))
    ic_df = pd.DataFrame({'trade_date': ic_series.index, 'ic': ic_series.values})
    if len(ic_df) != 0:
        ic_valid = ic_df["ic"].dropna()
        IR = float(ic_valid.mean() / ic_valid.std(ddof=1)) if len(ic_valid) > 0 and ic_valid.std(ddof=1) > 0 else np.nan

    # Sharpe Ratio (月度调整)
    ls_valid = ls["ls"].dropna()
    mean_ls = float(ls_valid.mean()) if len(ls_valid) > 0 else np.nan
    std_ls = float(ls_valid.std(ddof=1)) if len(ls_valid) > 0 else np.nan
    SR_monthly = float(mean_ls / std_ls) if std_ls > 0 else np.nan
    SR_annual = float(SR_monthly * np.sqrt(12)) if pd.notna(SR_monthly) else np.nan  # 月度换算为年度

    summary = pd.DataFrame(
        {
            "metric": ["IC_mean", "IC_std", "IR", "LS_mean", "LS_std", "SR_monthly", "SR_annual"],
            "value": [
                float(ic_valid.mean()) if len(ic_valid) else np.nan,
                float(ic_valid.std(ddof=1)) if len(ic_valid) else np.nan,
                IR,
                mean_ls if len(ls_valid) else np.nan,
                std_ls if len(ls_valid) else np.nan,
                SR_monthly,
                SR_annual,
            ],
        }
    )
    
    # 计算各分组的平均收益
    group_stats = grp_ret.groupby('group')['group_ret'].agg(['mean', 'std', 'count']).reset_index()
    group_stats.columns = ['group', 'avg_return', 'std_return', 'n_periods']
    print('\n各分组平均月度收益:')
    print(group_stats)

    # 输出到指定目录
    grp_ret.to_csv(f"{output_path}/group_ret.csv", index=False)
    ls.to_csv(f"{output_path}/ls_ret.csv", index=False)
    ic_df.to_csv(f"{output_path}/ic_series.csv", index=False)
    summary.to_csv(f"{output_path}/eval_summary.csv", index=False)
    group_stats.to_csv(f"{output_path}/group_stats.csv", index=False)

    # 图形输出
    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    axes[0].axis("tight")
    axes[0].axis("off")
    table_data = summary.values.tolist()
    axes[0].table(cellText=table_data, colLabels=summary.columns.tolist(), loc="center")
    axes[0].set_title("Factor Evaluation Summary", fontsize=14, pad=12)

    # 累计收益
    ls["cum_ls"] = ls["ls"].cumsum()
    bench_cum = ls["benchmark"].cumsum()

    # 绘图
    x = np.arange(len(ls))
    axes[1].plot(x, ls["cum_ls"], label="L-S cumulative return")
    axes[1].plot(x, bench_cum, label="Benchmark cumulative return")

    if len(ls) >= 2:
        axes[1].set_xticks([0, len(ls) - 1])
        axes[1].set_xticklabels([ls["trade_date"].iloc[0], ls["trade_date"].iloc[-1]])
    else:
        axes[1].set_xticks([0])
        axes[1].set_xticklabels([ls["trade_date"].iloc[0]])

    axes[1].set_title("Cumulative Returns (Simple Sum)", fontsize=12)
    axes[1].set_xlabel("Trade Date")
    axes[1].set_ylabel("Cumulative Return (Sum)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"{output_path}/factor_eval_result.png", dpi=200)
    plt.show()

    print(f'因子评估完成，结果已保存到 {output_path}/')
    
    return {"group_ret": grp_ret, "ls": ls, "ic": ic_df, "summary": summary}
