# -*- coding: utf-8 -*-
"""
因子分析模块 - IC计算、分组分析、回测
支持使用DuckDB数据库而不是直接读取CSV文件
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from quant_infra import db_utils
from quant_infra.const import *
from quant_infra.get_data import get_ins, get_index_data
from quant_infra.factor_calc import winsorize
import itertools
from pathlib import Path

def specific_group(fac, stk, group_set, bench_df, n_groups=10):
    """
    对特定分组进行评估，后续使用Joblib并行回测
    Args:
        fac: 已按样本过滤并排序的因子数据 ['date','factor','ts_code','trade_date']
        stk: 已按样本过滤并排序的股票数据 ['date','ret','ts_code','trade_date']
        group_set: 分组集合设定，形如('全市场','日度')
        bench_df: 基准指数日收益 DataFrame，列为 ['trade_date', 'bench_ret']
    Returns:
        分组收益、多空收益、IC、IR、Sharpe
    """
    # fac/stk 已在 evaluate_factor 中按样本过滤并预排序，此处无需重复操作

    # 非日频的收益率要提前计算。这是为了计算后面的IC，必须对其因子和股票的频率。如果是为了画收益率图，不用这部分
    if group_set[1] != '日度':
        # 1. 获取对应的 Pandas 频率缩写（如 'W', 'M', 'Q', 'A'）
        p_freq = FREQ_MAP.get(group_set[1])
        if not p_freq:
            raise ValueError(f"不支持的频率: {group_set[1]}")

        # 2. 生成周期标签列
        period_col = f"date_{p_freq}"
        ret_col = f"ret_{p_freq}"
        fac_col = f"factor_{p_freq}"
        stk[period_col] = stk["date"].dt.to_period(p_freq)
        fac[period_col] = fac["date"].dt.to_period(p_freq)

        # 3. 按股票和周期聚合收益、因子
        # 使用 **{ret_col: ...} 来动态传入列名
        stk_p = (stk.dropna(subset=[period_col])
                .groupby(["ts_code", period_col], as_index=False)
                .agg(**{
                    ret_col: ("ret", lambda x: (1 + x).prod() - 1), # 复利累计收益率
                    "trade_date": ("trade_date", "max")
                }))

        fac_p = (fac.dropna(subset=[period_col])
                .groupby(["ts_code", period_col], as_index=False)
                .agg(**{
                    fac_col: ("factor", "last"), # 取最后一个值作为该周期的因子值
                    "trade_date": ("trade_date", "max")
                }))

        stk_p.sort_values(["ts_code", "trade_date"], inplace=True)
        fac_p.sort_values(["ts_code", "trade_date"], inplace=True)

        # 4. 计算下一周期的收益（Look-ahead return）
        next_ret_col = f"next_ret_{p_freq}"
        stk_p[next_ret_col] = stk_p.groupby("ts_code")[ret_col].shift(-1)

    else:
        # 日度：数据本身已是日频，无需聚合
        period_col = 'date_D'
        ret_col = 'ret'
        fac_col = 'factor'
        next_ret_col = 'next_ret_D'

        stk[period_col] = stk['date'].dt.to_period('D')
        fac[period_col] = fac['date'].dt.to_period('D')

        fac_p = fac.sort_values(['ts_code', 'trade_date'], inplace=False)
        stk_p = stk.sort_values(['ts_code', 'trade_date'], inplace=False)

        # 计算下一个交易日的收益（shift 按实际交易日顺序，不会跨过非交易日）
        stk_p[next_ret_col] = stk_p.groupby('ts_code')[ret_col].shift(-1)

    # 合并因子和收益数据（用于IC计算）
    # 用 period_col 做合并键——双方通过同一 Period 对齐，避免 trade_date 细微差异导致大量丢失
    # trade_date 取因子侧的值（已是该周/月最后交易日）
    df = (
        fac_p[["ts_code", "trade_date", period_col, fac_col]]
           .merge(stk_p[["ts_code", period_col, next_ret_col, ret_col]], on=["ts_code", period_col], how="inner")
           .dropna(subset=[fac_col, next_ret_col])
           .reset_index(drop=True)
    )

    # 计算 IC & IR（基于期间收益率去计算，与下文更具日频收益率计算划分开）
    # 必须进行过滤，确保IC计算的相关系数有效。
    # 1. 计算每个交易日对应的股票数量
    counts = df.groupby('trade_date')[fac_col].transform('count')

    # 2. 仅保留数量大于 500 的行
    # df = df[counts > 500]

    # 计算每个周期最后一天的Rank IC（Spearman相关系数）
    # float() 强制每组返回标量，pandas 任何版本 groupby.apply 均只能返回 Series
    ic_series = df.groupby('trade_date').apply(
        lambda x: float(x[fac_col].corr(x[next_ret_col], method='spearman')),
        include_groups=False
    )
    ic_df = ic_series.reset_index()
    ic_df.columns = ['trade_date', 'ic']


    # 计算IR（信息比率）= IC均值 / IC标准差
    ic_valid = ic_df["ic"].dropna()
    IC = float(ic_valid.mean())
    IR = np.nan
    if len(ic_valid) != 0:
        IR = float(
            IC / ic_valid.std() ## 默认为样本标准差
            ) if ic_valid.std() > 0 else np.nan

    ## 按照当期因子值来分组（向量化分位分组，替代逐组 qcut）
    df = df.sort_values(["trade_date", "ts_code"])
    factor_rank = df.groupby("trade_date")[fac_col].rank(method='first', na_option='keep') # 每日因子值排名，method='first'确保相同因子值的股票按出现顺序排名，na_option='keep'保持NaN为NaN rank越大因子值越高
    daily_count = df.groupby("trade_date")[fac_col].transform('count')  # 每日股票数量
    group_raw   = np.ceil(factor_rank / daily_count * n_groups).clip(1, n_groups)
    df["group"] = group_raw.where(group_raw.notna(), other=np.nan).astype('Int64')

    ## --- 计算日度组合收益率，用于绘制净值曲线与计算指标LS、年化夏普 ---
    # T期末计算的因子分组，在T+1期（hold_period）持有
    group_map = df[["ts_code", period_col, "group"]].copy()
    ## 原本的是当月因子值与下月收益，也就是按当月因子值分组，对应的是当月收益，我们要让上月的分组与当月收益相匹配，只能使用时期period的编码去匹配。
    ## 现在变成hold_period = 2026-02 group = 10 （按2026-01分的）
    group_map["hold_period"] = group_map[period_col] + 1
    
    # 提取所有自带日频收益率的股票明细
    daily_stk = stk[["ts_code", "trade_date", "ret", period_col]].copy()
    
    # 将hold_period = 2026-02 group = 10 与 period = 2026-02的数据进行匹配。这样就有了2026-01的分组信息。
    daily_stk = daily_stk.merge(
        group_map[["ts_code", "hold_period", "group"]],
        left_on=["ts_code", period_col],
        right_on=["ts_code", "hold_period"],
        how="inner"
    )

    # 算术平均算出各持仓组每天的等权平均收益率
    daily_group_ret = daily_stk.groupby(["trade_date", "group"], as_index=False).agg(daily_ret=("ret", "mean"))
    
    # 计算每一天的多头和空头收益并合成全市场日度的多空收益序列
    long_daily = daily_group_ret[daily_group_ret["group"] == n_groups][["trade_date", "daily_ret"]].rename(columns={"daily_ret": "long"})
    short_daily = daily_group_ret[daily_group_ret["group"] == 1][["trade_date", "daily_ret"]].rename(columns={"daily_ret": "short"})
    
    daily_ls = long_daily.merge(short_daily, on="trade_date", how="inner")
    daily_ls["ls_ret"] = daily_ls["long"] - daily_ls["short"]
    daily_ls = daily_ls.sort_values("trade_date").reset_index(drop=True)

    # Sharpe Ratio (基于日度超额收益计算，年化需乘上根号下242，此处保持你前文直接除的风格或者可调整)
    ls_valid = daily_ls["ls_ret"].dropna()
    SR = float(ls_valid.mean() / ls_valid.std() * np.sqrt(242)) if ls_valid.std() > 0 else np.nan

    # 计算各分组年化收益率（跨时间用几何平均年化，同一时间截面已用算术平均合并为 daily_ret）
    # 以 % 形式显示，如 15.23 表示 15.23%
    group_stats = daily_group_ret.groupby('group', as_index=False).agg(
        group_ret=('daily_ret', lambda x: ((1 + x).prod() ** (242 / len(x)) - 1) * 100)
    )

    daily_ls = daily_ls.merge(bench_df, on=['trade_date'], how='inner')

    # 组装为一个扁平的字典，便于最后在 joblib 中直接 pd.DataFrame(results) 组合成一张大表
    result_dict = {
        '样本': group_set[0],
        '频率': group_set[1],
        'IC': round(IC, 2),
        'IR': round(IR, 2),
        'SR': round(SR, 2),
        'daily_ls': daily_ls[['trade_date', 'ls_ret', 'long', 'short', 'bench_ret']], # 同时存储多空收益和纯多头、纯空头收益
        'ic_series': ic_df[['trade_date', 'ic']]
    }

    # 将各分组平均收益也加入上述字典
    for g, val in zip(group_stats["group"], group_stats["group_ret"]):
        result_dict[f"Group_{g}_ret"] = f"{round(float(val), 2)}%"
        
    return result_dict

def evaluate_factor(factor_table, fac_freq, bench_index='000002.SH', other_name=None, samples=None, n_groups=10):
    """
    因子评估主流程
    
    算法逻辑：
    1. 读取因子数据（从DuckDB表）
    2. 计算各种设定组合（样本、频率）下的分组收益、多空收益、IC等指标
    3. 输出评价结果至CSV
    
    Args:
        factor_table(str): 因子表名（DuckDB中的表名，如'week_factor'）
        fac_freq(str): 因子的频率
        bench_index(str): 基准指数代码（默认'000002.SH'，全A指数）
        other_name(str): 因子列的原始名称（默认None，如果提供则重命名为'factor'）
        samples(list|str|None): 限定回测的样本，如 '全市场' 或 ['全市场','中证1000']，默认None表示全部
        n_groups(int): 分组组数（默认10组）
    """    
    try:
        fac = db_utils.read_sql(f"SELECT * FROM {factor_table}")
        if other_name:
            fac.rename(columns={other_name: "factor"}, inplace=True)

        # trade_date可能是整数，需要转为字符串后再转为日期
        fac["date"] = pd.to_datetime(fac["trade_date"].astype(str), format="%Y%m%d", errors="coerce")

    except Exception as e:
        print(f'错误：无法读取因子表 {factor_table}: {e}')
        return None
    
    stk = db_utils.read_sql("SELECT ts_code, trade_date, pct_chg FROM stock_bar")
    stk["date"] = pd.to_datetime(stk["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    stk["ret"] = stk["pct_chg"] / 100.0
    stk["ret"] = winsorize(stk["ret"], n=3)

    ## 根据因子的生成频率，过滤掉不合理的调仓频率
    # 比如因子如果是周频(W)或月频(M)，就没办法进行日度调仓，因此回测调仓频率必须大于等于因子频率。
    # 我们根据常见频率对应的时间跨度定一个简单的优先级来筛选
    freq_priority = {'日度': 1, '周度': 2, '月度': 3, '季度': 4}
    valid_freqs = {
        name: code for name, code in FREQ_MAP.items() 
        if freq_priority.get(name, 99) >= freq_priority.get(fac_freq, 0)
    }

    # 根据 samples 参数限定回测样本范围
    if samples is not None:
        sample_filter = [samples] if isinstance(samples, str) else list(samples)
        active_samples = {k: v for k, v in INDEX_NAME_TO_CODE.items() if k in sample_filter}
    else:
        active_samples = INDEX_NAME_TO_CODE

    ## 每种设定都回归一下
    # 直接进行笛卡尔积，且只使用过滤后的合理频率
    combinations = list(itertools.product(active_samples, valid_freqs))

    # 预计算各样本成分股集合并预过滤数据，避免在并行 worker 内重复 I/O 和大对象序列化
    # 日收益字典，key为样本名称，value为对应的股票日收益数据（已过滤成分股）
    stk_sorted = stk.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    sample_fac: dict = {}
    sample_stk: dict = {}

    for sample_name, index_code in active_samples.items():
        if sample_name == '全市场':
            sample_fac[sample_name] = fac.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
            sample_stk[sample_name] = stk_sorted
        else:
            ins_path = Path(f'./Data/Metadata/{index_code}_ins.csv')
            constituent_stocks = set(pd.read_csv(ins_path)['con_code'].unique()) if ins_path.exists() else get_ins(index_code)
            sample_fac[sample_name] = fac[fac['ts_code'].isin(constituent_stocks)].sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
            sample_stk[sample_name] = stk_sorted[stk_sorted['ts_code'].isin(constituent_stocks)].reset_index(drop=True)

    bench_raw = get_index_data(bench_index)
    bench_raw['bench_ret'] = bench_raw['pct_chg'] / 100.0
    bench_df = bench_raw[['trade_date', 'bench_ret']].copy()

    print("开始评估因子在不同样本和频率下的表现...")
    results = Parallel(n_jobs=-1)(
        delayed(specific_group)(sample_fac[combo[0]], sample_stk[combo[0]], combo, bench_df, n_groups)
        for combo in combinations
    )
    results = pd.DataFrame(results)
    
    #确保回测的根目录存在
    output_path = Path("factor_mining") / factor_table / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 提取daily_ls数据并写入数据库
    daily_ls_list = []
    for idx, row in results.iterrows():
        df_daily = row['daily_ls'].copy()
        df_daily['样本'] = row['样本']
        df_daily['频率'] = row['频率']
        daily_ls_list.append(df_daily)
    
    daily_ls_combined = pd.concat(daily_ls_list, ignore_index=True)
    db_utils.write_to_db(daily_ls_combined, f'{factor_table}_daily_ls', save_mode='replace')

    # 提取 ic_series 数据并写入数据库
    ic_series_list = []
    for idx, row in results.iterrows():
        df_ic = row['ic_series'].copy()
        df_ic['样本'] = row['样本']
        df_ic['频率'] = row['频率']
        ic_series_list.append(df_ic)

    ic_series_combined = pd.concat(ic_series_list, ignore_index=True)
    db_utils.write_to_db(ic_series_combined, f'{factor_table}_ic_series', save_mode='replace')

    # 提取除daily_ls、ic_series之外的指标，并导出为csv
    summary = results.drop(columns=['daily_ls', 'ic_series'])

    # 导出csv
    summary.to_csv(f"{output_path}/summary.csv", index=False, encoding='utf-8-sig')

    print(f'因子评估完成，结果已保存到 {output_path}/')
    
def group_plot(sample, freq, line, factor_table, mode='evaluate'):
    """绘制特定样本和频率的多空收益净值曲线及IC序列

    Args:
        mode: 'evaluate' 读 {factor_table}_daily_ls，含 IC 子图；
              'trade'    读 {factor_table}_trade_daily_ret，不含 IC 子图
    """
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    try:
        tbl_name = f"{factor_table}_trade_daily_ret" if mode == 'trade' else f"{factor_table}_daily_ls"

        df = db_utils.read_sql(f"""
            SELECT trade_date, {line}, bench_ret FROM {tbl_name}
            WHERE 样本 = '{sample}' AND 频率 = '{freq}'
            ORDER BY trade_date
        """)
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
        df[line] = winsorize(df[line], n=3)
        df['net_value'] = (1 + df[line]).cumprod()
        df['net_value_bench'] = (1 + df['bench_ret']).cumprod()

        # --- 计算绩效指标 ---
        line_ret = df[line].dropna()
        n = len(line_ret)
        net_value = df['net_value'].reindex(line_ret.index)

        ann_ret         = net_value.iloc[-1] ** (242 / n) - 1
        bench_r         = df['bench_ret'].reindex(line_ret.index)
        net_value_bench = df['net_value_bench'].reindex(line_ret.index)
        ann_bench       = net_value_bench.iloc[-1] ** (242 / len(bench_r)) - 1
        excess_ann      = ann_ret - ann_bench
        vol             = line_ret.std() * np.sqrt(242)
        max_dd          = - ((net_value - net_value.cummax()) / net_value.cummax()).min()

        excess_ann_str = f"{round(excess_ann * 100, 2)}%"
        vol_str        = f"{round(vol * 100, 2)}%"
        max_dd_str     = f"{round(max_dd * 100, 2)}%"

        if mode == 'trade':
            fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 7),
                                           gridspec_kw={'height_ratios': [2, 0.25]})
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 11),
                                                gridspec_kw={'height_ratios': [1.5, 1, 0.25]})

        # 上图：净值曲线，起始为 1；副坐标轴显示策略/基准相对净值
        ax1.plot(df['trade_date'], df['net_value'], label=f'{sample} - {freq}', color='blue')
        ax1.plot(df['trade_date'], df['net_value_bench'], label='基准净值', color='red')
        ax1.set_title(f'{sample} - {freq} {line}净值曲线')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('净值')
        ax1.legend(loc='upper left')
        ax1.grid()

        ax1_r = ax1.twinx()
        relative_nav = df['net_value'] / df['net_value_bench']
        ax1_r.plot(df['trade_date'], relative_nav, label='策略/基准', color='green', linewidth=1)
        ax1_r.set_ylabel('相对净值')
        ax1_r.legend(loc='upper right')

        if mode != 'trade':
            ic_df = db_utils.read_sql(f"""
                SELECT trade_date, ic FROM {factor_table}_ic_series
                WHERE 样本 = '{sample}' AND 频率 = '{freq}'
                ORDER BY trade_date
            """)
            ic_df['trade_date'] = pd.to_datetime(ic_df['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
            ic_mean = ic_df['ic'].mean()
            colors = ['red' if v >= 0 else 'green' for v in ic_df['ic']]
            ax2.bar(ic_df['trade_date'], ic_df['ic'], color=colors, width=15, label='IC')
            ax2.axhline(ic_mean, color='black', linestyle='--', linewidth=1, label=f'均值 {ic_mean:.3f}')
            ax2.axhline(0, color='gray', linestyle='-', linewidth=0.8)
            ax2.set_title(f'{sample} - {freq} IC序列')
            ax2.set_xlabel('日期')
            ax2.set_ylabel('IC')
            ax2.legend()
            ax2.grid(axis='y')

        # 底部：绩效指标表格
        ax3.axis('off')
        tbl = ax3.table(
            cellText=[[excess_ann_str, vol_str, max_dd_str]],
            colLabels=['超额年化收益', '波动率', '最大回撤'],
            loc='center',
            cellLoc='center',
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1, 1.8)

        plt.tight_layout()

        output_path = Path("factor_mining") / factor_table / "output"
        output_path.mkdir(parents=True, exist_ok=True)

        plt.savefig(output_path / f'{sample}_{freq}_{line}_{mode}_curve.png')
        plt.close()

    except Exception as e:
        print(f'错误：无法绘制{sample} - {freq}的{line}收益曲线: {e}')




