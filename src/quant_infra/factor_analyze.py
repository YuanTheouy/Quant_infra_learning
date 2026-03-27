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
from tqdm import tqdm

def calc_ic(df, fac_col, ret_col):
    """
    计算因子的IC
    Args:
        df: 合并因子和收益后的因子数据 DataFrame
        fac_col: 因子列名
        ret_col: 收益列名
    Returns:
        IC DataFrame，列名为 ['trade_date', 'ic']
    """
    # 1. [关键步] 转换成长表：日期为行，股票为列
    # 这一步 Pandas 会在底层一次性完成索引重排，速度极快
    fac_wide = df.pivot(index='trade_date', columns='ts_code', values=fac_col)
    ret_wide = df.pivot(index='trade_date', columns='ts_code', values=ret_col)
    
    # 2. 向量化计算排名 (Spearman 的本质)
    # axis=1 表示在每一天（行）内部进行排名，pct=True 转化为百分比排名
    fac_rank = fac_wide.rank(axis=1, pct=True)
    ret_rank = ret_wide.rank(axis=1, pct=True)
    
    # 3. 向量化计算 Pearson 相关系数 (排名后的 Pearson 就是 Spearman)
    # corrwith 会自动对齐日期，按行计算两个矩阵的相关性
    ic_series = fac_rank.corrwith(ret_rank, axis=1)
    
    # 4. 整理结果
    # 计算每一行非空值的数量（即每天有多少只股票参与计算）
    stock_counts = fac_wide.notnull().sum(axis=1)
    # 直接过滤 IC 序列
    ic_series = ic_series[stock_counts > MIN_IC_SIZE]

    ic_df = ic_series.reset_index()
    ic_df.columns = ['trade_date', 'ic']

    return ic_df

    
def specific_group(fac, stk, group_set, bench_df, n_groups=10, pathway_delay=0, trade_days=None):
    """
    对特定分组进行评估，后续使用Joblib并行回测
    不同频率下，只是简单地取当期因子值的最后一个值进行分组调仓。
    Args:
        fac: 已按样本过滤并排序的因子数据 ['date','factor','ts_code','trade_date']
        stk: 已按样本过滤并排序的股票数据 ['date','ret','ts_code','trade_date']
        group_set: 分组集合设定，形如('全市场','日度')
        bench_df: 基准指数日收益 DataFrame，列为 ['trade_date', 'bench_ret']
        pathway_delay: 轨道延迟量（交易日数），默认0表示不延迟
        trade_days: 全部交易日的有序数组，pathway_delay>0时必须提供
        
    Returns:
        分组收益、多空收益、IC、IR、Sharpe
    """
    # fac/stk 已在 evaluate_factor 中按样本过滤并预排序，此处无需重复操作

    # 轨道延迟：将日期前移 pathway_delay 个交易日后再划分周期，等效于将调仓边界后移
    if pathway_delay > 0 and trade_days is not None:
        # 通过改写日期实现推迟调仓。例如：
        # trade_days = [3-01, 3-02, 3-03], delay = 1
        # idx_arr = [0, 0, 1], _shift_map = {3-01:3-01, 3-02:3-01, 3-03:3-02}
        # 结果：原本3-02的数据伪装成了3-01，原本3-03的伪装成了3-02。
        # 后续按月或周划分Period时，原本在3-01触发的边界会因为日期“名义上前移”而推迟到3-02才触发。
        n_td = len(trade_days)
        idx_arr = np.clip(np.arange(n_td) - pathway_delay, 0, n_td - 1) 
        _shift_map = pd.Series(trade_days[idx_arr], index=pd.DatetimeIndex(trade_days))
        fac_pdate = fac['date'].map(_shift_map)  ## 直接修改日期列，这样fac与stk的日期都被统一偏移了，后续的周期划分自然就后移了
        stk_pdate = stk['date'].map(_shift_map)
    else:
        fac_pdate = fac['date']
        stk_pdate = stk['date']

    # 非日频的收益率要提前计算。这是为了计算后面的IC，必须对其因子和股票的频率。如果是为了画收益率图，不用这部分
    if group_set[1] != '日度':
        # 1. 获取对应的 Pandas 频率缩写（如 'W', 'M'）
        p_freq = FREQ_MAP.get(group_set[1])
        if not p_freq:
            raise ValueError(f"不支持的频率: {group_set[1]}")

        # 2. 生成周期标签列
        period_col = f"date_{p_freq}"
        ret_col = f"ret_{p_freq}"
        fac_col = f"factor_{p_freq}"
        stk[period_col] = stk_pdate.dt.to_period(p_freq)
        fac[period_col] = fac_pdate.dt.to_period(p_freq)

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

        stk[period_col] = stk_pdate.dt.to_period('D')
        fac[period_col] = fac_pdate.dt.to_period('D')

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
    # 引入一个函数，专门计算。防止pandas的apply过慢。我们直接在DataFrame上进行向量化计算。
    ic_df = calc_ic(df, fac_col, next_ret_col)

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
        '轨道': pathway_delay,
        '分组IC': round(IC, 2),
        '分组IR': round(IR, 2),
        '多空夏普比率': round(SR, 2),
        'daily_ls': daily_ls[['trade_date', 'ls_ret', 'long', 'short', 'bench_ret']], # 同时存储多空收益和纯多头、纯空头收益
        'ic_series': ic_df[['trade_date', 'ic']]
    }

    # 将各分组平均收益也加入上述字典
    for g, val in zip(group_stats["group"], group_stats["group_ret"]):
        result_dict[f"Group_{g}_ret"] = f"{round(float(val), 2)}%"
        
    return result_dict

def _prepare_evaluate_data(factor_table, fac_freq, bench_index='000002.SH', other_name=None, samples=None, freq=None):
    """因子评估数据预处理（加载因子/行情/基准，过滤样本与频率）

    Args:
        freq(str|None): 若指定，combinations 直接为 [(sample, freq)] 单元组，不再展开多频率笛卡尔积

    Returns:
        tuple: (sample_fac, sample_stk, bench_df, combinations, trade_days) 或 None
    """
    try:
        fac = db_utils.read_sql(f"SELECT * FROM {factor_table}")
        if other_name:
            fac.rename(columns={other_name: "factor"}, inplace=True)
        fac["date"] = pd.to_datetime(fac["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    except Exception as e:
        print(f'错误：无法读取因子表 {factor_table}: {e}')
        return None

    stk = db_utils.read_sql("SELECT ts_code, trade_date, pct_chg FROM stock_bar")
    stk["date"] = pd.to_datetime(stk["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    stk["ret"] = stk["pct_chg"] / 100.0
    stk["ret"] = winsorize(stk["ret"], n=N_SIGMAS)

    freq_priority = {'日度': 1, '周度': 2, '月度': 3}
    valid_freqs = {
        name: code for name, code in FREQ_MAP.items()
        if freq_priority.get(name, 99) >= freq_priority.get(fac_freq, 0)
    }

    if samples is not None:
        sample_filter = [samples] if isinstance(samples, str) else list(samples)
        active_samples = {k: v for k, v in INDEX_NAME_TO_CODE.items() if k in sample_filter}
    else:
        active_samples = INDEX_NAME_TO_CODE

    if freq is not None:
        # 直接固定为单一 (sample, freq) 组合，跳过笛卡尔积
        combinations = [(s, freq) for s in active_samples]
    else:
        combinations = list(itertools.product(active_samples, valid_freqs))

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

    trade_days = np.sort(stk['date'].dropna().unique())

    return sample_fac, sample_stk, bench_df, combinations, trade_days


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
    prepared = _prepare_evaluate_data(factor_table, fac_freq, bench_index, other_name, samples)
    if prepared is None:
        return None
    sample_fac, sample_stk, bench_df, combinations, _ = prepared

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

    # 提取除daily_ls、ic_series、轨道之外的指标，并导出为csv
    summary = results.drop(columns=['daily_ls', 'ic_series', '轨道'])

    # 导出csv
    summary.to_csv(f"{output_path}/summary.csv", index=False, encoding='utf-8-sig')

    print(f'因子评估完成，结果已保存到 {output_path}/')


def evaluate_factor_pathways(factor_table, fac_freq, n_pathways, line, sample='全市场', bench_index='000002.SH', other_name=None, n_groups=10):
    """
    多轨道因子评估 - 在不同建仓日期偏移下评估因子的稳健性

    通过将周期划分边界逐日偏移，模拟从不同交易日开始建仓的效果。
    每条轨道复用 specific_group 的完整评估逻辑，最终聚合各轨道指标。

    Args:
        factor_table(str): 因子表名
        fac_freq(str): 因子频率（如 '月度'）
        n_pathways(int): 轨道数量（如 20 表示偏移 0~19 个交易日）
        sample(str): 回测样本，如 '全市场'、'中证1000'
        bench_index(str): 基准指数代码
        other_name(str): 因子列原始名称
        n_groups(int): 分组组数

    Returns:
        pd.DataFrame: pathway_summary
    """
    prepared = _prepare_evaluate_data(factor_table, fac_freq, bench_index, other_name, samples=sample, freq=fac_freq)
    if prepared is None:
        return None
    sample_fac, sample_stk, bench_df, combinations, trade_days = prepared

    combo = combinations[0]  # 由 _prepare_evaluate_data 保证此处只有单个 (sample, fac_freq)
    all_tasks = [(combo, delay) for delay in range(n_pathways)]

    print(f"开始多轨道因子评估（样本={sample}，频率={fac_freq}，{n_pathways} 条轨道）...")
    raw_results = Parallel(n_jobs=-1)(
        delayed(specific_group)(
            sample_fac[sample], sample_stk[sample], combo, bench_df, n_groups,
            pathway_delay=delay, trade_days=trade_days
        )
        for combo, delay in tqdm(all_tasks)
    )
    all_df = pd.DataFrame(raw_results)

    output_path = Path("factor_mining") / factor_table / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    # 写入 daily_ls（含轨道编号）
    daily_ls_list = []
    for _, row in all_df.iterrows():
        df_daily = row['daily_ls'].copy()
        df_daily['样本'] = row['样本']
        df_daily['频率'] = row['频率']
        df_daily['轨道'] = row['轨道']
        daily_ls_list.append(df_daily)
    db_utils.write_to_db(pd.concat(daily_ls_list, ignore_index=True),
                         f'{factor_table}_pathway_daily_ls', save_mode='replace')

    # 写入 ic_series（含轨道编号）
    ic_series_list = []
    for _, row in all_df.iterrows():
        df_ic = row['ic_series'].copy()
        df_ic['样本'] = row['样本']
        df_ic['频率'] = row['频率']
        df_ic['轨道'] = row['轨道']
        ic_series_list.append(df_ic)
    db_utils.write_to_db(pd.concat(ic_series_list, ignore_index=True),
                         f'{factor_table}_pathway_ic_series', save_mode='replace')

    summary = all_df.drop(columns=['daily_ls', 'ic_series'])
    summary.to_csv(output_path / "pathway_summary.csv", index=False, encoding='utf-8-sig')
    print(f'多轨道因子评估完成，结果已保存到 {output_path}/pathway_summary.csv')

    pathway_plot(sample, fac_freq, line, factor_table)

    return summary

    
def group_plot(sample, freq, line, factor_table, mode='evaluate', pathway=None):
    """绘制特定样本和频率的多空收益净值曲线

    Args:
        mode: 'evaluate' 读 {factor_table}_daily_ls；
              'trade'    读 {factor_table}_trade_daily_ret；
              'pathway'  读 {factor_table}_pathway_daily_ls（需同时指定 pathway 参数）
        pathway(int|None): 轨道编号，mode='pathway' 时生效
    """
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    try:
        if mode == 'pathway':
            tbl_name = f"{factor_table}_pathway_daily_ls"
            where_clause = f"样本 = '{sample}' AND 频率 = '{freq}' AND 轨道 = {pathway}"
        elif mode == 'trade':
            tbl_name = f"{factor_table}_trade_daily_ret"
            where_clause = f"样本 = '{sample}' AND 频率 = '{freq}'"
        else:
            tbl_name = f"{factor_table}_daily_ls"
            where_clause = f"样本 = '{sample}' AND 频率 = '{freq}'"

        df = db_utils.read_sql(f"""
            SELECT trade_date, {line}, bench_ret FROM {tbl_name}
            WHERE {where_clause}
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
        calmar         = excess_ann / max_dd if max_dd > 0 else np.nan

        excess_ann_str = f"{round(excess_ann * 100, 2)}%"
        vol_str        = f"{round(vol * 100, 2)}%"
        max_dd_str     = f"{round(max_dd * 100, 2)}%"
        calmar_str     = f"{round(calmar, 2)}" if not np.isnan(calmar) else "N/A"

        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 7),
                                       gridspec_kw={'height_ratios': [2, 0.25]})

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

        # 底部：绩效指标表格
        ax3.axis('off')
        tbl = ax3.table(
            cellText=[[excess_ann_str, vol_str, max_dd_str, calmar_str]],
            colLabels=['超额年化收益', '波动率', '最大回撤', 'Calmar比率'],
            loc='center',
            cellLoc='center',
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1, 1.8)

        plt.tight_layout()

        output_path = Path("factor_mining") / factor_table / "output"
        output_path.mkdir(parents=True, exist_ok=True)

        suffix = f'_pathway{pathway}' if mode == 'pathway' else ''
        plt.savefig(output_path / f'{sample}_{freq}_{line}_{mode}{suffix}_curve.png')
        plt.close()

    except Exception as e:
        print(f'错误：无法绘制{sample} - {freq}的{line}收益曲线: {e}')


def pathway_plot(sample, freq, line, factor_table):
    """绘制多轨道最优/平均/最差累积超额收益对比图，并标注最优与最差轨道编号"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    try:
        df = db_utils.read_sql(f"""
            SELECT trade_date, {line}, bench_ret, 轨道
            FROM {factor_table}_pathway_daily_ls
            WHERE 样本 = '{sample}' AND 频率 = '{freq}'
            ORDER BY trade_date
        """)
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
        df['excess'] = df[line] - df['bench_ret']

        # 每条轨道的累计超额收益：(1 + excess).cumprod() - 1
        excess_wide = df.pivot_table(index='trade_date', columns='轨道', values='excess')
        cum_excess = (1 + excess_wide).cumprod() - 1

        final_vals = cum_excess.iloc[-1]
        best_pathway  = int(final_vals.idxmax())
        worst_pathway = int(final_vals.idxmin())
        avg_cum = cum_excess.mean(axis=1)

        fig, ax1 = plt.subplots(figsize=(13, 6))

        # 最差到最优的填充区间
        ax1.fill_between(cum_excess.index,
                         cum_excess[worst_pathway], cum_excess[best_pathway],
                         alpha=0.12, color='gray')

        ax1.plot(cum_excess.index, cum_excess[best_pathway],
                 color='#333333', linewidth=1.2, label=f'最优情形累计超额收益（轨道 {best_pathway}）')
        ax1.plot(cum_excess.index, avg_cum,
                 color='red', linewidth=2.0, label='平均累计超额收益')
        ax1.plot(cum_excess.index, cum_excess[worst_pathway],
                 color='#AAAAAA', linewidth=1.2, label=f'最差情形累计超额收益（轨道 {worst_pathway}）')

        # 标注最终累计超额收益（转为百分比）
        last_date = cum_excess.index[-1]
        ax1.annotate(f'最优: {cum_excess[best_pathway].iloc[-1]*100:.1f}%',
                     xy=(last_date, cum_excess[best_pathway].iloc[-1]),
                     xytext=(-60, 8), textcoords='offset points',
                     fontsize=9, color='#333333')
        ax1.annotate(f'最差: {cum_excess[worst_pathway].iloc[-1]*100:.1f}%',
                     xy=(last_date, cum_excess[worst_pathway].iloc[-1]),
                     xytext=(-60, -14), textcoords='offset points',
                     fontsize=9, color='#AAAAAA')

        ax1.axhline(0, color='gray', linewidth=0.8)
        ax1.set_title(f'{sample} - {freq}  多轨道累计超额收益（{line}）')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('累计超额收益')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v*100:.0f}%'))
        ax1.legend(loc='upper left')
        ax1.grid(axis='y', alpha=0.4)

        plt.tight_layout()

        output_path = Path("factor_mining") / factor_table / "output"
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path / f'{sample}_{freq}_{line}_pathway_curve.png')
        plt.close()

    except Exception as e:
        print(f'错误：无法绘制{sample} - {freq}的轨道对比图: {e}')


def ic_plot(sample, freq, factor_table):
    """绘制特定样本和频率的IC序列柱状图及IC累积值曲线"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    try:
        ic_df = db_utils.read_sql(f"""
            SELECT trade_date, ic FROM {factor_table}_ic_series
            WHERE 样本 = '{sample}' AND 频率 = '{freq}'
            ORDER BY trade_date
        """)
        ic_df['trade_date'] = pd.to_datetime(ic_df['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
        ic_mean = ic_df['ic'].mean()
        ic_df['ic_cumsum'] = ic_df['ic'].cumsum()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [1.5, 1]})

        colors = ['red' if v >= 0 else 'green' for v in ic_df['ic']]
        ax1.bar(ic_df['trade_date'], ic_df['ic'], color=colors, width=15, label='IC')
        ax1.axhline(ic_mean, color='black', linestyle='--', linewidth=1, label=f'均值 {ic_mean:.3f}')
        ax1.axhline(0, color='gray', linestyle='-', linewidth=0.8)
        ax1.set_title(f'{sample} - {freq} IC序列')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('IC')
        ax1.legend()
        ax1.grid(axis='y')

        ax2.plot(ic_df['trade_date'], ic_df['ic_cumsum'], color='blue', linewidth=1.2, label='IC累积值')
        ax2.axhline(0, color='gray', linestyle='-', linewidth=0.8)
        ax2.set_title(f'{sample} - {freq} IC累积值')
        ax2.set_xlabel('日期')
        ax2.set_ylabel('IC累积值')
        ax2.legend()
        ax2.grid(axis='y')

        plt.tight_layout()

        output_path = Path("factor_mining") / factor_table / "output"
        output_path.mkdir(parents=True, exist_ok=True)

        plt.savefig(output_path / f'{sample}_{freq}_ic.png')
        plt.close()

    except Exception as e:
        print(f'错误：无法绘制{sample} - {freq}的IC序列: {e}')


def _ic_lag_worker(fac_rank, ret_wide, valid_dates, lag):
    """ic_half_life 的 joblib 并行工作单元：计算单个滞后期的 Rank IC 均值"""
    ret_shifted = ret_wide.shift(-lag).loc[valid_dates]
    ret_rank = ret_shifted.rank(axis=1, pct=True)
    return lag, float(fac_rank.corrwith(ret_rank, axis=1).mean())


def ic_half_life(factor_table, max_lag=200, other_name=None):
    """
    计算日频因子在全样本下不同滞后期的 Rank IC，绘制 IC 衰减图并标注半衰期。

    SQL 层只完成因子与股票日收益的合并，各 lag 的未来收益通过 pandas shift(-lag)
    在宽表上 O(1) 完成，避免在 SQL 侧生成 max_lag 个 LEAD 列导致内存膨胀。

    Args:
        factor_table (str): 因子表名（DuckDB 中的表名）
        max_lag (int): 最大滞后期数，默认 200 个交易日
        other_name (str|None): 因子列的原始名称，若非 'factor' 则在此指定

    Returns:
        pd.DataFrame: columns=['lag', 'ic']，各滞后期 Rank IC 均值
    """
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. SQL 合并因子与股票日收益（一次 I/O，无额外 lag 列）
    fac_col = other_name if other_name else 'factor'
    sql = f"""
        SELECT
            f.ts_code,
            f.trade_date,
            f.{fac_col} AS factor,
            s.pct_chg / 100.0 AS ret
        FROM {factor_table} f
        JOIN stock_bar s USING (ts_code, trade_date)
    """
    df = db_utils.read_sql(sql)

    # 2. 转宽表；因子排名只算一次，收益宽表保留完整以支持 shift
    fac_wide = df.pivot(index='trade_date', columns='ts_code', values='factor')
    ret_wide = df.pivot(index='trade_date', columns='ts_code', values='ret')

    valid_dates = fac_wide.notnull().sum(axis=1).pipe(lambda s: s[s > 500].index)
    fac_rank = fac_wide.loc[valid_dates].rank(axis=1, pct=True)

    # 3. 并行计算各 lag 的 Rank IC 均值，tqdm 显示完成进度
    raw = Parallel(n_jobs=-1, return_as='generator_unordered')(
        delayed(_ic_lag_worker)(fac_rank, ret_wide, valid_dates, lag)
        for lag in range(1, max_lag + 1)
    )
    ic_list = [{'lag': lag, 'ic': ic}
               for lag, ic in tqdm(raw, total=max_lag, desc='计算 IC 衰减')]
    ic_list.sort(key=lambda x: x['lag'])  # 恢复 lag 顺序（并行结果乱序）

    ic_decay_df = pd.DataFrame(ic_list)

    # 4. 半衰期：首次 |IC| ≤ |IC(lag=1)| / 2 的滞后期
    ic_1 = ic_decay_df.loc[0, 'ic']
    half_life = None
    if not (np.isnan(ic_1) or ic_1 == 0):
        mask = ic_decay_df['ic'].abs() <= abs(ic_1) / 2
        if mask.any():
            half_life = int(ic_decay_df.loc[mask.idxmax(), 'lag'])

    # 5. 绘图
    fig, ax = plt.subplots(figsize=(12, 5))
    ic_pct = ic_decay_df['ic'] * 100

    ax.plot(ic_decay_df['lag'], ic_pct, color='steelblue', linewidth=1.5, label='Rank IC 均值')
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.8)

    # 半值水平基准线
    half_val_pct = ic_1 / 2 * 100
    ax.axhline(half_val_pct, color='orange', linestyle='--', linewidth=1,
               label=f'0.5IC = {half_val_pct:.2f}%')

    # 半衰期竖线 + 标注
    if half_life is not None:
        hl_ic_pct = float(ic_decay_df.loc[ic_decay_df['lag'] == half_life, 'ic'].values[0]) * 100
        ax.axvline(half_life, color='red', linestyle='--', linewidth=1.2)
        y_offset = max(abs(ic_1 * 100) * 0.15, 0.2) * np.sign(ic_1)
        ax.annotate(
            f'半衰期 = {half_life} 期',
            xy=(half_life, hl_ic_pct),
            xytext=(half_life + max(1, max_lag // 12), hl_ic_pct + y_offset),
            fontsize=10,
            color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
        )
        print(f'IC 半衰期（首次降至初始 IC 一半）：{half_life} 期')
    else:
        print(f'在 max_lag={max_lag} 范围内未找到 IC 半衰期，可适当增大 max_lag')

    ax.set_title(f'Rank IC 衰减图  [{factor_table}  全市场  日频]')
    ax.set_xlabel('滞后期（交易日）')
    ax.set_ylabel('Rank IC 均值')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax.set_xticks(range(0, max_lag + 1, 10))
    ax.set_xticklabels(range(0, max_lag + 1, 10), rotation=90)
    ax.legend()
    ax.grid(axis='y', alpha=0.5)
    plt.tight_layout()

    output_path = Path("factor_mining") / factor_table / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / 'ic_half_life.png'
    plt.savefig(save_path)
    plt.close()
    print(f'IC 衰减图已保存至 {save_path}')

    return


