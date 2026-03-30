# -*- coding: utf-8 -*-
"""
交易模拟模块：更真实地模拟小资金交易
- 增加手续费 + 滑点（默认单边滑点 0.1%，手续费万 2.5）
- 过滤创业板 / 科创板 / 北交所 及高价股（收盘价 > 100 元）
- 导出每个换仓时间点的持仓记录 → output/trade_holdings.csv
- 导出日频净值（兼容 group_plot，line='long'）→ {factor_table}_trade_daily_ret
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from quant_infra import db_utils
from quant_infra.const import *
from quant_infra.get_data import get_ins, get_index_data
from quant_infra.factor_calc import winsorize
from quant_infra.factor_analyze import group_plot

def simulate_trade(
    factor_table,
    trade_freq,
    bench_index="000002.SH",
    other_name=None,
    sample="全市场",
    n_top=5,
    is_ascending=False , # 默认以降序排列，即取因子值最高的前几个
    commission_rate=2.5,
    slippage_rate=0.1,
    price_max=100.0,
    filter_boards=("创业板", "科创板", "北交所"),
):
    """
    真实交易模拟回测，评价因子的头部选股能力

    Args:
        factor_table(str): 因子表名（DuckDB 中的表名，如 'spec_vol'）
        trade_freq(str): 因子频率（如 '月度'）
        bench_index(str): 基准指数代码（默认 '000002.SH'）
        other_name(str): 因子列原始名称（默认 None，提供则重命名为 'factor'）
        sample(str): 回测样本，如 '全市场'、'中证1000'，默认 '全市场'
        n_top(int): 每期买入股票数（默认 5）
        is_ascending(bool): 因子排序方向，False 表示因子越大越好（取前 n_top），True 表示因子越小越好（取后 n_top）
        commission_rate(float): 单边手续费率，按万分比输入（默认 2.5，即万 2.5）
        slippage_rate(float): 单边滑点率，按百分比输入（默认 0.1，即 0.1%）
        price_max(float): 股价过滤上限（默认 100 元）
        filter_boards(tuple/list/set): 交易权限过滤，指定要排除的板块。
            可选值：'创业板'（创业板，300开头）、'科创板'（科创板，68开头）、'北交所'（北交所，4/8开头）。
            默认 ('创业板', '科创板', '北交所') 过滤全部三个板块，传入空元组则不过滤。
    """
    commission_rate = commission_rate / 10000.0
    slippage_rate = slippage_rate / 100.0

    # ----- 1. 加载因子数据，并过滤高价股 -----
    try:
        fac = db_utils.read_sql(
            f"""
            SELECT fac.*
            FROM {factor_table} AS fac
            INNER JOIN stock_bar AS sb
                ON fac.ts_code = sb.ts_code
               AND fac.trade_date = sb.trade_date
            WHERE sb.close <= {price_max}
            """
        )
        if other_name:
            fac.rename(columns={other_name: "factor"}, inplace=True)
        fac["date"] = pd.to_datetime(fac["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    except Exception as e:
        print(f"错误：无法读取因子表 {factor_table}: {e}")
        return None

    # ----- 2. 加载股票日行情 -----
    stk = db_utils.read_sql("SELECT ts_code, trade_date, pct_chg FROM stock_bar")
    stk["date"] = pd.to_datetime(stk["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    stk["ret"] = stk["pct_chg"] / 100.0
    stk["ret"] = winsorize(stk["ret"], n=3)

    # 构建股票代码：名称映射字典：ts_code → name
    name_df = db_utils.read_sql("SELECT ts_code, name FROM stock_basic")
    code_to_name= dict(zip(name_df["ts_code"], name_df["name"]))

    # ----- 3. 过滤散户不可购买板块 -----
    _board_prefixes = {"创业板": "300", "科创板": "68", "北交所": ("4", "8")}
    for board in filter_boards:
        prefix = _board_prefixes.get(board)
        if prefix:
            fac = fac[~fac["ts_code"].str.startswith(prefix)]
            stk = stk[~stk["ts_code"].str.startswith(prefix)]

    # ----- 4. 生成周期列 -----
    p_freq = FREQ_MAP.get(trade_freq)
    if not p_freq:
        raise ValueError(f"不支持的频率: {trade_freq}")
    period_col = f"date_{p_freq}"

    fac[period_col] = fac["date"].dt.to_period(p_freq)
    stk[period_col] = stk["date"].dt.to_period(p_freq)

    # ----- 5. 基准收益 -----
    bench_raw = get_index_data(bench_index)
    bench_raw["bench_ret"] = bench_raw["pct_chg"] / 100.0
    bench_df = bench_raw[["trade_date", "bench_ret"]].copy()

    # ----- 6. 样本范围 -----
    if sample != "全市场":
        ## 保留样本内的股票
        index_code = INDEX_NAME_TO_CODE[sample]
        ins_path = Path(f"./Data/Metadata/{index_code}_ins.csv")
        constituent_stocks = set(pd.read_csv(ins_path)["con_code"].unique()) if ins_path.exists()  else get_ins(index_code)
        fac = fac[fac["ts_code"].isin(constituent_stocks)]
        stk = stk[stk["ts_code"].isin(constituent_stocks)]

    all_holdings = []
    print("开始生成调仓数据...")
    # ----- 7. 根据调仓频率进行聚合 -----
    fac_last = (fac.dropna(subset=["factor"])
        .sort_values(["ts_code", "trade_date"])
        .groupby(["ts_code", period_col], as_index=False) ## 不作为索引，后续按周期处理时更方便
        .agg(
            factor=("factor", "last"),
            trade_date=("trade_date", "max"),
        ))

    fac_last = fac_last.sort_values([period_col, "factor"],ascending=[True, is_ascending]).reset_index(drop=True)

    holdings = {}  # 关于 hold_period (Period):list[ts_code]的映射关系
    for period, group in fac_last.groupby(period_col):
        selected = group.head(n_top)["ts_code"].tolist()

        hold_period = period + 1
        holdings[hold_period] = selected

        all_holdings.append(
            {
                "样本": sample,
                "换仓日": int(group["trade_date"].max()),
                "持有周期": str(hold_period),
                "持仓股票": ",".join(selected),
                "持仓股票名称": ",".join(code_to_name.get(c, c) for c in selected),
                "持仓数量": len(selected),
            }
        )
    holdings_df = pd.DataFrame(all_holdings).sort_values("换仓日", ascending = False, ignore_index=True)
    output_path = Path("factor_mining") / factor_table / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    holdings_df.to_csv(output_path / "trade_holdings.csv", index=False, encoding="utf-8-sig")
    print(f"交易模拟完成，持仓记录已保存到 {output_path}/trade_holdings.csv")

    print("开始计算日度组合收益（含交易成本）...")
    # ----- 8. 计算日度组合收益（含交易成本） -----
    daily_ret = compute_portfolio_daily_ret(stk, holdings, period_col, commission_rate, slippage_rate)
    if daily_ret.empty:
        print("无有效回测结果")
        return

    daily_ret = daily_ret.merge(bench_df, on="trade_date", how="inner")
    daily_ret["样本"] = sample
    daily_ret["频率"] = trade_freq
    long_ret = daily_ret["long"].dropna()
    SR = (
        float(long_ret.mean() / long_ret.std() * np.sqrt(242))
        if long_ret.std() > 0 else np.nan
    )
    print(f"[{sample} - {trade_freq}] 多头年化夏普（含交易成本）: {round(SR, 2)}")

    db_utils.write_to_db(daily_ret, f"{factor_table}_trade_daily_ret", save_mode="replace")
    group_plot(sample=sample, freq=trade_freq, line="long", factor_table=factor_table, mode="trade")
def compute_portfolio_daily_ret(stk_df, holdings, period_col, commission_rate, slippage_rate):
    """
    按换仓周期合并日度等权收益，在每期首日扣除双边交易成本。
    利用 holdings[period-1] 直接获取上期持仓，各期独立可并行。

    Args:
        stk_df: 含 [ts_code, trade_date, ret, period_col] 的日行情 DataFrame
        holdings: {hold_period (Period): list[ts_code]}
        period_col: 周期列名
        commission_rate: 单边手续费率
        slippage_rate: 单边滑点率

    Returns:
        DataFrame[trade_date, long]（trade_date 为 int）
    """
    from joblib import Parallel, delayed

    if not holdings:
        return pd.DataFrame()

    sorted_periods = sorted(holdings.keys())
    one_way_cost = slippage_rate + commission_rate

    # stk_df 建立 (period_col, ts_code) MultiIndex 以便快速切片
    stk_indexed = stk_df.set_index([period_col, "ts_code"]).sort_index()
    def _calc_one_period(i):
        hp = sorted_periods[i]
        curr_codes = holdings[hp]
        if not curr_codes:
            return None
        curr_set = set(curr_codes)

        # 换仓成本：直接用 holdings[period-1] 获取上期持仓
        prev_period = hp - 1
        prev_codes = holdings.get(prev_period, [])
        if not prev_codes:  # 首次建仓：仅买入成本
            cost = 1.0 * one_way_cost
        else:
            ## 集合运算，相减表示前面有但后面没有的
            prev_set = set(prev_codes)
            sold = prev_set - curr_set
            bought = curr_set - prev_set
            cost = (len(sold) / len(prev_set) + len(bought) / len(curr_set)) * one_way_cost

        # 当期持仓股票的日收益（通过 MultiIndex 切片）
        try:
            period_data = stk_indexed.loc[hp]  ## 默认作用于第一个索引，即 period_col
        except KeyError:
            return None
        period_data = period_data[period_data.index.isin(curr_set)]
        if period_data.empty:
            return None

        daily_eq = (
            period_data.groupby("trade_date", as_index=False)
            .agg(group_daily_ret=("ret", "mean"))
            .sort_values("trade_date")
            .reset_index(drop=True)
        )
        daily_eq["cost"] = 0.0
        daily_eq.loc[0, "cost"] = cost
        daily_eq["long"] = daily_eq["group_daily_ret"] - daily_eq["cost"]
        return daily_eq[["trade_date", "long"]]

    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_calc_one_period)(i) for i in range(len(sorted_periods))
    )

    result_rows = [r for r in results if r is not None]
    if not result_rows:
        return pd.DataFrame()

    return pd.concat(result_rows, ignore_index=True).sort_values("trade_date").reset_index(drop=True)


